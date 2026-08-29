//! JSONL 会话存储。
//!
//! 参见 `02-session-management.html`、`12-api-contracts.html §5`。
//! 会话以 JSONL 存储，第一行为 `SessionEvent::Meta`，后续为事件行。
//! 支持崩溃恢复（截断坏尾行，V-STORE-01）与原子写入（V-STORE-02）。

use agent_contract::{Message, Session, SessionEvent, SessionMeta, SessionStatus, Usage};
use agent_error::{Error, Result};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
};
use tokio::io::AsyncWriteExt;

type SessionAppendLock = tokio::sync::Mutex<()>;

/// A user message durably staged before an active runtime is asked to accept a steer.
/// The record itself is the outbox: loaders materialize it as a normal message unless a matching
/// cancellation record was appended after the runtime explicitly rejected the steer.
pub const DURABLE_USER_MESSAGE_EVENT: &str = "durable_user_message";
pub const DURABLE_USER_MESSAGE_CANCEL_EVENT: &str = "durable_user_message_cancelled";

/// legacy name, read-only：旧 JSONL 里的事件名带 `r_code_` 前缀，加载侧继续识别，
/// 写入侧只用上面的新名。
const LEGACY_DURABLE_USER_MESSAGE_EVENT: &str = "r_code_durable_user_message";
const LEGACY_DURABLE_USER_MESSAGE_CANCEL_EVENT: &str = "r_code_durable_user_message_cancelled";

fn is_durable_user_message_event(event: &str) -> bool {
    event == DURABLE_USER_MESSAGE_EVENT || event == LEGACY_DURABLE_USER_MESSAGE_EVENT
}

fn is_durable_user_message_cancel_event(event: &str) -> bool {
    event == DURABLE_USER_MESSAGE_CANCEL_EVENT || event == LEGACY_DURABLE_USER_MESSAGE_CANCEL_EVENT
}

/// 扩容阈值：超过该数才做一次全表 retain 清理（F-perf-05）。此前每次 append
/// 都全表扫描，热路径（每事件一次）在会话较多时会互相放大。
const APPEND_LOCK_RETAIN_THRESHOLD: usize = 256;

fn append_lock_for(path: &Path) -> Arc<SessionAppendLock> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<SessionAppendLock>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().expect("session append lock registry poisoned");
    // 只在注册表膨胀到阈值时清扫一次死锁（strong_count == 0）。
    // 懒清理由容量驱动，避免每次 append 付 O(活跃会话数) 的 retain。
    if locks.len() >= APPEND_LOCK_RETAIN_THRESHOLD {
        locks.retain(|_, lock| lock.strong_count() > 0);
    }
    if let Some(lock) = locks.get(path).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(SessionAppendLock::new(()));
    locks.insert(path.to_path_buf(), Arc::downgrade(&lock));
    lock
}

/// JSONL 会话存储。
pub struct SessionStore {
    base_dir: PathBuf,
}

impl SessionStore {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.base_dir.join(format!("{id}.jsonl"))
    }

    /// 创建新会话：写入 Meta 行，返回内存视图。
    pub async fn create(&self, model: &str, provider: &str) -> Result<Session> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        let meta = SessionMeta::new(model, provider);
        let path = self.session_path(&meta.id);
        let mut line = serde_json::to_string(&SessionEvent::Meta(meta.clone()))?;
        line.push('\n');
        // 第一行用原子写
        Self::atomic_write(&path, line.as_bytes()).await?;
        Ok(Session::new(meta))
    }

    /// 追加事件到会话日志。
    pub async fn append(&self, session_id: &str, event: SessionEvent) -> Result<()> {
        self.append_batch(session_id, &[event]).await
    }

    /// Atomically append complete JSONL records under a process-wide per-session lock.
    ///
    /// A single `SessionStore` is not the only writer: the desktop command path and runtime event
    /// drain deliberately construct independent handles for the same directory. Serializing by
    /// resolved file path prevents their records from interleaving, and writing each batch with
    /// one `write_all` prevents a body/newline split from producing invalid JSONL.
    pub async fn append_batch(&self, session_id: &str, events: &[SessionEvent]) -> Result<()> {
        let path = self.session_path(session_id);
        let mut encoded = Vec::new();
        for event in events {
            serde_json::to_writer(&mut encoded, event)?;
            encoded.push(b'\n');
        }
        let append_lock = append_lock_for(&path);
        let _guard = append_lock.lock().await;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .await?;
        file.write_all(&encoded).await?;
        file.flush().await?;
        Ok(())
    }

    /// Stage an idempotent user message before handing it to an already-running model.
    pub async fn append_durable_user_message(
        &self,
        session_id: &str,
        operation_id: &str,
        message: &Message,
        mode: &str,
        queue_id: Option<&str>,
    ) -> Result<()> {
        self.append_once(
            session_id,
            operation_id,
            SessionEvent::System {
                event: DURABLE_USER_MESSAGE_EVENT.into(),
                data: serde_json::json!({
                    "operation_id": operation_id,
                    "message": message,
                    "mode": mode,
                    "queue_id": queue_id,
                }),
            },
        )
        .await
    }

    /// Cancel a staged message after the runtime explicitly declined it. The ordinary queue may
    /// then own delivery without creating two visible/history messages.
    pub async fn cancel_durable_user_message(
        &self,
        session_id: &str,
        operation_id: &str,
    ) -> Result<()> {
        self.append_once(
            session_id,
            &format!("cancel:{operation_id}"),
            SessionEvent::System {
                event: DURABLE_USER_MESSAGE_CANCEL_EVENT.into(),
                data: serde_json::json!({ "operation_id": operation_id }),
            },
        )
        .await
    }

    async fn append_once(
        &self,
        session_id: &str,
        operation_id: &str,
        event: SessionEvent,
    ) -> Result<()> {
        let path = self.session_path(session_id);
        let append_lock = append_lock_for(&path);
        let _guard = append_lock.lock().await;
        let mut existing = tokio::fs::read(&path).await.unwrap_or_default();

        // A process crash can leave only the final record incomplete. Remove that tail before an
        // idempotent retry so the file stays parseable and the complete operation can be written.
        if !existing.is_empty() && !existing.ends_with(b"\n") {
            let complete_len = existing
                .iter()
                .rposition(|byte| *byte == b'\n')
                .map_or(0, |index| index + 1);
            let file = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .await?;
            file.set_len(complete_len as u64).await?;
            existing.truncate(complete_len);
        }

        if existing
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .any(|line| {
                serde_json::from_slice::<SessionEvent>(line)
                    .ok()
                    .and_then(|event| durable_operation_id(&event).map(str::to_owned))
                    .as_deref()
                    == Some(operation_id)
            })
        {
            return Ok(());
        }

        let mut encoded = serde_json::to_vec(&event)?;
        encoded.push(b'\n');
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .await?;
        file.write_all(&encoded).await?;
        file.flush().await?;
        Ok(())
    }

    /// 读取完整会话，重建内存视图。
    pub async fn load(&self, session_id: &str) -> Result<Session> {
        let path = self.session_path(session_id);
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| Error::SessionNotFound(session_id.to_string()))?;

        let events = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str::<SessionEvent>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let cancelled_operations = events
            .iter()
            .filter_map(cancelled_durable_operation_id)
            .collect::<std::collections::HashSet<_>>();
        let mut materialized_operations = std::collections::HashSet::new();
        let mut meta: Option<SessionMeta> = None;
        let mut messages: Vec<Message> = Vec::new();
        let mut model_projection: Option<Vec<Message>> = None;
        let mut usage = Usage::default();
        let mut tool_calls = 0u32;

        for event in events {
            match event {
                SessionEvent::Meta(m) => meta = Some(m),
                SessionEvent::Message(msg) => {
                    messages.push(msg.clone());
                    if let Some(projection) = model_projection.as_mut() {
                        projection.push(msg);
                    }
                }
                // 快照包含运行时的完整 Message 工作集（包括 ToolUse / ToolResult）。
                // 后续新追加的 Message 仍需保留，因此替换的是此前已重建的前缀而非
                // 直接 return。
                SessionEvent::HistorySnapshot { messages: snapshot } => {
                    messages = snapshot;
                    // 快照属于一个更新的 canonical revision；紧随其后的显式
                    // ModelProjection 会恢复对应投影。旧投影不可跨 revision 复用。
                    model_projection = None;
                }
                SessionEvent::ModelProjection {
                    messages: projection,
                } => model_projection = projection,
                // 1.3（docs/archive/implementation/harness-migration.md §1.3）：请求信封快照只用于审计与
                // 派发前重建自检，不参与 messages / usage / tool_calls 投影
                // （no-op）。若不显式列出，编译器会因新增变体而拒绝穷举 match，
                // 这正是把「不得进入投影」固化为类型义务的方式。
                SessionEvent::RequestHeader { .. } => {}
                SessionEvent::Usage(u) => usage += u,
                SessionEvent::ToolCall { .. } | SessionEvent::ToolResult { .. } => tool_calls += 1,
                SessionEvent::System { event, data } => {
                    if is_durable_user_message_event(&event) {
                        let operation_id =
                            data.get("operation_id").and_then(serde_json::Value::as_str);
                        if operation_id.is_some_and(|operation_id| {
                            !cancelled_operations.contains(operation_id)
                                && materialized_operations.insert(operation_id.to_string())
                        }) {
                            if let Some(message) = data.get("message").and_then(|value| {
                                serde_json::from_value::<Message>(value.clone()).ok()
                            }) {
                                messages.push(message.clone());
                                if let Some(projection) = model_projection.as_mut() {
                                    projection.push(message);
                                }
                            }
                        }
                    }
                }
            }
        }

        let meta = meta.ok_or_else(|| Error::InvalidSession("missing meta line".into()))?;
        let mut session = Session::new(meta);
        session.messages = messages;
        session.model_projection = model_projection;
        session.total_input_tokens = usage.input_tokens;
        session.total_output_tokens = usage.output_tokens;
        session.total_tool_calls = tool_calls;
        session.status = SessionStatus::Active;
        Ok(session)
    }

    /// 列出所有会话元数据（按 created_at 倒序）。
    pub async fn list(&self) -> Result<Vec<SessionMeta>> {
        let mut sessions = Vec::new();
        let mut rd = match tokio::fs::read_dir(&self.base_dir).await {
            Ok(rd) => rd,
            Err(_) => return Ok(Vec::new()),
        };
        while let Some(entry) = rd.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                if let Ok(meta) = self.read_meta(&path).await {
                    sessions.push(meta);
                }
            }
        }
        sessions.sort_by_key(|a| std::cmp::Reverse(a.created_at));
        Ok(sessions)
    }

    async fn read_meta(&self, path: &Path) -> Result<SessionMeta> {
        let content = tokio::fs::read_to_string(path).await?;
        let first = content
            .lines()
            .next()
            .ok_or_else(|| Error::InvalidSession("empty session file".into()))?;
        let event: SessionEvent = serde_json::from_str(first)?;
        match event {
            SessionEvent::Meta(m) => Ok(m),
            _ => Err(Error::InvalidSession("first line is not meta".into())),
        }
    }

    /// 删除会话文件。
    pub async fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        Ok(())
    }

    /// 恢复中断的会话：截断不完整的最后一行（V-STORE-01）。
    pub async fn recover(&self, session_id: &str) -> Result<Session> {
        let path = self.session_path(session_id);
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| Error::SessionNotFound(session_id.to_string()))?;

        if content.ends_with('\n') {
            // 完整，直接返回
            return self.load(session_id).await;
        }

        // 截断最后一行
        let lines: Vec<&str> = content.lines().collect();
        let recovered = if lines.is_empty() {
            String::new()
        } else {
            lines[..lines.len() - 1].join("\n") + "\n"
        };
        tokio::fs::write(&path, recovered).await?;
        self.load(session_id).await
    }

    /// 归档会话：gzip 压缩为 `.jsonl.gz` 后删除原文件。
    pub async fn archive(&self, session_id: &str) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::fs::File;
        use std::io::{self, Read};

        let path = self.session_path(session_id);
        let archive_path = self.base_dir.join(format!("{session_id}.jsonl.gz"));

        let input = File::open(&path)?;
        let output = File::create(&archive_path)?;
        let mut encoder = GzEncoder::new(output, Compression::default());
        let mut reader = io::BufReader::new(input);
        let mut buf = [0u8; 8192];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            io::Write::write_all(&mut encoder, &buf[..n])?;
        }
        encoder.finish()?;

        tokio::fs::remove_file(&path).await?;
        Ok(())
    }

    /// 原子写入：写临时文件 + rename（V-STORE-02）。
    ///
    /// 中断后旧文件或新文件二者之一完整可读，绝不产生混合内容。
    async fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
        let tmp_path = path.with_extension("jsonl.tmp");
        tokio::fs::write(&tmp_path, data).await?;
        tokio::fs::rename(&tmp_path, path).await?;
        Ok(())
    }

    /// 原子地写入完整会话（多个事件一次性落盘）。用于 V-STORE-02 演示。
    pub async fn write_session_atomic(
        &self,
        session_id: &str,
        events: &[SessionEvent],
    ) -> Result<()> {
        let path = self.session_path(session_id);
        let mut data = String::new();
        for ev in events {
            data.push_str(&serde_json::to_string(ev)?);
            data.push('\n');
        }
        Self::atomic_write(&path, data.as_bytes()).await
    }
}

fn durable_operation_id(event: &SessionEvent) -> Option<&str> {
    let SessionEvent::System { event, data } = event else {
        return None;
    };
    if !is_durable_user_message_event(event) && !is_durable_user_message_cancel_event(event) {
        return None;
    }
    data.get("operation_id")?.as_str()
}

fn cancelled_durable_operation_id(event: &SessionEvent) -> Option<String> {
    let SessionEvent::System { event, data } = event else {
        return None;
    };
    is_durable_user_message_cancel_event(event)
        .then(|| data.get("operation_id")?.as_str().map(str::to_owned))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contract::Message;
    use serde_json::json;

    async fn setup() -> (tempfile::TempDir, SessionStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        (dir, store)
    }

    #[tokio::test]
    async fn create_append_load_roundtrip() {
        let (_d, store) = setup().await;
        let session = store.create("claude-sonnet-4", "anthropic").await.unwrap();
        let id = &session.meta.id;

        store
            .append(id, SessionEvent::Message(Message::user_text("hello")))
            .await
            .unwrap();
        store
            .append(id, SessionEvent::Usage(Usage::new(10, 5)))
            .await
            .unwrap();
        store
            .append(
                id,
                SessionEvent::ToolCall {
                    name: "read_file".into(),
                    input: json!({"path": "/a"}),
                },
            )
            .await
            .unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.total_input_tokens, 10);
        assert_eq!(loaded.total_output_tokens, 5);
        assert_eq!(loaded.total_tool_calls, 1);
    }

    #[tokio::test]
    async fn history_snapshot_restores_complete_working_set_then_appends_new_messages() {
        let (_d, store) = setup().await;
        let session = store.create("model", "provider").await.unwrap();
        let id = &session.meta.id;

        store
            .append(
                id,
                SessionEvent::Message(Message::user_text("obsolete prefix")),
            )
            .await
            .unwrap();
        store
            .append(
                id,
                SessionEvent::HistorySnapshot {
                    messages: vec![Message::assistant_text("persisted answer")],
                },
            )
            .await
            .unwrap();
        store
            .append(
                id,
                SessionEvent::Message(Message::user_text("next question")),
            )
            .await
            .unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].text_content(), "persisted answer");
        assert_eq!(loaded.messages[1].text_content(), "next question");
    }

    #[tokio::test]
    async fn model_projection_does_not_replace_canonical_history() {
        let (_d, store) = setup().await;
        let session = store.create("model", "provider").await.unwrap();
        let id = &session.meta.id;

        store
            .append(id, SessionEvent::Message(Message::user_text("evidence")))
            .await
            .unwrap();
        store
            .append(
                id,
                SessionEvent::ModelProjection {
                    messages: Some(vec![Message::user_text("summary")]),
                },
            )
            .await
            .unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages[0].text_content(), "evidence");
        assert_eq!(
            loaded.model_projection.unwrap()[0].text_content(),
            "summary"
        );
    }

    #[tokio::test]
    async fn later_none_projection_event_clears_an_old_projection() {
        let (_d, store) = setup().await;
        let session = store.create("model", "provider").await.unwrap();
        let id = &session.meta.id;
        store
            .append(
                id,
                SessionEvent::ModelProjection {
                    messages: Some(vec![Message::user_text("old summary")]),
                },
            )
            .await
            .unwrap();
        store
            .append(id, SessionEvent::ModelProjection { messages: None })
            .await
            .unwrap();

        assert!(store.load(id).await.unwrap().model_projection.is_none());
    }

    #[tokio::test]
    async fn messages_appended_after_projection_are_visible_in_both_views() {
        let (_d, store) = setup().await;
        let session = store.create("model", "provider").await.unwrap();
        let id = &session.meta.id;
        store
            .append(
                id,
                SessionEvent::ModelProjection {
                    messages: Some(vec![Message::user_text("summary")]),
                },
            )
            .await
            .unwrap();
        store
            .append(id, SessionEvent::Message(Message::user_text("new turn")))
            .await
            .unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages[0].text_content(), "new turn");
        let projection = loaded.model_projection.unwrap();
        assert_eq!(projection[0].text_content(), "summary");
        assert_eq!(projection[1].text_content(), "new turn");
    }

    #[tokio::test]
    async fn v_store_01_recover_truncates_incomplete_line() {
        // V-STORE-01：最后一行只写一半时，恢复保留完整历史并截断
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        store
            .append(id, SessionEvent::Message(Message::user_text("good")))
            .await
            .unwrap();

        // 人为追加一个不完整的半行（无换行结尾）
        let path = store.session_path(id);
        let incomplete = r#"{"message":{"role":"user","content":[{"type":"text","text":"bad"}]}"#;
        // 注意：半行不以 \n 结尾
        use tokio::io::AsyncWriteExt;
        let mut f = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .unwrap();
        f.write_all(incomplete.as_bytes()).await.unwrap();
        f.flush().await.unwrap();

        // 此时直接 load 应失败（半行无法反序列化）
        assert!(store.load(id).await.is_err());

        // recover 截断后可正常 load
        let recovered = store.recover(id).await.unwrap();
        assert_eq!(recovered.messages.len(), 1);
        assert_eq!(recovered.messages[0].text_content(), "good");
    }

    #[tokio::test]
    async fn v_store_02_atomic_write_keeps_old_or_new() {
        // V-STORE-02：原子写入中断后旧/新文件二选一完整可读
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        let path = store.session_path(id);

        // 原子写入完整内容（含 Meta 行）
        let meta = session.meta.clone();
        store
            .write_session_atomic(
                id,
                &[
                    SessionEvent::Meta(meta),
                    SessionEvent::Message(Message::user_text("a")),
                    SessionEvent::Message(Message::user_text("b")),
                ],
            )
            .await
            .unwrap();

        // 写入后文件存在且可读
        assert!(path.exists());
        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 2);

        // 临时文件已被清理（rename 后不存在）
        let tmp = path.with_extension("jsonl.tmp");
        assert!(!tmp.exists());
    }

    #[tokio::test]
    async fn list_sorted_desc() {
        let (_d, store) = setup().await;
        let s1 = store.create("m", "p").await.unwrap();
        // 确保时间戳不同
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let s2 = store.create("m", "p").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let s3 = store.create("m", "p").await.unwrap();

        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 3);
        // 倒序：最新在前
        assert_eq!(list[0].id, s3.meta.id);
        assert_eq!(list[1].id, s2.meta.id);
        assert_eq!(list[2].id, s1.meta.id);
    }

    #[tokio::test]
    async fn delete_removes_file() {
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        assert!(store.session_path(id).exists());
        store.delete(id).await.unwrap();
        assert!(!store.session_path(id).exists());
        assert!(store.load(id).await.is_err());
    }

    #[tokio::test]
    async fn archive_produces_gz() {
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        store
            .append(id, SessionEvent::Message(Message::user_text("data")))
            .await
            .unwrap();

        let jsonl_path = store.session_path(id);
        let gz_path = store.base_dir.join(format!("{id}.jsonl.gz"));
        assert!(jsonl_path.exists());
        assert!(!gz_path.exists());

        store.archive(id).await.unwrap();
        assert!(!jsonl_path.exists(), "original .jsonl should be removed");
        assert!(gz_path.exists(), "archive .jsonl.gz should exist");
    }

    #[tokio::test]
    async fn load_missing_session_errors() {
        let (_d, store) = setup().await;
        let r = store.load("nonexistent").await;
        assert!(matches!(r, Err(Error::SessionNotFound(_))));
    }

    #[tokio::test]
    async fn accept_drill_restart_recovers_after_crash() {
        // accept-drill-restart：进程重启演练——崩溃后 JSONL 可恢复，状态可投影
        let (_d, store) = setup().await;
        let session = store.create("claude", "anthropic").await.unwrap();
        let id = &session.meta.id;
        store
            .append(id, SessionEvent::Message(Message::user_text("msg1")))
            .await
            .unwrap();
        store
            .append(id, SessionEvent::Message(Message::user_text("msg2")))
            .await
            .unwrap();
        store
            .append(id, SessionEvent::Usage(Usage::new(5, 3)))
            .await
            .unwrap();

        // 模拟崩溃：追加一个不完整的半行（无换行）
        let path = store.session_path(id);
        use tokio::io::AsyncWriteExt;
        let mut f = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .unwrap();
        f.write_all(b"{\"message\":").await.unwrap(); // 半行
        f.flush().await.unwrap();
        drop(f);

        // 「重启」后用 recover 恢复
        let recovered = store.recover(id).await.unwrap();
        assert_eq!(recovered.messages.len(), 2, "完整历史保留");
        assert_eq!(recovered.total_input_tokens, 5);
        assert_eq!(recovered.total_output_tokens, 3);
        assert_eq!(recovered.messages[1].text_content(), "msg2");
    }

    #[tokio::test]
    async fn accept_perf_stream_linear_append() {
        // accept-perf-stream：会话追加为线性写入（每事件一行），不全量复制历史
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        for i in 0..5 {
            store
                .append(
                    id,
                    SessionEvent::Message(Message::user_text(format!("m{i}"))),
                )
                .await
                .unwrap();
        }
        // 验证：文件行数 = meta(1) + 5 条消息 = 6 行（线性，每事件独立一行）
        let content = tokio::fs::read_to_string(store.session_path(id))
            .await
            .unwrap();
        let non_empty = content.lines().filter(|l| !l.trim().is_empty()).count();
        assert_eq!(non_empty, 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_store_handles_append_complete_jsonl_records() {
        let (dir, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let session_id = session.meta.id;
        let base_dir = dir.path().to_path_buf();
        let mut writers = tokio::task::JoinSet::new();

        for writer in 0..8 {
            let base_dir = base_dir.clone();
            let session_id = session_id.clone();
            writers.spawn(async move {
                let store = SessionStore::new(base_dir);
                for item in 0..50 {
                    store
                        .append(
                            &session_id,
                            SessionEvent::Message(Message::user_text(format!(
                                "writer-{writer}-item-{item}"
                            ))),
                        )
                        .await
                        .unwrap();
                }
            });
        }
        while let Some(result) = writers.join_next().await {
            result.unwrap();
        }

        let content = tokio::fs::read_to_string(store.session_path(&session_id))
            .await
            .unwrap();
        for line in content.lines() {
            serde_json::from_str::<SessionEvent>(line).unwrap();
        }
        assert_eq!(store.load(&session_id).await.unwrap().messages.len(), 400);
    }

    #[tokio::test]
    async fn durable_user_message_is_idempotent_and_cancellable() {
        let (_dir, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        let message = Message::user_text("guide the active run");

        store
            .append_durable_user_message(id, "operation-1", &message, "steer", None)
            .await
            .unwrap();
        store
            .append_durable_user_message(id, "operation-1", &message, "steer", None)
            .await
            .unwrap();
        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text_content(), "guide the active run");

        store
            .cancel_durable_user_message(id, "operation-1")
            .await
            .unwrap();
        store
            .cancel_durable_user_message(id, "operation-1")
            .await
            .unwrap();
        assert!(store.load(id).await.unwrap().messages.is_empty());
    }

    #[tokio::test]
    async fn legacy_prefixed_durable_events_are_read_but_never_written() {
        // legacy name, read-only：旧 JSONL 的 r_code_ 前缀事件必须继续被物化/取消，
        // 新写入只用无前缀事件名。
        let (_dir, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        let legacy_staged = r#"{"system":{"event":"r_code_durable_user_message","data":{"operation_id":"legacy-1","message":{"role":"user","content":[{"type":"text","text":"legacy staged"}]},"mode":"steer"}}}"#;
        let legacy_cancelled = r#"{"system":{"event":"r_code_durable_user_message","data":{"operation_id":"legacy-2","message":{"role":"user","content":[{"type":"text","text":"legacy cancelled"}]},"mode":"steer"}}}
{"system":{"event":"r_code_durable_user_message_cancelled","data":{"operation_id":"legacy-2"}}}"#;
        let path = store.session_path(id);
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .unwrap();
        use tokio::io::AsyncWriteExt;
        file.write_all(format!("{legacy_staged}\n{legacy_cancelled}\n").as_bytes())
            .await
            .unwrap();
        file.flush().await.unwrap();
        drop(file);

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 1, "旧 staged 物化、被取消的不物化");
        assert_eq!(loaded.messages[0].text_content(), "legacy staged");

        store
            .append_durable_user_message(
                id,
                "fresh-operation",
                &Message::user_text("fresh write"),
                "steer",
                None,
            )
            .await
            .unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        let last_line = content
            .lines()
            .rfind(|line| !line.trim().is_empty())
            .unwrap();
        assert!(
            last_line.contains(r#""event":"durable_user_message""#),
            "新写入必须使用无前缀事件名：{last_line}"
        );
        assert!(!last_line.contains("r_code_"));
    }

    #[tokio::test]
    async fn durable_retry_repairs_an_incomplete_tail_before_appending() {
        let (_dir, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        let path = store.session_path(id);
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .unwrap();
        file.write_all(b"{\"system\":").await.unwrap();
        file.flush().await.unwrap();
        drop(file);

        store
            .append_durable_user_message(
                id,
                "operation-after-crash",
                &Message::user_text("recovered"),
                "steer",
                None,
            )
            .await
            .unwrap();

        let content = tokio::fs::read_to_string(path).await.unwrap();
        assert!(content
            .lines()
            .all(|line| serde_json::from_str::<SessionEvent>(line).is_ok()));
        assert_eq!(store.load(id).await.unwrap().messages.len(), 1);
    }

    #[tokio::test]
    async fn request_header_events_are_no_op_in_load_projection() {
        // 1.3：RequestHeader 行夹在正常事件之间时，load 投影必须与没有它时完全
        // 一致（不进 messages / usage / tool_calls），且原始行使用 snake_case tag。
        let (_d, store) = setup().await;
        let session = store.create("m", "p").await.unwrap();
        let id = &session.meta.id;
        store
            .append(id, SessionEvent::Message(Message::user_text("question")))
            .await
            .unwrap();
        for reason in ["initial", "change"] {
            store
                .append(
                    id,
                    SessionEvent::RequestHeader {
                        system_sha256: format!("sys-{reason}"),
                        tools_sha256: "tools".into(),
                        messages_sha256: "messages".into(),
                        reason: reason.into(),
                        excluded_tails: vec!["local_clock".into()],
                        tool_names: vec!["read_file".into()],
                        hosted_tool_names: Vec::new(),
                        max_tokens: 4_096,
                        // 预算审计组（阶段 A）：此处测试缺省组即可（serde 默认）。
                        provider_name: None,
                        provider_kind: None,
                        model: None,
                        protocol: None,
                        context_window_tokens: 0,
                        text_tokens: 0,
                        image_tokens: 0,
                        document_tokens: 0,
                        tool_schema_tokens: 0,
                        estimated_input_tokens: 0,
                        requested_output_tokens: 0,
                        reserve_tokens: 0,
                        materialized_wire_bytes: 0,
                        attachment_count: 0,
                        anchoring_phase: None,
                        context_profile: None,
                        attachment_ids: Vec::new(),
                    },
                )
                .await
                .unwrap();
        }
        store
            .append(id, SessionEvent::Message(Message::assistant_text("answer")))
            .await
            .unwrap();
        store
            .append(id, SessionEvent::Usage(Usage::new(11, 7)))
            .await
            .unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 2, "RequestHeader 不得物化为消息");
        assert_eq!(loaded.messages[0].text_content(), "question");
        assert_eq!(loaded.messages[1].text_content(), "answer");
        assert_eq!(loaded.total_input_tokens, 11);
        assert_eq!(loaded.total_output_tokens, 7);
        assert_eq!(loaded.total_tool_calls, 0);

        // 原始 JSONL 使用与其他变体一致的外部 tagged + snake_case 表示，
        // 供 jq 抽取（验收：手工跑长会话后 jq '.request_header.reason'）。
        let content = tokio::fs::read_to_string(store.session_path(id))
            .await
            .unwrap();
        let header_lines: Vec<&str> = content
            .lines()
            .filter(|line| line.contains(r#""request_header""#))
            .collect();
        assert_eq!(header_lines.len(), 2);
        assert!(header_lines[0].contains(r#""reason":"initial""#));
        assert!(header_lines[1].contains(r#""reason":"change""#));
    }

    #[tokio::test]
    async fn test_mig_three_version_fixtures() {
        // test-mig-three：三版本迁移 fixture 都可被当前实现读取
        let (_d, store) = setup().await;
        let base = store.base_dir.clone();

        // v0 最早版本：仅 Meta + Message，无额外字段
        let v0 = r#"{"meta":{"id":"v0","created_at":"2024-01-01T00:00:00Z","model":"m","provider":"p"}}
{"message":{"role":"user","content":[{"type":"text","text":"old"}]}}
"#;
        // v1 上一版本：含 Usage
        let v1 = r#"{"meta":{"id":"v1","created_at":"2024-06-01T00:00:00Z","model":"m","provider":"p"}}
{"message":{"role":"assistant","content":[{"type":"text","text":"prev"}]}}
{"usage":{"input_tokens":7,"output_tokens":2}}
"#;
        // v2 未来版本：含未知字段（future_field）与未知事件类型（应被跳过/忽略）
        let v2 = r#"{"meta":{"id":"v2","created_at":"2025-01-01T00:00:00Z","model":"m","provider":"p","future_field":42}}
{"message":{"role":"user","content":[{"type":"text","text":"future"}]}}
"#;

        for (name, content) in [("v0", v0), ("v1", v1), ("v2", v2)] {
            tokio::fs::write(base.join(format!("{name}.jsonl")), content)
                .await
                .unwrap();
            let loaded = store
                .load(name)
                .await
                .unwrap_or_else(|e| panic!("load {name} failed: {e}"));
            assert!(!loaded.messages.is_empty(), "{name} should have messages");
            assert_eq!(loaded.meta.id, name);
        }
        // v1 的 usage 累计应正确
        let v1_loaded = store.load("v1").await.unwrap();
        assert_eq!(v1_loaded.total_input_tokens, 7);
    }
}
