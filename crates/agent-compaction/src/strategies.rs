//! 压缩策略实现。
//!
//! 参见 `06-compaction.html §3`。三种策略：滑动窗口、LLM 摘要、智能选择。

use agent_contract::{
    Capabilities, CompactionStrategy, CompletionRequest, CompletionResponse, LlmProvider, Message,
    Role, Session, StopReason, StreamEvent,
};
use agent_error::{Error, Result};
use std::sync::Arc;

/// 默认管理器使用的空 Provider（SmartCompaction 仅在 should_compact 时才用，
/// 默认实例化不触发真实调用）。
pub struct NoopProvider;

#[async_trait::async_trait]
impl LlmProvider for NoopProvider {
    async fn complete(&self, _: Arc<CompletionRequest>) -> Result<CompletionResponse> {
        Err(Error::NotImplemented("noop provider".into()))
    }
    async fn stream(
        &self,
        _: Arc<CompletionRequest>,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        Err(Error::NotImplemented("noop provider".into()))
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: false,
            supports_tool_use: false,
            supports_vision: false,
            supports_prompt_caching: false,
            max_context_tokens: 0,
            max_output_tokens: 0,
        }
    }
    fn name(&self) -> &str {
        "noop"
    }
}

/// 滑动窗口压缩：保留前 N + 后 N，中间插占位符。
pub struct SlidingWindowCompaction {
    pub keep_recent: usize,
    pub keep_first: usize,
}

impl SlidingWindowCompaction {
    pub fn new(keep_first: usize, keep_recent: usize) -> Self {
        Self {
            keep_recent,
            keep_first,
        }
    }
}

#[async_trait::async_trait]
impl CompactionStrategy for SlidingWindowCompaction {
    fn should_compact(&self, session: &Session, max_tokens: u32) -> bool {
        let total: u32 = session.estimate_tokens();
        total > max_tokens * 80 / 100
    }

    async fn compact(&self, session: &Session) -> Result<Vec<Message>> {
        let len = session.messages.len();
        if len <= self.keep_recent + self.keep_first {
            return Ok(session.messages.clone());
        }

        let mut result = Vec::new();
        result.extend(session.messages[..self.keep_first].iter().cloned());
        let compressed = len - self.keep_recent - self.keep_first;
        // 审计：记录被压缩的范围与数量（accept-perf-audit）
        result.push(Message::system_text(format!(
            "[compaction: sliding_window compressed {compressed} messages, range {keep_first}..{end}, kept first {keep_first} + last {keep_recent}]",
            keep_first = self.keep_first,
            keep_recent = self.keep_recent,
            end = len - self.keep_recent
        )));
        result.extend(session.messages[len - self.keep_recent..].iter().cloned());
        Ok(result)
    }

    fn name(&self) -> &str {
        "sliding_window"
    }
}

/// LLM 摘要压缩：前半段生成摘要，保留后半段。
pub struct LlmSummaryCompaction {
    pub provider: Arc<dyn LlmProvider>,
    pub trigger_messages: usize,
}

impl LlmSummaryCompaction {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            trigger_messages: 20,
        }
    }
}

fn summary_source(messages: &[Message]) -> Result<String> {
    let mut source = String::new();
    for message in messages {
        let role = match message.role {
            Role::User => "USER",
            Role::Assistant => "ASSISTANT",
        };
        source.push_str(role);
        source.push_str(":\n");
        for block in &message.content {
            let serialized = serde_json::to_string(block).map_err(Error::Json)?;
            source.push_str(&serialized);
            source.push('\n');
        }
        source.push('\n');
    }
    Ok(source)
}

fn atomic_summary_split(messages: &[Message]) -> usize {
    let mut split = messages.len() / 2;
    if split == 0 || split >= messages.len() {
        return split;
    }
    let left_has_tool_use = messages[split - 1]
        .content
        .iter()
        .any(|block| block.is_tool_use());
    let right_has_tool_result = messages[split]
        .content
        .iter()
        .any(|block| block.is_tool_result());
    if left_has_tool_use && right_has_tool_result {
        split += 1;
    }
    split
}

#[async_trait::async_trait]
impl CompactionStrategy for LlmSummaryCompaction {
    fn should_compact(&self, session: &Session, _max_tokens: u32) -> bool {
        session.messages.len() > self.trigger_messages
    }

    async fn compact(&self, session: &Session) -> Result<Vec<Message>> {
        let split = atomic_summary_split(&session.messages);
        if split == 0 {
            return Ok(session.messages.clone());
        }
        let to_compact = &session.messages[..split];
        let summary_prompt = format!(
            "Create a precise continuation checkpoint from the conversation below. Preserve user \
goals and constraints, decisions, tool names and important inputs, complete tool-result evidence, \
file paths and symbols, commands and exit status, edits, verification results, errors and root \
causes, and unfinished work. Do not invent facts. Return only the checkpoint.\n\n{}",
            summary_source(to_compact)?
        );

        let response = self
            .provider
            .complete(Arc::new(CompletionRequest {
                model: session.meta.model.clone(),
                system: Some("You are a conversation summarizer.".into()),
                messages: vec![Message::user_text(summary_prompt)],
                tools: vec![],
                hosted_tools: vec![],
                max_tokens: 4096,
                temperature: Some(0.3),
                enable_caching: false,
                inference: Default::default(),
            }))
            .await?;

        if response.stop_reason == StopReason::MaxTokens {
            return Err(Error::Compaction(
                "summary reached the output limit and was not installed".into(),
            ));
        }
        let summary_text = response.text();
        if summary_text.trim().is_empty() {
            return Err(Error::Compaction("summary was empty".into()));
        }
        let mut result = vec![Message::system_text(format!(
            "[compaction: llm_summary of messages 0..{split}: {summary_text}]",
        ))];
        result.extend(session.messages[split..].iter().cloned());
        Ok(result)
    }

    fn name(&self) -> &str {
        "llm_summary"
    }
}

/// 智能压缩：保留首条 + 最近 N 条 + 所有工具结果，压缩闲聊。
pub struct SmartCompaction {
    #[allow(dead_code)]
    pub provider: Arc<dyn LlmProvider>,
    pub keep_recent: usize,
}

impl SmartCompaction {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            keep_recent: 10,
        }
    }
}

#[async_trait::async_trait]
impl CompactionStrategy for SmartCompaction {
    fn should_compact(&self, session: &Session, _max_tokens: u32) -> bool {
        session.messages.len() > 30
    }

    async fn compact(&self, session: &Session) -> Result<Vec<Message>> {
        let len = session.messages.len();
        let mut preserved = std::collections::HashSet::new();
        if len > 0 {
            preserved.insert(0usize); // 第一条（目标）
        }
        for i in len.saturating_sub(self.keep_recent)..len {
            preserved.insert(i);
        }
        // 保留所有工具结果
        for (i, msg) in session.messages.iter().enumerate() {
            if msg.content.iter().any(|c| c.is_tool_result()) {
                preserved.insert(i);
            }
        }

        let mut result = Vec::new();
        let mut compressed_count = 0usize;
        for (i, msg) in session.messages.iter().enumerate() {
            if preserved.contains(&i) {
                result.push(msg.clone());
            } else {
                compressed_count += 1;
            }
        }
        if compressed_count > 0 {
            result.push(Message::system_text(format!(
                "[{compressed_count} messages compressed]"
            )));
        }
        Ok(result)
    }

    fn name(&self) -> &str {
        "smart"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::CompactionManager;
    use agent_contract::{
        Capabilities, CompletionResponse, ContentBlock, StopReason, StreamEvent, Usage,
    };
    use async_trait::async_trait;

    /// 测试用 DummyProvider，返回固定摘要文本。
    struct DummyProvider {
        text: String,
        stop_reason: StopReason,
        requests: Option<Arc<std::sync::Mutex<Vec<CompletionRequest>>>>,
    }

    #[async_trait]
    impl LlmProvider for DummyProvider {
        async fn complete(&self, request: Arc<CompletionRequest>) -> Result<CompletionResponse> {
            if let Some(requests) = &self.requests {
                requests.lock().unwrap().push(request.as_ref().clone());
            }
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.text.clone(),
                }],
                stop_reason: self.stop_reason.clone(),
                usage: Usage::default(),
            })
        }

        async fn stream(
            &self,
            _request: Arc<CompletionRequest>,
        ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
            Err(Error::NotImplemented("dummy stream".into()))
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_streaming: false,
                supports_tool_use: false,
                supports_vision: false,
                supports_prompt_caching: false,
                max_context_tokens: 100_000,
                max_output_tokens: 0,
            }
        }

        fn name(&self) -> &str {
            "dummy"
        }
    }

    fn make_session(n: usize) -> Session {
        let mut meta = agent_contract::SessionMeta::new("m", "p");
        meta.id = "test".into();
        let mut session = Session::new(meta);
        for i in 0..n {
            session.push_user(format!("msg {i}"));
        }
        session
    }

    #[tokio::test]
    async fn sliding_window_compresses_middle() {
        let session = make_session(20);
        let s = SlidingWindowCompaction::new(2, 5);
        // should_compact 需要超阈值；这里直接测 compact
        let result = s.compact(&session).await.unwrap();
        // 前 2 + 1 占位 + 后 5 = 8
        assert_eq!(result.len(), 8);
        assert!(result[2].text_content().contains("compressed 13 messages"));
    }

    #[tokio::test]
    async fn sliding_window_no_op_when_small() {
        let session = make_session(5);
        let s = SlidingWindowCompaction::new(2, 5);
        let result = s.compact(&session).await.unwrap();
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn llm_summary_uses_provider() {
        let mut session = make_session(30);
        // 让前半段含可读文本
        session.messages[0] = Message::user_text("important goal");
        let provider = Arc::new(DummyProvider {
            text: "SUMMARY".into(),
            stop_reason: StopReason::EndTurn,
            requests: None,
        });
        let s = LlmSummaryCompaction::new(provider);
        let result = s.compact(&session).await.unwrap();
        assert!(result[0].text_content().contains("llm_summary"));
        assert!(result[0].text_content().contains("SUMMARY"));
        // 后半段保留
        assert!(result.len() < session.messages.len());
    }

    #[tokio::test]
    async fn llm_summary_source_contains_tool_calls_and_results() {
        let mut session = make_session(24);
        session.messages[2] = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "src/main.rs"}),
            }],
        };
        session.messages[3] = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: "evidence-tail-保留".into(),
                is_error: false,
            }],
        };
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(DummyProvider {
            text: "SUMMARY".into(),
            stop_reason: StopReason::EndTurn,
            requests: Some(requests.clone()),
        });

        LlmSummaryCompaction::new(provider)
            .compact(&session)
            .await
            .unwrap();

        let prompt = requests.lock().unwrap()[0].messages[0].text_content();
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("src/main.rs"));
        assert!(prompt.contains("evidence-tail-保留"));
    }

    #[tokio::test]
    async fn llm_summary_rejects_max_tokens_response() {
        let session = make_session(30);
        let provider = Arc::new(DummyProvider {
            text: "TRUNCATED".into(),
            stop_reason: StopReason::MaxTokens,
            requests: None,
        });
        let error = LlmSummaryCompaction::new(provider)
            .compact(&session)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("output limit"));
    }

    #[test]
    fn summary_split_never_separates_tool_use_from_its_result_message() {
        let messages = vec![
            Message::user_text("goal"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "src/lib.rs"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: "evidence".into(),
                    is_error: false,
                }],
            },
            Message::assistant_text("done"),
        ];
        assert_eq!(atomic_summary_split(&messages), 3);
    }

    #[tokio::test]
    async fn v_comp_01_smart_preserves_goal_and_tool_results() {
        // V-COMP-01：压缩前后 user 目标、工具结果可追溯
        let mut meta = agent_contract::SessionMeta::new("m", "p");
        meta.id = "t".into();
        let mut session = Session::new(meta);
        session.push_user("我的目标是 X"); // 第一条（目标）
        for i in 0..40 {
            session.push_assistant(format!("chat {i}")); // 闲聊
        }
        // 一条工具结果（在中间）
        session.messages.push(Message {
            role: agent_contract::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "result data".into(),
                is_error: false,
            }],
        });
        for i in 0..5 {
            session.push_user(format!("recent {i}")); // 最近消息
        }

        let provider = Arc::new(DummyProvider {
            text: "x".into(),
            stop_reason: StopReason::EndTurn,
            requests: None,
        });
        let s = SmartCompaction::new(provider);
        let result = s.compact(&session).await.unwrap();

        // 目标保留
        assert!(
            result
                .iter()
                .any(|m| m.text_content().contains("我的目标是 X")),
            "user goal must be preserved"
        );
        // 工具结果保留
        assert!(
            result
                .iter()
                .any(|m| m.content.iter().any(|c| c.is_tool_result())),
            "tool result must be preserved"
        );
        // 最近消息保留
        assert!(
            result.iter().any(|m| m.text_content().contains("recent 4")),
            "recent messages must be preserved"
        );
        // 有压缩占位符
        assert!(result
            .iter()
            .any(|m| m.text_content().contains("compressed")));
    }

    #[test]
    fn manager_default_strategies() {
        // 仅验证构造不 panic
        let _m = CompactionManager::new(180_000);
    }

    #[tokio::test]
    async fn accept_perf_audit_compaction_records_range_and_source() {
        // accept-perf-audit：压缩输出记录输入范围与摘要来源，可审计
        let session = make_session(20);
        let s = SlidingWindowCompaction::new(2, 5);
        let result = s.compact(&session).await.unwrap();
        let audit = result[2].text_content();
        assert!(
            audit.contains("compaction"),
            "缺少 compaction 标记: {audit}"
        );
        assert!(audit.contains("range"), "缺少输入范围: {audit}");
        assert!(audit.contains("compressed"), "缺少压缩计数: {audit}");

        // LLM 摘要策略记录来源
        let mut session2 = make_session(30);
        session2.messages[0] = Message::user_text("goal");
        let provider = Arc::new(DummyProvider {
            text: "S".into(),
            stop_reason: StopReason::EndTurn,
            requests: None,
        });
        let s2 = LlmSummaryCompaction::new(provider);
        let result2 = s2.compact(&session2).await.unwrap();
        let audit2 = result2[0].text_content();
        assert!(audit2.contains("llm_summary"), "缺少摘要来源标记: {audit2}");
        assert!(audit2.contains("messages 0.."), "缺少输入范围: {audit2}");
    }
}
