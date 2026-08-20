//! `agent-tauri` -- Tauri 应用壳的运行时无关核心。
//!
//! 参见 `10-tauri-shell.html`。提供 `AppState`（全局状态）与 `AppEvent`（事件推送）。
//!
//! **说明**：完整的 `#[tauri::command]` 宏、`tauri::Builder` 与前端集成需要
//! `tauri` 框架依赖与 WebView 运行时，属于产品层（R-Code/Hermes）职责。本公共
//! crate 只提供与 Tauri 无关的状态/事件类型与初始化逻辑，便于在无 GUI 环境下
//! 测试与复用。

use agent_config::Config;
use agent_contract::ToolHost;
use agent_store::SessionStore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

/// 应用事件（推送到前端）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEvent {
    TextDelta { text: String },
    ToolUse { name: String, input: Value },
    ToolResult { call_id: String, output: String },
    MessageComplete,
    Error { message: String },
    SessionChanged { session_id: String },
    ConfigUpdated,
}

/// 全局应用状态。
pub struct AppState {
    pub config: Config,
    pub session_store: SessionStore,
    pub tool_host: Arc<RwLock<Box<dyn ToolHost>>>,
    pub current_session: RwLock<Option<String>>,
    pub event_sender: broadcast::Sender<AppEvent>,
}

impl AppState {
    /// 从配置初始化：加载 config、创建 session_store、组装 tool_host、建立事件通道。
    pub async fn init(config: Config) -> agent_error::Result<Self> {
        // 初始化日志
        tracing_subscriber_try_init(&config.log_level);

        // 创建存储目录
        tokio::fs::create_dir_all(&config.storage.base_dir).await?;
        tokio::fs::create_dir_all(&config.storage.sessions_dir).await?;

        let session_store = SessionStore::new(config.storage.sessions_dir.clone());

        // 组装 tool_host：从 MCP 配置构造（失败则用 NullToolHost）
        let tool_host: Box<dyn ToolHost> = if config.mcp_servers.is_empty() {
            Box::new(agent_contract::NullToolHost)
        } else {
            // agent-config 与 agent-mcp 各自定义 ServerSpec（结构相同），
            // 通过 JSON 往返转换，避免让 config 反向依赖 mcp。
            let mcp_cfg = convert_mcp_config(&config);
            match agent_mcp::McpToolHost::from_config(mcp_cfg).await {
                Ok(host) => Box::new(host),
                Err(e) => {
                    tracing::warn!(
                        "failed to build MCP tool host: {e}; falling back to NullToolHost"
                    );
                    Box::new(agent_contract::NullToolHost)
                }
            }
        };

        let (event_sender, _) = broadcast::channel(256);

        Ok(Self {
            config,
            session_store,
            tool_host: Arc::new(RwLock::new(tool_host)),
            current_session: RwLock::new(None),
            event_sender,
        })
    }

    /// 订阅事件流。
    pub fn subscribe(&self) -> broadcast::Receiver<AppEvent> {
        self.event_sender.subscribe()
    }

    /// 广播事件（忽略无订阅者错误）。
    pub fn emit(&self, event: AppEvent) {
        let _ = self.event_sender.send(event);
    }
}

/// 尽力初始化 tracing（重复调用安全）。
fn tracing_subscriber_try_init(_level: &str) {
    // 实际产品应使用 tracing_subscriber::fmt().with_env_filter(level).try_init()
    // 此处保持无依赖；产品层负责日志订阅。
}

/// 将 agent-config 的 ServerSpec 转为 agent-mcp 的 ServerSpec（JSON 往返）。
fn convert_mcp_config(config: &Config) -> agent_mcp::McpConfig {
    let mut servers = std::collections::HashMap::new();
    for (k, v) in &config.mcp_servers {
        let json = serde_json::to_value(v).unwrap_or(Value::Null);
        if let Ok(spec) = serde_json::from_value::<agent_mcp::ServerSpec>(json) {
            servers.insert(k.clone(), spec);
        }
    }
    agent_mcp::McpConfig { servers }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_config::{Config, ProviderConfig, StorageConfig};
    use std::collections::HashMap;

    fn test_config(dir: &std::path::Path) -> Config {
        Config {
            default_provider: "anthropic".into(),
            log_level: "info".into(),
            providers: {
                let mut m = HashMap::new();
                m.insert(
                    "anthropic".into(),
                    ProviderConfig {
                        base_url: "x".into(),
                        api_key: "k".into(),
                        model: "m".into(),
                        max_tokens: None,
                        temperature: None,
                        protocol: None,
                        provider_kind: None,
                        show_reasoning: true,
                    },
                );
                m
            },
            mcp_servers: HashMap::new(),
            storage: StorageConfig {
                base_dir: dir.to_path_buf(),
                sessions_dir: dir.join("s"),
                skills_dir: dir.join("k"),
                memories_dir: dir.join("m"),
            },
            compaction: Default::default(),
            orchestration: Default::default(),
            planning: Default::default(),
            diagnostics: Default::default(),
            tauri: None,
        }
    }

    #[tokio::test]
    async fn init_creates_state_with_null_host() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let state = AppState::init(config).await.unwrap();
        // 无 MCP server -> NullToolHost，list_tools 返回空
        let host = state.tool_host.read().await;
        let tools = host.list_tools().await.unwrap();
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn event_broadcast() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let state = AppState::init(config).await.unwrap();
        let mut rx = state.subscribe();
        state.emit(AppEvent::MessageComplete);
        let ev = rx.recv().await.unwrap();
        assert!(matches!(ev, AppEvent::MessageComplete));
    }

    #[test]
    fn app_event_serde() {
        let ev = AppEvent::TextDelta { text: "hi".into() };
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains(r#""type":"text_delta""#));
    }
}
