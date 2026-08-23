//! `agent-contract` -- agent-contracts 核心抽象。
//!
//! 定义 Agent 对话中的消息、内容块、角色、会话数据模型，以及 Provider /
//! ToolHost / CompactionStrategy 等 trait 契约。所有公共 crate 建立在本 crate
//! 之上。
//!
//! 参见文档 `04-message-types.html`、`02-session-management.html`、
//! `01-llm-provider.html`、`03-tool-host.html`、`06-compaction.html`。

pub mod message;
pub mod session;
pub mod usage;

// ── trait 契约（按阶段逐步冻结）──────────────────────────────
pub mod compaction;
pub mod provider;
pub mod tool_host;

// ── 错误类型统一从 agent-error 复用 ──────────────────────────
pub use agent_error::{
    is_recoverable, other, Error, ErrorContext, RecoveryStrategy, Result, ResultExt,
};

pub use compaction::CompactionStrategy;
pub use message::{
    AttachmentKind, AttachmentPurpose, AttachmentRefV1, ContentBlock, FileSource, ImageSource,
    Message, Role,
};
pub use provider::{
    Capabilities, CompletionRequest, CompletionResponse, HostedToolFormat, HostedToolSpec,
    InferenceOptions, LlmProvider, StopReason, StreamEvent, VisionBudgetProfile,
};
pub use session::{Session, SessionEvent, SessionMeta, SessionStatus};
pub use tool_host::{
    CompositeToolHost, NullToolHost, ToolCallOutcome, ToolHost, ToolSource, ToolSpec,
};
pub use usage::Usage;
