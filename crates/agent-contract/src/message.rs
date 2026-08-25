//! 消息与内容块类型。
//!
//! 参见 `04-message-types.html`。设计参考 Anthropic Messages API（最表达力），
//! 其他 Provider 向下兼容。`Role` 只有 `User` / `Assistant`；系统提示不是消息
//! 角色，而是 `CompletionRequest.system` 的独立顶层字段（与 Anthropic API 一致）。
//! Provider 适配层负责转 OpenAI 的 `system` role。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 消息角色。只有 User / Assistant 两种。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// 一条消息：角色 + 内容块列表。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// 构造仅含文本的 User 消息。
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// 构造仅含文本的 Assistant 消息。
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// 注入的「系统级」上下文备注（如压缩摘要 / 占位符）。
    ///
    /// **设计说明**：`Role` 契约只有 User / Assistant（见 04 篇）。压缩等模块
    /// 需要向消息流注入上下文备注，这里以 Assistant 角色承载（视作助手提供的
    /// 背景信息），避免破坏 Role 契约。Provider 适配层可将其折叠进 system
    /// 字段或保留为 assistant 消息。
    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// 拼接所有文本块。
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// 返回所有 `ToolUse` 块的引用。
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content.iter().filter(|b| b.is_tool_use()).collect()
    }

    /// 粗略 token 估算（len / 4），用于压缩触发判定，非精确计数。
    pub fn estimate_tokens(&self) -> u32 {
        let mut tokens = 0u32;
        for block in &self.content {
            tokens += block.estimate_tokens();
        }
        tokens
    }

    /// 推入一个内容块。
    pub fn push_block(&mut self, block: ContentBlock) {
        self.content.push(block);
    }
}

/// 内容块。参考 Anthropic 内容块结构，仅保留 LLM 协议级别的通用类型；
/// 产品专属扩展通过 `Custom` 钩子携带，公共层不解释其语义。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// 纯文本。
    Text { text: String },

    /// 思考过程（Claude extended thinking）。
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    /// 工具调用请求。
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    /// 工具调用结果。
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },

    /// 图像。
    Image { source: ImageSource },

    /// 用户附加的可读文件。
    ///
    /// 文本类文件直接保留 UTF-8 正文；二进制文档（目前为 PDF）使用 Base64。
    /// Provider 适配层负责映射为原生 document/file 输入，或将文本文件展开为
    /// 普通文本块。文件名仅用于向模型说明来源，不参与本地路径解析。
    File { source: FileSource },

    /// 附件引用块（R-Code 持久附件，docs/archive/implementation/multimodal-attachments-and-deepseek-plan-anchoring-implementation.md §4.1）。
    ///
    /// 二进制正文只存一份于宿主 BlobStore；会话/队列/投影只携带本引用。
    /// Base64 仅允许出现在两个临时边界：WebView staging IPC 与 Provider 请求
    /// 构造期的物化副本。Provider 适配层在最终序列化前必须断言不存在未物化
    /// 的 Attachment（fail closed），不得降级为占位文本后继续请求。
    Attachment { source: AttachmentRefV1 },

    /// 产品层扩展块（如文件引用、选区引用等）。公共层不解释其语义，
    /// 仅保证可序列化与透传；Provider 适配层将其降级为占位文本。
    Custom {
        /// 扩展类型名，由产品层定义（如 "file_ref"、"selection_ref"）。
        type_name: String,
        /// 不透明载荷。
        data: Value,
    },
}

impl ContentBlock {
    /// 若为文本块，返回文本引用。
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::File { source } => source.text.as_deref(),
            _ => None,
        }
    }

    /// 是否为工具调用块。
    pub fn is_tool_use(&self) -> bool {
        matches!(self, ContentBlock::ToolUse { .. })
    }

    /// 是否为工具结果块。
    pub fn is_tool_result(&self) -> bool {
        matches!(self, ContentBlock::ToolResult { .. })
    }

    /// 工具调用的名称。
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            ContentBlock::ToolUse { name, .. } => Some(name),
            _ => None,
        }
    }

    /// 工具调用的 id。
    pub fn tool_id(&self) -> Option<&str> {
        match self {
            ContentBlock::ToolUse { id, .. } => Some(id),
            _ => None,
        }
    }

    /// 工具结果对应的 tool_use_id。
    pub fn tool_use_id(&self) -> Option<&str> {
        match self {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id),
            _ => None,
        }
    }

    /// 粗略 token 估算。
    pub fn estimate_tokens(&self) -> u32 {
        match self {
            ContentBlock::Text { text } => (text.len() / 4) as u32,
            ContentBlock::Thinking { thinking, .. } => (thinking.len() / 4) as u32,
            ContentBlock::ToolUse { input, .. } => (input.to_string().len() / 4) as u32,
            ContentBlock::ToolResult { content, .. } => (content.len() / 4) as u32,
            ContentBlock::File { source } => source
                .text
                .as_deref()
                .map(|text| (text.len() / 4) as u32)
                .unwrap_or(10),
            // 引用块只有少量可见元数据；图片/PDF token 由 VisionBudgetProfile
            // 等确定性 profile 单独核算，绝不由字节数估算。
            ContentBlock::Attachment { .. } => 10,
            _ => 10,
        }
    }

    /// 构造一个 `is_error=true` 的已取消工具结果（用于取消配对，见 V-MSG-03）。
    pub fn cancelled_tool_result(tool_use_id: impl Into<String>) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: "[cancelled]".to_string(),
            is_error: true,
        }
    }
}

/// 图像来源。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: String,
    pub media_type: String,
    pub data: String,
}

/// 文件来源。`kind` 为 `text` 或 `base64`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSource {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// 附件类别（AttachmentRefV1.kind）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentKind {
    Image,
    Text,
    Pdf,
}

/// 附件用途：决定引用块在 Provider 请求中的去向。
///
/// - `NativeInput`：当前主模型直接读取的原图/PDF（多模态主模型场景）。
/// - `TextInput`：普通文本附件；Provider 投影阶段读取 Blob 展开 UTF-8 文本。
/// - `DisplayOnly`：文本主模型场景下保留原图预览；不进入 Provider 请求。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentPurpose {
    NativeInput,
    TextInput,
    DisplayOnly,
}

/// 附件引用（v1）。`attachment_id` 是解析 Blob 的唯一公开键——调用方不得
/// 提交 blob hash 或文件路径来绕过所有权检查；读取时以数据库元数据为权威，
/// 消息内元数据只用于无 IO 展示与预算预检。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentRefV1 {
    /// 固定为 1。
    pub version: u8,
    /// UUID；不包含本机路径。
    pub attachment_id: String,
    pub name: String,
    pub media_type: String,
    pub kind: AttachmentKind,
    pub byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub purpose: AttachmentPurpose,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_text_constructor() {
        let m = Message::user_text("hello");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.text_content(), "hello");
    }

    #[test]
    fn content_block_text_roundtrip() {
        let block = ContentBlock::Text { text: "hi".into() };
        let json = serde_json::to_string(&block).unwrap();
        // tag = "text"
        assert!(json.contains(r#""type":"text""#));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.as_text(), Some("hi"));
    }

    #[test]
    fn file_block_roundtrip_keeps_safe_metadata_and_text() {
        let block = ContentBlock::File {
            source: FileSource {
                kind: "text".into(),
                name: "main.rs".into(),
                media_type: "text/x-rust".into(),
                text: Some("fn main() {}".into()),
                data: None,
            },
        };
        let json = serde_json::to_string(&block).unwrap();
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.as_text(), Some("fn main() {}"));
    }

    #[test]
    fn tool_use_tagged_serde() {
        let block = ContentBlock::ToolUse {
            id: "t1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/a"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"tool_use""#));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_name(), Some("read_file"));
        assert_eq!(back.tool_id(), Some("t1"));
    }

    #[test]
    fn unknown_field_ignored() {
        // 未来版本可能新增字段，旧实现读取时应忽略未知键。
        let json = r#"{"type":"text","text":"x","future_field":123}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert_eq!(block.as_text(), Some("x"));
    }

    #[test]
    fn cancelled_tool_result_is_error() {
        let r = ContentBlock::cancelled_tool_result("t1");
        assert!(matches!(r, ContentBlock::ToolResult { is_error: true, .. }));
        assert_eq!(r.tool_use_id(), Some("t1"));
    }

    #[test]
    fn estimate_tokens_nonzero() {
        let m = Message::user_text("a fairly long piece of text for estimation");
        assert!(m.estimate_tokens() > 0);
    }

    #[test]
    fn attachment_ref_roundtrip_and_no_payload_leak() {
        let block = ContentBlock::Attachment {
            source: AttachmentRefV1 {
                version: 1,
                attachment_id: "0e0f1d2a-1111-4222-8333-444455556666".into(),
                name: "screenshot.png".into(),
                media_type: "image/png".into(),
                kind: AttachmentKind::Image,
                byte_len: 3_383_259,
                width: Some(1818),
                height: Some(1026),
                purpose: AttachmentPurpose::NativeInput,
            },
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"attachment""#), "encoded: {json}");
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        match back {
            ContentBlock::Attachment { source } => {
                assert_eq!(source.version, 1);
                assert_eq!(source.kind, AttachmentKind::Image);
                assert_eq!(source.purpose, AttachmentPurpose::NativeInput);
                assert_eq!(source.width, Some(1818));
                assert_eq!(source.height, Some(1026));
            }
            other => panic!("wrong variant: {other:?}"),
        }
        // 引用块序列化只含元数据：不携带 Base64 正文或本机路径。
        assert!(!json.contains("data"));
        assert!(!json.contains("base64"));
        assert!(!json.contains('\\'));
    }

    #[test]
    fn attachment_purpose_and_kind_serde_names() {
        assert_eq!(
            serde_json::to_string(&AttachmentPurpose::DisplayOnly).unwrap(),
            r#""display_only""#
        );
        assert_eq!(
            serde_json::to_string(&AttachmentPurpose::NativeInput).unwrap(),
            r#""native_input""#
        );
        assert_eq!(
            serde_json::to_string(&AttachmentPurpose::TextInput).unwrap(),
            r#""text_input""#
        );
        assert_eq!(
            serde_json::to_string(&AttachmentKind::Image).unwrap(),
            r#""image""#
        );
        assert_eq!(
            serde_json::to_string(&AttachmentKind::Text).unwrap(),
            r#""text""#
        );
        assert_eq!(
            serde_json::to_string(&AttachmentKind::Pdf).unwrap(),
            r#""pdf""#
        );
    }

    #[test]
    fn legacy_file_block_still_roundtrips_alongside_attachment() {
        // 双读合同：旧 File 块与新 Attachment 块可在同一条消息中并存。
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::File {
                    source: FileSource {
                        kind: "base64".into(),
                        name: "legacy.png".into(),
                        media_type: "image/png".into(),
                        text: None,
                        data: Some("aGVsbG8=".into()),
                    },
                },
                ContentBlock::Attachment {
                    source: AttachmentRefV1 {
                        version: 1,
                        attachment_id: "id-1".into(),
                        name: "new.png".into(),
                        media_type: "image/png".into(),
                        kind: AttachmentKind::Image,
                        byte_len: 8,
                        width: None,
                        height: None,
                        purpose: AttachmentPurpose::TextInput,
                    },
                },
            ],
        };
        let json = serde_json::to_string(&message).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content.len(), 2);
        assert!(matches!(back.content[0], ContentBlock::File { .. }));
        assert!(matches!(back.content[1], ContentBlock::Attachment { .. }));
    }
}
