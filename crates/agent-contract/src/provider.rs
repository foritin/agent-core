//! LLM Provider 抽象 trait 与请求 / 响应类型。
//!
//! 参见 `01-llm-provider.html`、`12-api-contracts.html §1`。
//! trait 定义在 `agent-contract`，具体实现（Anthropic / OpenAI / DeepSeek / Mock）
//! 在 `agent-llm` crate 中。

use crate::message::{ContentBlock, Message};
use crate::tool_host::ToolSpec;
use crate::usage::Usage;
use crate::Result;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 统一 LLM Provider 抽象。
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// 非流式完成。
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// 流式完成，返回事件流。
    async fn stream(&self, request: CompletionRequest) -> Result<BoxStream<'static, StreamEvent>>;

    /// 声明能力。
    fn capabilities(&self) -> Capabilities;

    /// 明文思考内容是否需要在下一轮逐字回传（Kimi k2/k3、DeepSeek Responses）。
    /// 默认 false，保持既有 provider 行为。
    fn echoes_reasoning(&self) -> bool {
        false
    }

    /// Provider 名称。
    fn name(&self) -> &str;
}

/// 完成请求。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceOptions {
    /// Provider 原生思考模式：enabled / disabled / adaptive。None 表示服务默认。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// 推理强度：none / minimal / low / medium / high / xhigh / max。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// 输出详略：low / medium / high。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// 完成请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// 系统提示（独立顶层字段，与 Anthropic API 一致；非消息角色）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// 由模型服务执行的托管工具。它们与客户端 `ToolSpec` 不同：Provider 会在
    /// 服务端完成调用，Agent 不能再次通过本地 ToolHost 执行。
    #[serde(default)]
    pub hosted_tools: Vec<HostedToolSpec>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub enable_caching: bool,
    /// 会话级模型推理参数；全部为空时完全沿用 Provider 默认行为。
    #[serde(default, skip_serializing_if = "InferenceOptions::is_default")]
    pub inference: InferenceOptions,
}

/// 跨 Provider 的服务端托管工具声明。
///
/// 每个协议适配器负责把它映射为厂商自己的请求结构；不支持的适配器应忽略或
/// 显式拒绝，而不能把它伪装成需要客户端执行的函数调用。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostedToolFormat {
    /// 协议原生格式：Anthropic 的版本化工具，或 Responses 的 `web_search`。
    #[default]
    Standard,
    /// 阿里百炼 Responses：网页读取使用 `web_extractor`。
    DashScope,
    /// OpenRouter Server Tools：使用 `openrouter:*` 类型。
    OpenRouter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostedToolSpec {
    WebSearch {
        #[serde(default, skip_serializing_if = "is_standard_hosted_tool_format")]
        format: HostedToolFormat,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        blocked_domains: Vec<String>,
    },
    WebFetch {
        #[serde(default, skip_serializing_if = "is_standard_hosted_tool_format")]
        format: HostedToolFormat,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        blocked_domains: Vec<String>,
    },
}

fn is_standard_hosted_tool_format(format: &HostedToolFormat) -> bool {
    *format == HostedToolFormat::Standard
}

impl HostedToolSpec {
    /// 默认的按需联网搜索：模型自行决定是否调用，每个请求最多搜索五次。
    pub fn web_search() -> Self {
        Self::web_search_with_format(HostedToolFormat::Standard)
    }

    pub fn web_search_with_format(format: HostedToolFormat) -> Self {
        Self::WebSearch {
            format,
            max_uses: Some(5),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
        }
    }

    /// 默认的按需网页读取：模型自行决定是否访问 URL，每个请求最多读取五次。
    pub fn web_fetch() -> Self {
        Self::web_fetch_with_format(HostedToolFormat::Standard)
    }

    pub fn web_fetch_with_format(format: HostedToolFormat) -> Self {
        Self::WebFetch {
            format,
            max_uses: Some(5),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
        }
    }

    pub fn is_web_search(&self) -> bool {
        matches!(self, Self::WebSearch { .. })
    }

    pub fn is_web_fetch(&self) -> bool {
        matches!(self, Self::WebFetch { .. })
    }
}

impl InferenceOptions {
    pub fn is_default(&self) -> bool {
        self.thinking.is_none() && self.reasoning_effort.is_none() && self.verbosity.is_none()
    }
}

/// 完成响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl CompletionResponse {
    /// 取首个文本块的文本（便于简单用例）。
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// 流式事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    /// Provider 明确返回、允许客户端展示的思考内容或推理摘要增量。
    ///
    /// 这里只承载可见文本；加密推理块、签名和其它 provider-private 载荷不得进入。
    ReasoningDelta {
        text: String,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolUseDelta {
        id: String,
        input_json: String,
    },
    ToolUseComplete {
        id: String,
        input: Value,
    },
    /// A tool executed by the model provider itself. Consumers may display it, but must not
    /// dispatch it to the local ToolHost.
    HostedToolUse {
        id: String,
        name: String,
        input: Value,
        /// Provider-native block retained only for protocol continuation. Product event logs
        /// should use the public fields above and must not expose this payload directly.
        #[serde(skip)]
        provider_content: Option<Value>,
    },
    /// Public, sanitized result metadata for a provider-hosted tool.
    HostedToolResult {
        id: String,
        name: String,
        output: Value,
        is_error: bool,
        /// Provider-native result retained for mixed client/server tool turns and pause/resume.
        /// It may contain encrypted search content, so UI consumers must use `output` instead.
        #[serde(skip)]
        provider_content: Option<Value>,
    },
    Stop {
        reason: StopReason,
    },
    Usage(Usage),
}

/// 停止原因。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

/// Provider 能力声明。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub supports_streaming: bool,
    pub supports_tool_use: bool,
    pub supports_vision: bool,
    pub supports_prompt_caching: bool,
    pub max_context_tokens: u32,
    /// 单次输出（max output tokens）能力；`0 = 未声明`。
    ///
    /// 运行时应回退到旧的 `max_tokens` 启发（用户设置或默认值），
    /// 不要把它当作“输出上限为 0”。
    #[serde(default)]
    pub max_output_tokens: u32,
}

impl Capabilities {
    pub fn can_use_tools(&self) -> bool {
        self.supports_tool_use
    }

    pub fn cache_key(&self) -> &'static str {
        if self.supports_prompt_caching {
            "cached"
        } else {
            "uncached"
        }
    }
}

/// 多模态模型的确定性视觉 token 预算 profile（docs/multimodal-attachments §6.2）。
///
/// 目录中 `vision=true` 的模型必须携带一个 profile；缺失时宿主返回
/// `VISION_BUDGET_PROFILE_MISSING`，不得回退到 Base64 字符估算或 OCR。
/// 首版 `deepseek_vision_exp_v1` 使用保守、确定性的 tile 上界。
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VisionBudgetProfile {
    pub profile_id: &'static str,
    pub tile_width: u32,
    pub tile_height: u32,
    pub base_tokens: u32,
    pub tokens_per_tile: u32,
    /// 保守安全乘数（tile 求和后乘它并向上取整）。
    pub safety_multiplier: f32,
    /// 请求侧单边像素上限；超过时生成请求专用派生图。
    pub max_request_edge: u32,
    /// 请求侧总像素上限。
    pub max_request_pixels: u64,
}

impl VisionBudgetProfile {
    /// 首版 DeepSeek vision 实验模型 profile。
    pub const DEEPSEEK_VISION_EXP_V1: Self = Self {
        profile_id: "deepseek_vision_exp_v1",
        tile_width: 512,
        tile_height: 512,
        base_tokens: 1_024,
        tokens_per_tile: 2_048,
        safety_multiplier: 1.25,
        max_request_edge: 4_096,
        max_request_pixels: 16_777_216,
    };

    /// 图片 token 估算：`ceil((base + tiles × per_tile) × multiplier)`。
    ///
    /// 1818×1026 → ceil(1818/512)=4 × ceil(1026/512)=3 = 12 tile →
    /// ceil((1024 + 12×2048) × 1.25) = 32,000。该值与 Base64 字符数完全
    /// 无关（固定回归锚点；旧伪预算为 1,127,753 token）。
    pub fn image_tokens(&self, width: u32, height: u32) -> u64 {
        if width == 0 || height == 0 {
            return 0;
        }
        let tiles_x = (u64::from(width)).div_ceil(u64::from(self.tile_width));
        let tiles_y = (u64::from(height)).div_ceil(u64::from(self.tile_height));
        let tiles = tiles_x.saturating_mul(tiles_y);
        let raw = u64::from(self.base_tokens)
            .saturating_add(tiles.saturating_mul(u64::from(self.tokens_per_tile)));
        let scaled = (raw as f64 * f64::from(self.safety_multiplier)).ceil();
        scaled as u64
    }

    /// 请求尺寸是否超出 profile 上限（需生成等比缩放派生图）。
    pub fn exceeds_request_bounds(&self, width: u32, height: u32) -> bool {
        width > self.max_request_edge
            || height > self.max_request_edge
            || u64::from(width) * u64::from(height) > self.max_request_pixels
    }

    /// 等比缩放到请求上限内的目标尺寸（保持宽高比，至少 1×1）。
    pub fn scale_to_request_bounds(&self, width: u32, height: u32) -> (u32, u32) {
        if !self.exceeds_request_bounds(width, height) {
            return (width.max(1), height.max(1));
        }
        let edge_scale = if width.max(height) > self.max_request_edge {
            f64::from(self.max_request_edge) / f64::from(width.max(height))
        } else {
            1.0
        };
        let pixel_scale = if u64::from(width) * u64::from(height) > self.max_request_pixels {
            (self.max_request_pixels as f64 / (f64::from(width) * f64::from(height))).sqrt()
        } else {
            1.0
        };
        let scale = edge_scale.min(pixel_scale).min(1.0);
        let scaled_width = ((f64::from(width) * scale).floor() as u32).max(1);
        let scaled_height = ((f64::from(height) * scale).floor() as u32).max(1);
        (scaled_width, scaled_height)
    }
}

#[cfg(test)]
mod vision_budget_tests {
    use super::*;

    #[test]
    fn deepseek_regression_fixture_is_exactly_32000_tokens() {
        let profile = VisionBudgetProfile::DEEPSEEK_VISION_EXP_V1;
        // 固定回归样本：1818×1026 / 3,383,259 bytes / base64 4,511,012 chars。
        assert_eq!(profile.image_tokens(1818, 1026), 32_000);
    }

    #[test]
    fn image_tokens_independent_of_base64_length() {
        let profile = VisionBudgetProfile::DEEPSEEK_VISION_EXP_V1;
        let a = profile.image_tokens(1818, 1026);
        // 尺寸相同即估算相同；Base64 体积不参与。
        assert_eq!(profile.image_tokens(1818, 1026), a);
        assert_eq!(a, 32_000);
        assert!(a < 40_000, "tile 上界必须远离 Base64 伪预算 1,127,753");
    }

    #[test]
    fn scaling_respects_bounds_and_ratio() {
        let profile = VisionBudgetProfile::DEEPSEEK_VISION_EXP_V1;
        let (w, h) = profile.scale_to_request_bounds(8000, 6000);
        assert!(w <= profile.max_request_edge && h <= profile.max_request_edge);
        assert!(u64::from(w) * u64::from(h) <= profile.max_request_pixels);
        let ratio = f64::from(w) / f64::from(h);
        assert!((ratio - 8.0 / 6.0).abs() < 0.05, "ratio drifted: {ratio}");
    }
}
