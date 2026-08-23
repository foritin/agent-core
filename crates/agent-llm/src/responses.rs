//! OpenAI Responses API Provider（`POST {base}/responses`）。
//!
//! 与 Chat Completions 是**两套协议**，不是同一协议的两种写法：
//!
//! | | Chat Completions | Responses |
//! | --- | --- | --- |
//! | 输入 | `messages[]` | `input[]`（异构 item 数组） |
//! | 系统提示 | `messages[0].role="system"` | 顶层 `instructions` |
//! | 输出上限 | `max_tokens` | `max_output_tokens` |
//! | 输出 | `choices[0].message` | `output[]`（message / reasoning / function_call 平级） |
//! | 工具定义 | `tools[].function.{...}` 嵌套 | `tools[].{type,name,parameters}` 扁平 |
//! | 工具回传 | `role:"tool"` + `tool_call_id` | `function_call_output` item + **`call_id`** |
//! | 流式 | 单一 chunk 靠字段判类型，`[DONE]` 收尾 | 类型化事件，`response.completed` 收尾，**无 `[DONE]`** |
//!
//! 采用 **无状态**策略（`store: false`，不使用 `previous_response_id`）：整段历史
//! 每轮重放。原因是我们的会话历史保存在本地 SQLite，服务端会话反而会引入
//! 双份真相；而且火山方舟这类实现只有服务端会话、没有加密回传，两条路无法统一。
//!
//! 两个高频 400 由 [`sanitize_input_items`] 在构造请求时静态挡掉：
//! - `Item 'rs_…' of type 'reasoning' was provided without its required following item.`
//! - `No tool call found for function call output with call_id …`

use agent_contract::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, HostedToolFormat,
    HostedToolSpec, LlmProvider, Message, Role, StopReason, StreamEvent, ToolSpec, Usage,
};
use agent_error::{Error, Result};
use serde_json::{json, Value};

use std::time::Duration;

use crate::openai::normalize_api_error;
use crate::url::openai_api_root;

/// DeepSeek's stateless Responses endpoint accepts completed provider search items on replay.
/// Keep a single malformed or unexpectedly large server item from poisoning every later turn.
const MAX_REPLAYABLE_HOSTED_WEB_ITEM_BYTES: usize = 512 * 1024;

/// 推理内容的处理策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningMode {
    /// 不索取、也不回传 reasoning item。兼容性最好。
    ///
    /// 代价是多轮工具调用之间模型的思维链不连续；好处是永远不会触发
    /// "孤儿 reasoning" 400，且对不支持 `include` 的实现（如火山方舟）同样可用。
    #[default]
    Drop,
    /// 索取 `reasoning.encrypted_content` 并在下一轮原样回传。
    ///
    /// 仅 OpenAI 官方与 xAI 支持。加密块存放在 [`ContentBlock::Thinking::signature`] 里，
    /// 编解码见 [`encode_reasoning_signature`] / [`decode_reasoning_signature`]。
    EncryptedReplay,
    /// DeepSeek Responses thinking 模式：reasoning 以**明文 `reasoning_text` 内容块**返回，
    /// 下一轮必须把 `reasoning` item 连同其 `content` 原样回传，否则 400
    /// （`The reasoning_text in the thinking mode must be passed back to the API`）。
    ///
    /// 与 OpenAI 的 `encrypted_content` 不同，这里没有加密块：明文链直接保存在
    /// [`ContentBlock::Thinking::thinking`]，`signature` 恒为 `None`。
    PlaintextReplay,
    /// 火山方舟（Ark）Responses：reasoning 以 **`summary[].summary_text`** 返回，
    /// 没有 `content` 也没有 `encrypted_content`。Ark 不强制回传 reasoning item，
    /// 但回传会被接受；R-Code 仍把摘要保存在
    /// [`ContentBlock::Thinking::thinking`] 并在下一轮原样回传，保持多轮工具调用
    /// 之间的思维链不丢（对齐 Ark 的 `response.reasoning_summary_text.delta`）。
    SummaryReplay,
}

pub struct ResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
    reasoning: ReasoningMode,
    max_context_tokens: u32,
    max_output_tokens: u32,
    deepseek_automatic_cache: bool,
    ark_effort: bool,
    supports_vision: bool,
    provider_name: &'static str,
}

impl ResponsesProvider {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .user_agent(concat!("R-Code/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("responses provider client"),
            api_key,
            base_url,
            model,
            reasoning: ReasoningMode::default(),
            max_context_tokens: 200_000,
            max_output_tokens: 0,
            deepseek_automatic_cache: false,
            ark_effort: false,
            supports_vision: true,
            provider_name: "openai_responses",
        }
    }

    /// DeepSeek Responses 仍使用无状态全历史重放，但能力、上下文窗口与缓存
    /// usage 语义必须保留 DeepSeek 身份，不能退化为通用 Responses provider。
    pub fn new_deepseek(api_key: String, model: String, base_url: String) -> Self {
        let lowercase_model = model.trim().to_ascii_lowercase();
        let is_v4 =
            lowercase_model.starts_with("deepseek-v4-") || lowercase_model == "deepseek-chat";
        let mut provider = Self::new(api_key, model, base_url);
        provider.max_context_tokens = 1_000_000;
        // DeepSeek V4 的单次输出上限是 393_216（API 报错口径）；非 V4 未声明。
        provider.max_output_tokens = if is_v4 { 393_216 } else { 0 };
        provider.deepseek_automatic_cache = true;
        // 目录级真值（deepseek.rs）：只有 vision 实验模型支持图片输入。
        provider.supports_vision =
            crate::deepseek::deepseek_model_supports_vision(&lowercase_model);
        provider.provider_name = "deepseek_responses";
        // DeepSeek Responses 的 thinking 模式返回明文 reasoning_text，下一轮必须原样
        // 回传，否则 400（见 ReasoningMode::PlaintextReplay）。
        provider.reasoning = ReasoningMode::PlaintextReplay;
        provider
    }

    /// Ark Coding Plan / Agent Plan 的 Responses 口。
    ///
    /// 与 DeepSeek 一样走 `store=false` 的无状态全历史重放。Ark 的 reasoning 是
    /// 明文 summary（非加密、非必回传），用 [`ReasoningMode::SummaryReplay`] 保存
    /// 并回传。Coding Plan 上下文 256K，Agent Plan 上下文 1M。
    pub fn new_ark(api_key: String, model: String, base_url: String, kind: &str) -> Self {
        let kind = kind.trim().to_ascii_lowercase();
        let max_context_tokens = if kind == "ark_agent" {
            1_048_576
        } else {
            256_000
        };
        let mut provider = Self::new(api_key, model.clone(), base_url);
        provider.max_context_tokens = max_context_tokens;
        provider.max_output_tokens = 0;
        provider.reasoning = ReasoningMode::SummaryReplay;
        provider.ark_effort = true;
        provider.supports_vision = model.to_ascii_lowercase().contains("doubao-seed");
        provider.provider_name = "ark_responses";
        provider
    }

    /// 开启加密 reasoning 回传。只对 OpenAI / xAI 有意义。
    pub fn with_reasoning(mut self, mode: ReasoningMode) -> Self {
        self.reasoning = mode;
        self
    }

    /// 覆盖能力声明里的上下文窗口。
    pub fn with_max_context_tokens(mut self, tokens: u32) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", openai_api_root(&self.base_url))
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> Value {
        // Responses has no `thinking` field.  For DeepSeek only, translate R-Code's local
        // thinking selection into the vendor's native reasoning effort: disabled means no
        // reasoning, while enabled/adaptive starts at high unless the user/governor supplied an
        // explicit effort.  Never send the local `adaptive` marker over the wire.  Non-DeepSeek
        // providers retain the pre-existing behavior of considering `reasoning_effort` only.
        let is_deepseek_v4 = self.deepseek_automatic_cache
            && request.model.to_ascii_lowercase().contains("deepseek-v4");
        let effort = if is_deepseek_v4 {
            match request.inference.thinking.as_deref() {
                Some("disabled") => Some("none"),
                Some("enabled" | "adaptive") => request
                    .inference
                    .reasoning_effort
                    .as_deref()
                    .or(Some("high")),
                _ => request.inference.reasoning_effort.as_deref(),
            }
        } else if self.ark_effort {
            // Ark 探针冻结事实（2026-08-16，glm-5.3 / /api/plan/v3/responses）：
            // reasoning_effort 只接受 low/medium/high/xhigh/max；none 与 minimal 直接
            // 400。Ark 没有关闭 reasoning 的 wire 值（不传参数时服务端仍会输出
            // reasoning summary），因此 disabled 只能退化为「不发送 reasoning 参数」。
            const ARK_EFFORT_VOCAB: &[&str] = &["low", "medium", "high", "xhigh", "max"];
            let explicit = request.inference.reasoning_effort.as_deref();
            match explicit {
                Some(value) if ARK_EFFORT_VOCAB.contains(&value) => Some(value),
                Some("none" | "minimal") => None,
                Some(_) => None,
                None => match request.inference.thinking.as_deref() {
                    Some("enabled") => Some("high"),
                    Some("disabled" | "adaptive") | None => None,
                    _ => None,
                },
            }
        } else {
            request.inference.reasoning_effort.as_deref()
        };

        // DeepSeek Responses 的 thinking 模式要求工具调用轮把 reasoning item 回传
        // （空 reasoning_text 可接受，缺 item 400 "must be passed back"）。历史
        // 可能因旧版本落盘或压缩而丢块，发送前补一个空 reasoning item 兜底；
        // 本轮 thinking 关闭（effort=none）时不注入，保持请求字节稳定。
        let deepseek_thinking =
            self.reasoning == ReasoningMode::PlaintextReplay && effort != Some("none");

        let mut input: Vec<Value> = Vec::new();
        for message in &request.messages {
            let mut items = message_to_items(message, self.reasoning);
            if deepseek_thinking
                && message.role == Role::Assistant
                && message.content.iter().any(ContentBlock::is_tool_use)
                && !message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Thinking { .. }))
            {
                items.insert(
                    0,
                    json!({
                        "type": "reasoning",
                        "content": [
                            { "type": "reasoning_text", "text": "" }
                        ],
                    }),
                );
            }
            input.extend(items);
        }
        let input = sanitize_input_items(input);

        let mut body = json!({
            "model": request.model,
            "input": input,
            "max_output_tokens": request.max_tokens,
            // 本地保存历史 + 每轮全量重放，不依赖服务端会话
            "store": false,
        });
        if let Some(system) = &request.system {
            // Responses 的 instructions 不会被上一轮继承，每次都要带
            body["instructions"] = json!(system);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(effort) = effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        if let Some(verbosity) = request.inference.verbosity.as_deref() {
            body["text"] = json!({ "verbosity": verbosity });
        }
        if stream {
            body["stream"] = json!(true);
        }
        let mut tools = request
            .tools
            .iter()
            .map(tool_to_responses)
            .collect::<Vec<_>>();
        tools.extend(
            request
                .hosted_tools
                .iter()
                .filter_map(hosted_tool_to_responses),
        );
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }
        if self.reasoning == ReasoningMode::EncryptedReplay {
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        body
    }
}

#[async_trait::async_trait]
impl LlmProvider for ResponsesProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let body = self.build_body(&request, false);
        let resp = crate::openai::send_with_retry(
            &self.client,
            &self.responses_url(),
            &self.api_key,
            &body,
        )
        .await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("invalid response json: {e}")))?;
        parse_responses_response_with_cache(&value, self.reasoning, self.deepseek_automatic_cache)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let body = self.build_body(&request, true);
        let resp = crate::openai::send_with_retry(
            &self.client,
            &self.responses_url(),
            &self.api_key,
            &body,
        )
        .await?;
        Ok(Box::pin(parse_responses_sse(
            crate::openai::watch_sse_idle(
                resp.bytes_stream(),
                crate::openai::DEFAULT_STREAM_IDLE_TIMEOUT,
            ),
            self.deepseek_automatic_cache,
            self.reasoning,
        )))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: self.supports_vision,
            supports_prompt_caching: self.deepseek_automatic_cache,
            max_context_tokens: self.max_context_tokens,
            max_output_tokens: self.max_output_tokens,
        }
    }

    fn echoes_reasoning(&self) -> bool {
        self.name() == "deepseek_responses"
    }

    fn name(&self) -> &str {
        self.provider_name
    }
}

// ── reasoning 签名编解码 ──────────────────────────────────────────
//
// 内部消息模型没有"reasoning item"这一类内容块，复用 `ContentBlock::Thinking`：
// `thinking` 放摘要文本（可能为空），`signature` 放回传所需的 id + 加密块。
// 用前缀 + 冒号分隔而不是 JSON，避免二次转义；id 形如 `rs_xxx`、加密块是
// base64 字符集，两者都不含冒号。

const REASONING_SIG_PREFIX: &str = "resp-reasoning:v1";

pub fn encode_reasoning_signature(id: &str, encrypted: &str) -> String {
    format!("{REASONING_SIG_PREFIX}:{id}:{encrypted}")
}

pub fn decode_reasoning_signature(signature: &str) -> Option<(String, String)> {
    let rest = signature
        .strip_prefix(REASONING_SIG_PREFIX)?
        .strip_prefix(':')?;
    let (id, encrypted) = rest.split_once(':')?;
    if encrypted.is_empty() {
        return None;
    }
    Some((id.to_string(), encrypted.to_string()))
}

// ── 请求转换 ──────────────────────────────────────────────────

/// 一条内部消息 → 若干个 Responses input item。**保持块顺序**，
/// reasoning 与其配对产物的相邻关系依赖于此。
pub fn message_to_items(msg: &Message, reasoning: ReasoningMode) -> Vec<Value> {
    let mut items = Vec::new();

    match msg.role {
        Role::User => {
            // 工具结果：每个 ToolResult 是一个独立 item，不是 message
            let mut has_tool_result = false;
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                {
                    has_tool_result = true;
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": content,
                    }));
                }
            }
            if !has_tool_result {
                let content = msg
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            Some(json!({ "type": "input_text", "text": text }))
                        }
                        ContentBlock::Image { source } => Some(json!({
                            "type": "input_image",
                            "image_url": format!("data:{};base64,{}", source.media_type, source.data),
                        })),
                        ContentBlock::File { source }
                            if source.media_type.starts_with("image/") && source.data.is_some() =>
                        {
                            Some(json!({
                                "type": "input_image",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    source.media_type,
                                    source.data.as_deref().unwrap_or_default()
                                ),
                            }))
                        }
                        ContentBlock::File { source }
                            if source.media_type == "application/pdf" && source.data.is_some() =>
                        {
                            Some(json!({
                                "type": "input_file",
                                "filename": source.name,
                                "file_data": format!(
                                    "data:{};base64,{}",
                                    source.media_type,
                                    source.data.as_deref().unwrap_or_default()
                                ),
                            }))
                        }
                        ContentBlock::File { source } if source.kind == "text" => Some(json!({
                            "type": "input_text",
                            "text": format!(
                                "\n\n--- Attached file: {} ({}) ---\n{}\n--- End attached file: {} ---",
                                source.name,
                                source.media_type,
                                source.text.as_deref().unwrap_or_default(),
                                source.name,
                            ),
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                items.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": content,
                }));
            }
        }
        Role::Assistant => {
            let mut text_buffer = String::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        match reasoning {
                            ReasoningMode::EncryptedReplay => {
                                let Some((id, encrypted)) =
                                    signature.as_deref().and_then(decode_reasoning_signature)
                                else {
                                    continue;
                                };
                                // 文本先落盘，保证 reasoning 排在它的配对产物之前
                                flush_assistant_text(&mut text_buffer, &mut items);
                                let _ = thinking;
                                items.push(json!({
                                    "type": "reasoning",
                                    "id": id,
                                    "summary": [],
                                    "encrypted_content": encrypted,
                                }));
                            }
                            ReasoningMode::PlaintextReplay => {
                                // DeepSeek Responses 的 thinking 模式要求把明文 reasoning_text
                                // 原样回传；reasoning item 必须紧邻它的配对产物之前。工具调用
                                // 轮次可能返回空 reasoning_text，空字符串同样有协议语义，
                                // 丢弃会让下一轮请求 400，因此这里不能按 trim 后是否为空过滤。
                                flush_assistant_text(&mut text_buffer, &mut items);
                                items.push(json!({
                                    "type": "reasoning",
                                    "content": [
                                        { "type": "reasoning_text", "text": thinking.as_str() }
                                    ],
                                }));
                            }
                            ReasoningMode::SummaryReplay => {
                                // Ark 接受最小 summary-only 的 reasoning item 回传
                                // （2026-08-16 探针冻结：id / status 均可省略）。
                                if !thinking.is_empty() {
                                    flush_assistant_text(&mut text_buffer, &mut items);
                                    items.push(json!({
                                        "type": "reasoning",
                                        "summary": [
                                            { "type": "summary_text", "text": thinking.as_str() }
                                        ],
                                    }));
                                }
                            }
                            ReasoningMode::Drop => {}
                        }
                    }
                    ContentBlock::Text { text } => text_buffer.push_str(text),
                    ContentBlock::ToolUse { id, name, input } => {
                        flush_assistant_text(&mut text_buffer, &mut items);
                        items.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": input.to_string(),
                        }));
                    }
                    ContentBlock::Custom { type_name, data }
                        if is_hosted_web_item_type(type_name) =>
                    {
                        // Responses hosted tools are output items, not function calls. DeepSeek's
                        // stateless endpoint asks clients to replay completed items as-is. Failed,
                        // partial, id-less or abnormally large items are not valid continuation
                        // state and can make every later request fail with HTTP 400.
                        if let Some(item) = replayable_hosted_web_item(type_name, data) {
                            flush_assistant_text(&mut text_buffer, &mut items);
                            items.push(item);
                        }
                    }
                    _ => {}
                }
            }
            flush_assistant_text(&mut text_buffer, &mut items);
        }
    }

    items
}

fn flush_assistant_text(buffer: &mut String, items: &mut Vec<Value>) {
    if buffer.is_empty() {
        return;
    }
    items.push(json!({
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": buffer }],
    }));
    buffer.clear();
}

/// 静态挡掉两个高频 400。
///
/// 1. `function_call_output` 的 `call_id` 必须在前面出现过同 id 的 `function_call`，
///    否则整个请求 400。历史被压缩、或上一轮工具调用被中止时会出现落单的结果。
/// 2. `reasoning` item 后面必须紧跟它的配对产物（`function_call` 或 assistant
///    `message`）。落在数组末尾、或后面只剩 user 消息的 reasoning 一律丢弃。
pub fn sanitize_input_items(items: Vec<Value>) -> Vec<Value> {
    let mut seen_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut without_orphan_outputs = Vec::with_capacity(items.len());

    for item in items {
        match item_type(&item) {
            "function_call" => {
                if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                    seen_calls.insert(call_id.to_string());
                }
                without_orphan_outputs.push(item);
            }
            "function_call_output" => {
                let paired = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .is_some_and(|call_id| seen_calls.contains(call_id));
                if paired {
                    without_orphan_outputs.push(item);
                }
            }
            _ => without_orphan_outputs.push(item),
        }
    }

    let mut result = Vec::with_capacity(without_orphan_outputs.len());
    for (index, item) in without_orphan_outputs.iter().enumerate() {
        if item_type(item) == "reasoning" {
            let followed_by_product = without_orphan_outputs
                .get(index + 1)
                .is_some_and(is_reasoning_product);
            if !followed_by_product {
                continue;
            }
        }
        result.push(item.clone());
    }
    result
}

fn item_type(item: &Value) -> &str {
    item.get("type").and_then(|v| v.as_str()).unwrap_or("")
}

fn is_hosted_web_item_type(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_call" | "web_fetch_call" | "web_extractor_call"
    )
}

fn replayable_hosted_web_item(type_name: &str, data: &Value) -> Option<Value> {
    if !is_hosted_web_item_type(type_name) {
        return None;
    }
    let mut item = data.as_object()?.clone();
    let has_id = item
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty());
    let completed = item.get("status").and_then(Value::as_str) == Some("completed");
    if !has_id || !completed {
        return None;
    }
    item.insert("type".to_string(), Value::String(type_name.to_string()));
    let item = Value::Object(item);
    (serde_json::to_vec(&item).ok()?.len() <= MAX_REPLAYABLE_HOSTED_WEB_ITEM_BYTES).then_some(item)
}

fn hosted_web_tool_name(kind: &str) -> &'static str {
    match kind {
        "web_fetch_call" | "web_extractor_call" => "web_fetch",
        _ => "web_search",
    }
}

/// reasoning 的合法后继：函数调用，或助手消息。
fn is_reasoning_product(item: &Value) -> bool {
    match item_type(item) {
        kind if kind == "function_call" || is_hosted_web_item_type(kind) => true,
        "message" => item.get("role").and_then(|v| v.as_str()) == Some("assistant"),
        _ => false,
    }
}

/// 工具定义。Responses 是扁平结构，没有 `function` 这一层嵌套。
fn tool_to_responses(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        // 我们的工具 schema 没有按 strict 模式写（缺 additionalProperties:false），
        // 显式关掉，避免服务端按严格模式校验后拒绝。
        "strict": false,
    })
}

fn hosted_tool_to_responses(tool: &HostedToolSpec) -> Option<Value> {
    match tool {
        // OpenAI, xAI, Azure, DeepSeek and Ark use the standard Responses schema. DashScope uses
        // the same search type but a dedicated extractor type for full-page reads.
        HostedToolSpec::WebSearch {
            format: HostedToolFormat::Standard | HostedToolFormat::DashScope,
            ..
        } => Some(json!({"type": "web_search"})),
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::DashScope,
            ..
        } => Some(json!({"type": "web_extractor"})),
        HostedToolSpec::WebSearch {
            format: HostedToolFormat::OpenRouter,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(openrouter_server_tool(
            "openrouter:web_search",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::OpenRouter,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(openrouter_server_tool(
            "openrouter:web_fetch",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        // Standard Responses does not define a standalone web-fetch type. OpenAI/xAI/Azure
        // expose page opening through `web_search`; Anthropic maps WebFetch in its own adapter.
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::Standard,
            ..
        } => None,
    }
}

fn openrouter_server_tool(
    tool_type: &str,
    max_uses: Option<u32>,
    allowed_domains: &[String],
    blocked_domains: &[String],
) -> Value {
    let mut parameters = serde_json::Map::new();
    if let Some(max_uses) = max_uses {
        parameters.insert("max_uses".into(), json!(max_uses));
    }
    if !allowed_domains.is_empty() {
        parameters.insert("allowed_domains".into(), json!(allowed_domains));
    }
    if !blocked_domains.is_empty() {
        let key = if tool_type == "openrouter:web_search" {
            "excluded_domains"
        } else {
            "blocked_domains"
        };
        parameters.insert(key.into(), json!(blocked_domains));
    }

    let mut tool = json!({"type": tool_type});
    if !parameters.is_empty() {
        tool["parameters"] = Value::Object(parameters);
    }
    tool
}

// ── 响应解析 ──────────────────────────────────────────────────

pub fn parse_responses_response(
    value: &Value,
    reasoning: ReasoningMode,
) -> Result<CompletionResponse> {
    parse_responses_response_with_cache(value, reasoning, false)
}

fn parse_responses_response_with_cache(
    value: &Value,
    reasoning: ReasoningMode,
    deepseek_automatic_cache: bool,
) -> Result<CompletionResponse> {
    let output = value
        .get("output")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Provider("response missing output".into()))?;

    let mut content = Vec::new();
    let mut saw_tool_call = false;

    for item in output {
        match item_type(item) {
            "reasoning" => {
                match reasoning {
                    ReasoningMode::EncryptedReplay => {
                        let Some(encrypted) =
                            item.get("encrypted_content").and_then(|v| v.as_str())
                        else {
                            continue;
                        };
                        let id = item.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                        content.push(ContentBlock::Thinking {
                            thinking: reasoning_summary_text(item),
                            signature: Some(encode_reasoning_signature(id, encrypted)),
                        });
                    }
                    ReasoningMode::PlaintextReplay => {
                        // DeepSeek V4 的工具调用轮次会返回空 reasoning_text，下一轮必须
                        // 把这个空 item 原样回传，否则 400。解析端不能丢弃空 reasoning。
                        content.push(ContentBlock::Thinking {
                            thinking: reasoning_text_of(item),
                            signature: None,
                        });
                    }
                    ReasoningMode::SummaryReplay => {
                        // Ark 的 reasoning item 只有 summary[]，没有 content 或
                        // encrypted_content。摘要落入 Thinking，下一轮按原样回传。
                        let thinking = reasoning_summary_text(item);
                        if !thinking.is_empty() {
                            content.push(ContentBlock::Thinking {
                                thinking,
                                signature: None,
                            });
                        }
                    }
                    ReasoningMode::Drop => {}
                }
            }
            "function_call" => {
                saw_tool_call = true;
                // 回传时必须用 call_id，不是 item 的 id
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                content.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input: parse_arguments(item.get("arguments").and_then(|v| v.as_str())),
                });
            }
            kind if is_hosted_web_item_type(kind) => {
                if let Some(block) = responses_hosted_item_to_custom(item) {
                    content.push(block);
                }
            }
            "message" => {
                let text = output_text_of(item);
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }
            _ => {}
        }
    }

    Ok(CompletionResponse {
        content,
        stop_reason: response_stop_reason(value, saw_tool_call),
        usage: parse_usage(value.get("usage"), deepseek_automatic_cache),
    })
}

/// `arguments` 永远是字符串化 JSON，不是对象。
fn parse_arguments(raw: Option<&str>) -> Value {
    raw.filter(|s| !s.trim().is_empty())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(Value::Null)
}

fn reasoning_summary_text(item: &Value) -> String {
    item.get("summary")
        .and_then(|v| v.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

/// DeepSeek Responses 的 reasoning 明文放在 `content[].reasoning_text`，而不是
/// OpenAI 的 `summary[]` 或 `encrypted_content`。
fn reasoning_text_of(item: &Value) -> String {
    item.get("content")
        .and_then(|v| v.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter(|part| item_type(part) == "reasoning_text")
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn output_text_of(item: &Value) -> String {
    item.get("content")
        .and_then(|v| v.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter(|part| item_type(part) == "output_text")
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn response_stop_reason(value: &Value, saw_tool_call: bool) -> StopReason {
    if saw_tool_call {
        return StopReason::ToolUse;
    }
    match value.get("status").and_then(|v| v.as_str()) {
        Some("completed") | None => StopReason::EndTurn,
        Some("incomplete") => {
            let reason = value
                .pointer("/incomplete_details/reason")
                .and_then(|v| v.as_str())
                .unwrap_or("incomplete");
            if reason == "max_output_tokens" {
                StopReason::MaxTokens
            } else {
                StopReason::Other(reason.to_string())
            }
        }
        Some(other) => StopReason::Other(other.to_string()),
    }
}

fn parse_usage(usage: Option<&Value>, deepseek_automatic_cache: bool) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let cache_read_tokens = usage
        .get("prompt_cache_hit_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .pointer("/input_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
        })
        .map(|tokens| tokens as u32);
    let cache_write_tokens = usage
        .get("prompt_cache_miss_tokens")
        .and_then(Value::as_u64)
        .map(|tokens| tokens as u32)
        .or_else(|| {
            (deepseek_automatic_cache && cache_read_tokens.is_some())
                .then(|| input_tokens.saturating_sub(cache_read_tokens.unwrap_or(0)))
        });
    Usage {
        input_tokens,
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_read_tokens,
        cache_write_tokens,
    }
}

// ── SSE 流解析 ────────────────────────────────────────────────

/// 流内可变状态。
#[derive(Default)]
struct StreamState {
    /// `item_id` → `(call_id, name)`。delta 事件只带 item_id，回传要 call_id。
    calls: std::collections::HashMap<String, (String, String)>,
    /// 已发出 ToolUseComplete 的 call_id，防止 `.done` 与 `response.completed` 重复发。
    completed_calls: std::collections::HashSet<String>,
    /// 已发出 HostedToolUse / HostedToolResult 的 Responses web-search item id。
    announced_hosted_searches: std::collections::HashSet<String>,
    completed_hosted_searches: std::collections::HashSet<String>,
    /// Reasoning item ids that already emitted streamed deltas or a completed summary.
    emitted_reasoning_items: std::collections::HashSet<String>,
    /// 上一个 `sequence_number`，用于断线重连后的去重。
    last_sequence: Option<u64>,
    saw_tool_call: bool,
    stopped: bool,
    deepseek_automatic_cache: bool,
    /// Which reasoning wire format to expect in completed reasoning items. DeepSeek
    /// Responses puts plaintext in `content[].reasoning_text`; OpenAI/xAI put a
    /// summary in `summary[]` (or an encrypted block in `encrypted_content`).
    reasoning: ReasoningMode,
}

fn parse_responses_sse(
    byte_stream: impl futures::Stream<
        Item = std::result::Result<bytes::Bytes, crate::openai::SseChunkError>,
    >,
    deepseek_automatic_cache: bool,
    reasoning: ReasoningMode,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;
    let mut buffer = String::new();
    let mut state = StreamState {
        deepseek_automatic_cache,
        reasoning,
        ..StreamState::default()
    };

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(crate::openai::SseChunkError::IdleTimeout) => {
                    if !state.stopped {
                        state.stopped = true;
                        return vec![StreamEvent::Stop {
                            reason: StopReason::Other(
                                crate::openai::STREAM_IDLE_TIMEOUT_REASON.to_string(),
                            ),
                        }];
                    }
                    return Vec::new();
                }
                Err(crate::openai::SseChunkError::Transport(_)) => return Vec::new(),
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            let mut events = Vec::new();
            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..pos + 1).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // `event:` 行可以忽略：负载 JSON 自带 "type" 字段
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() {
                    continue;
                }
                // Responses 规范里没有 [DONE]，但部分网关照抄了 Chat Completions
                if data == "[DONE]" {
                    if !state.stopped {
                        state.stopped = true;
                        events.push(StreamEvent::Stop {
                            reason: if state.saw_tool_call {
                                StopReason::ToolUse
                            } else {
                                StopReason::EndTurn
                            },
                        });
                    }
                    continue;
                }
                events.extend(parse_one_responses_event(data, &mut state));
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn parse_one_responses_event(data: &str, state: &mut StreamState) -> Vec<StreamEvent> {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };

    // 断线重连会重放已处理的事件；sequence_number 单调递增，据此去重
    if let Some(sequence) = value.get("sequence_number").and_then(|v| v.as_u64()) {
        if state.last_sequence.is_some_and(|last| sequence <= last) {
            return Vec::new();
        }
        state.last_sequence = Some(sequence);
    }

    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut events = Vec::new();

    match event_type {
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                if !delta.is_empty() {
                    if let Some(item_id) = value.get("item_id").and_then(Value::as_str) {
                        state.emitted_reasoning_items.insert(item_id.to_string());
                    }
                    events.push(StreamEvent::ReasoningDelta {
                        text: delta.to_string(),
                    });
                }
            }
        }
        "response.output_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                if !delta.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: delta.to_string(),
                    });
                }
            }
        }
        "response.output_item.added" => {
            if let Some(item) = value.get("item") {
                match item_type(item) {
                    "function_call" => {
                        state.saw_tool_call = true;
                        let item_id = item
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        state.calls.insert(item_id, (call_id.clone(), name.clone()));
                        events.push(StreamEvent::ToolUseStart { id: call_id, name });
                    }
                    kind if is_hosted_web_item_type(kind) => {
                        announce_hosted_web_item(item, state, &mut events)
                    }
                    _ => {}
                }
            }
        }
        "response.function_call_arguments.delta" => {
            if let (Some(item_id), Some(delta)) = (
                value.get("item_id").and_then(|v| v.as_str()),
                value.get("delta").and_then(|v| v.as_str()),
            ) {
                if let Some((call_id, _)) = state.calls.get(item_id) {
                    events.push(StreamEvent::ToolUseDelta {
                        id: call_id.clone(),
                        input_json: delta.to_string(),
                    });
                }
            }
        }
        // `.done` 携带完整字符串，作为唯一可信来源；delta 只用于 UI 增量显示
        "response.output_item.done" => {
            if let Some(item) = value.get("item") {
                match item_type(item) {
                    "reasoning" => emit_completed_reasoning_item(item, state, &mut events),
                    "function_call" => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if !call_id.is_empty() && state.completed_calls.insert(call_id.clone()) {
                            events.push(StreamEvent::ToolUseComplete {
                                id: call_id,
                                input: parse_arguments(
                                    item.get("arguments").and_then(|v| v.as_str()),
                                ),
                            });
                        }
                    }
                    kind if is_hosted_web_item_type(kind) => {
                        complete_hosted_web_item(item, state, &mut events)
                    }
                    _ => {}
                }
            }
        }
        "response.completed" | "response.incomplete" => {
            if let Some(response) = value.get("response") {
                // 少数实现不发 output_item.done，只在终帧给出完整 output
                events.extend(drain_final_output(response, state));
                events.push(StreamEvent::Usage(parse_usage(
                    response.get("usage"),
                    state.deepseek_automatic_cache,
                )));
                if !state.stopped {
                    state.stopped = true;
                    events.push(StreamEvent::Stop {
                        reason: response_stop_reason(response, state.saw_tool_call),
                    });
                }
            } else if !state.stopped {
                state.stopped = true;
                events.push(StreamEvent::Stop {
                    reason: StopReason::EndTurn,
                });
            }
        }
        "response.failed" | "error" if !state.stopped => {
            state.stopped = true;
            let message = responses_stream_error_message(&value);
            events.push(StreamEvent::Stop {
                reason: StopReason::Other(message),
            });
        }
        // 未知的 response.* 事件一律忽略：OpenAI 持续新增事件类型，
        // 各家兼容实现也只做子集，报错会让流白白中断。
        _ => {}
    }

    events
}

fn responses_stream_error_message(value: &Value) -> String {
    if let Some(error) = value
        .pointer("/response/error")
        .or_else(|| value.get("error"))
    {
        return normalize_api_error(&error.to_string(), "");
    }

    // In a top-level `{"type":"error", ...}` frame, `type` names the SSE event rather than the
    // API error class. Copy only the documented safe scalar fields and map `error_type`, when a
    // compatible gateway provides it, into the normal OpenAI error shape.
    let mut error = serde_json::Map::new();
    for field in ["message", "code", "param"] {
        if let Some(value) = value.get(field) {
            error.insert(field.to_string(), value.clone());
        }
    }
    if let Some(error_type) = value.get("error_type") {
        error.insert("type".to_string(), error_type.clone());
    }
    normalize_api_error(&Value::Object(error).to_string(), "")
}

/// 终帧兜底：补发没有通过 `output_item.done` 走完的工具调用。
fn drain_final_output(response: &Value, state: &mut StreamState) -> Vec<StreamEvent> {
    let Some(output) = response.get("output").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut events = Vec::new();
    for item in output {
        match item_type(item) {
            "reasoning" => emit_completed_reasoning_item(item, state, &mut events),
            "function_call" => {
                let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                if call_id.is_empty() || !state.completed_calls.insert(call_id.to_string()) {
                    continue;
                }
                state.saw_tool_call = true;
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                events.push(StreamEvent::ToolUseStart {
                    id: call_id.to_string(),
                    name,
                });
                events.push(StreamEvent::ToolUseComplete {
                    id: call_id.to_string(),
                    input: parse_arguments(item.get("arguments").and_then(|v| v.as_str())),
                });
            }
            kind if is_hosted_web_item_type(kind) => {
                complete_hosted_web_item(item, state, &mut events)
            }
            _ => {}
        }
    }
    events
}

fn emit_completed_reasoning_item(
    item: &Value,
    state: &mut StreamState,
    events: &mut Vec<StreamEvent>,
) {
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("__reasoning_without_id__");
    if !state.emitted_reasoning_items.insert(item_id.to_string()) {
        return;
    }
    let text = match state.reasoning {
        ReasoningMode::PlaintextReplay => reasoning_text_of(item),
        _ => reasoning_summary_text(item),
    };
    // PlaintextReplay（DeepSeek Responses）必须保留空 reasoning 的“存在性”：工具调用
    // 轮次可能返回空 reasoning_text，上层需要据此在下一轮回传一个空 reasoning item。
    // 其余模式只在有可见摘要时才发事件，避免制造无意义的空事件。
    if state.reasoning == ReasoningMode::PlaintextReplay || !text.is_empty() {
        events.push(StreamEvent::ReasoningDelta { text });
    }
}

fn responses_hosted_item_to_custom(item: &Value) -> Option<ContentBlock> {
    let mut data = item.as_object()?.clone();
    let type_name = data.remove("type")?.as_str()?.to_string();
    is_hosted_web_item_type(&type_name).then_some(ContentBlock::Custom {
        type_name,
        data: Value::Object(data),
    })
}

fn hosted_web_item_id(item: &Value) -> Option<String> {
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

fn hosted_web_item_input(item: &Value) -> Value {
    if item_type(item) == "web_extractor_call" {
        return json!({
            "goal": item.get("goal").cloned().unwrap_or(Value::Null),
            "urls": item.get("urls").cloned().unwrap_or_else(|| json!([])),
        });
    }
    let action = item.get("action").unwrap_or(&Value::Null);
    if let Some(query) = action.get("query").and_then(Value::as_str) {
        return json!({"query": query});
    }
    if let Some(url) = action.get("url").and_then(Value::as_str) {
        return json!({
            "action": action.get("type").and_then(Value::as_str).unwrap_or("open_page"),
            "url": url,
        });
    }
    json!({
        "action": action.get("type").and_then(Value::as_str).unwrap_or("search")
    })
}

fn announce_hosted_web_item(item: &Value, state: &mut StreamState, events: &mut Vec<StreamEvent>) {
    let Some(id) = hosted_web_item_id(item) else {
        return;
    };
    if state.announced_hosted_searches.insert(id.clone()) {
        events.push(StreamEvent::HostedToolUse {
            id,
            name: hosted_web_tool_name(item_type(item)).to_string(),
            input: hosted_web_item_input(item),
            // The completed output item is retained once in HostedToolResult below.
            provider_content: None,
        });
    }
}

fn complete_hosted_web_item(item: &Value, state: &mut StreamState, events: &mut Vec<StreamEvent>) {
    let Some(id) = hosted_web_item_id(item) else {
        return;
    };
    announce_hosted_web_item(item, state, events);
    if !state.completed_hosted_searches.insert(id.clone()) {
        return;
    }
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    let is_error = matches!(status, "failed" | "incomplete");
    events.push(StreamEvent::HostedToolResult {
        id,
        name: hosted_web_tool_name(item_type(item)).to_string(),
        output: if item_type(item) == "web_extractor_call" {
            json!({
                "status": status,
                "urls": item.get("urls").cloned().unwrap_or_else(|| json!([])),
            })
        } else {
            json!({"status": status})
        },
        is_error,
        provider_content: Some(item.clone()),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(base_url: &str) -> ResponsesProvider {
        ResponsesProvider::new("sk-test".into(), "gpt-5.6-sol".into(), base_url.into())
    }

    fn request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: "gpt-5.6-sol".into(),
            system: Some("be nice".into()),
            messages,
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 1024,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        }
    }

    fn long_tool_evidence() -> String {
        let mut evidence =
            "path=src/核心.rs; command=cargo test; exit=0; 证据=完整\n".repeat(4_096);
        evidence.push_str("__TOOL_EVIDENCE_TAIL_RESPONSES__");
        assert!(evidence.len() > 100_000);
        evidence
    }

    #[test]
    fn responses_url_respects_custom_version_segment() {
        assert_eq!(
            provider("https://api.deepseek.com").responses_url(),
            "https://api.deepseek.com/v1/responses"
        );
        assert_eq!(
            provider("https://api.openai.com").responses_url(),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            provider("https://api.x.ai/v1").responses_url(),
            "https://api.x.ai/v1/responses"
        );
        assert_eq!(
            provider("https://ark.cn-beijing.volces.com/api/v3").responses_url(),
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        );
    }

    #[test]
    fn system_goes_to_instructions_not_messages() {
        let body = provider("https://api.openai.com").build_body(&request(vec![]), false);
        assert_eq!(body["instructions"], "be nice");
        assert!(body.get("messages").is_none());
        assert_eq!(body["max_output_tokens"], 1024);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["store"], false);
    }

    #[test]
    fn deepseek_plaintext_reasoning_round_trips_through_input_items() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        assert_eq!(provider.reasoning, ReasoningMode::PlaintextReplay);

        // 序列化端：Thinking 块必须以明文 reasoning_text 内容块回传。
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "先想一下再回答".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "回答".into(),
                },
            ],
        };
        let items = message_to_items(&message, provider.reasoning);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(items[0]["content"][0]["text"], "先想一下再回答");
        assert_eq!(items[1]["type"], "message");

        // 解析端：DeepSeek 返回的 reasoning item 内容能被提取回 Thinking 块。
        let response = json!({
            "output": [
                { "type": "reasoning", "content": [ { "type": "reasoning_text", "text": "先想一下再回答" } ] },
                { "type": "message", "role": "assistant", "content": [ { "type": "output_text", "text": "回答" } ] }
            ]
        });
        let parsed =
            parse_responses_response_with_cache(&response, ReasoningMode::PlaintextReplay, true)
                .unwrap();
        assert_eq!(parsed.content.len(), 2);
        assert!(matches!(
            &parsed.content[0],
            ContentBlock::Thinking { thinking, signature: None } if thinking == "先想一下再回答"
        ));
    }

    #[test]
    fn deepseek_empty_reasoning_survives_serialization_parse_and_sanitization() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );

        // 序列化端：工具调用轮次的空 reasoning 必须以空 reasoning_text 内容块回传。
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "/a"}),
                },
            ],
        };
        let items = message_to_items(&message, provider.reasoning);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(items[0]["content"][0]["text"], "");
        assert_eq!(items[1]["type"], "function_call");

        // sanitize 不能把空 reasoning 当作“无配对产物”丢掉。
        let sanitized = sanitize_input_items(items);
        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0]["type"], "reasoning");

        // 解析端：空 reasoning_text 也保留为 Thinking 块，供下一轮原样回传。
        let response = json!({
            "output": [
                { "type": "reasoning", "content": [ { "type": "reasoning_text", "text": "" } ] },
                { "type": "function_call", "call_id": "call_1", "name": "read_file",
                  "arguments": "{\"path\":\"/a\"}" }
            ]
        });
        let parsed =
            parse_responses_response_with_cache(&response, ReasoningMode::PlaintextReplay, true)
                .unwrap();
        assert_eq!(parsed.content.len(), 2);
        assert!(matches!(
            &parsed.content[0],
            ContentBlock::Thinking {
                thinking,
                signature: None,
            } if thinking.is_empty()
        ));
        assert!(matches!(&parsed.content[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn deepseek_injects_empty_reasoning_for_legacy_tool_turns_missing_it() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let mut req = request(vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "/a"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "ok".into(),
                    is_error: false,
                }],
            },
        ]);
        req.model = "deepseek-v4-pro".into();
        req.inference = agent_contract::InferenceOptions {
            thinking: Some("enabled".into()),
            ..Default::default()
        };

        let body = provider.build_body(&req, false);
        let input = body["input"].as_array().unwrap();
        // 缺失 reasoning 的旧历史工具轮在发送前补一个空 reasoning item。
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(input[0]["content"][0]["text"], "");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
    }

    #[test]
    fn deepseek_does_not_inject_reasoning_when_thinking_is_disabled() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let mut req = request(vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                input: json!({"path": "/a"}),
            }],
        }]);
        req.model = "deepseek-v4-pro".into();
        req.inference = agent_contract::InferenceOptions {
            thinking: Some("disabled".into()),
            ..Default::default()
        };

        let body = provider.build_body(&req, false);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call");
    }

    #[test]
    fn inference_options_use_responses_reasoning_and_text_objects() {
        let mut req = request(vec![]);
        req.inference = agent_contract::InferenceOptions {
            thinking: None,
            reasoning_effort: Some("high".into()),
            verbosity: Some("low".into()),
        };

        let body = provider("https://api.openai.com").build_body(&req, false);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["text"]["verbosity"], "low");
    }

    #[test]
    fn deepseek_responses_maps_thinking_modes_to_native_reasoning_effort() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let mut request = request(vec![]);
        request.model = "deepseek-v4-pro".into();

        for thinking in ["adaptive", "enabled"] {
            request.inference = agent_contract::InferenceOptions {
                thinking: Some(thinking.into()),
                ..Default::default()
            };
            let body = provider.build_body(&request, false);
            assert_eq!(body["reasoning"]["effort"], "high");
            assert!(!body.to_string().contains("adaptive"));
        }

        request.inference = agent_contract::InferenceOptions {
            thinking: Some("disabled".into()),
            reasoning_effort: Some("max".into()),
            ..Default::default()
        };
        let body = provider.build_body(&request, false);
        assert_eq!(body["reasoning"]["effort"], "none");
        assert_eq!(
            body["reasoning"].as_object().map(serde_json::Map::len),
            Some(1)
        );
    }

    #[test]
    fn deepseek_responses_preserves_explicit_high_and_max_effort() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let mut request = request(vec![]);
        request.model = "deepseek-v4-pro".into();
        request.inference.thinking = Some("enabled".into());

        for effort in ["high", "max"] {
            request.inference.reasoning_effort = Some(effort.into());
            let body = provider.build_body(&request, false);
            assert_eq!(body["reasoning"]["effort"], effort);
        }
    }

    #[test]
    fn non_deepseek_responses_does_not_reinterpret_adaptive_marker() {
        let mut request = request(vec![]);
        request.inference.thinking = Some("adaptive".into());

        let body = provider("https://api.openai.com").build_body(&request, false);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn tools_are_flat_not_nested_under_function() {
        let mut req = request(vec![]);
        req.tools = vec![ToolSpec {
            name: "read_file".into(),
            description: "read a file".into(),
            input_schema: json!({"type": "object"}),
            source: agent_contract::ToolSource::Builtin,
            requires_confirmation: false,
        }];
        let body = provider("https://api.openai.com").build_body(&req, false);
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "read_file");
        assert!(tool.get("function").is_none());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn hosted_web_search_uses_the_responses_server_tool_schema() {
        let mut req = request(vec![]);
        req.hosted_tools = vec![agent_contract::HostedToolSpec::web_search()];

        let body = provider("https://api.deepseek.com").build_body(&req, false);

        assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert!(body["tools"][0].get("name").is_none());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn provider_specific_responses_web_tools_use_their_documented_types() {
        let mut dashscope = request(vec![]);
        dashscope.hosted_tools = vec![
            HostedToolSpec::web_search_with_format(HostedToolFormat::DashScope),
            HostedToolSpec::web_fetch_with_format(HostedToolFormat::DashScope),
        ];
        let dashscope_body = provider("https://dashscope.aliyuncs.com/compatible-mode/v1")
            .build_body(&dashscope, false);
        assert_eq!(dashscope_body["tools"][0]["type"], "web_search");
        assert_eq!(dashscope_body["tools"][1]["type"], "web_extractor");

        let mut openrouter = request(vec![]);
        openrouter.hosted_tools = vec![
            HostedToolSpec::web_search_with_format(HostedToolFormat::OpenRouter),
            HostedToolSpec::web_fetch_with_format(HostedToolFormat::OpenRouter),
        ];
        let openrouter_body =
            provider("https://openrouter.ai/api/v1").build_body(&openrouter, false);
        assert_eq!(openrouter_body["tools"][0]["type"], "openrouter:web_search");
        assert_eq!(openrouter_body["tools"][0]["parameters"]["max_uses"], 5);
        assert_eq!(openrouter_body["tools"][1]["type"], "openrouter:web_fetch");
        assert_eq!(openrouter_body["tools"][1]["parameters"]["max_uses"], 5);
    }

    #[test]
    fn standard_responses_does_not_invent_a_web_fetch_type() {
        let mut req = request(vec![]);
        req.hosted_tools = vec![HostedToolSpec::web_fetch()];

        let body = provider("https://api.openai.com/v1").build_body(&req, false);

        assert!(body.get("tools").is_none());
    }

    #[test]
    fn include_only_requested_in_encrypted_replay_mode() {
        let plain = provider("https://api.openai.com").build_body(&request(vec![]), false);
        assert!(plain.get("include").is_none());

        let replay = provider("https://api.openai.com")
            .with_reasoning(ReasoningMode::EncryptedReplay)
            .build_body(&request(vec![]), false);
        assert_eq!(replay["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn tool_use_becomes_function_call_with_call_id() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_abc".into(),
                name: "read_file".into(),
                input: json!({"path": "/a"}),
            }],
        };
        let items = message_to_items(&msg, ReasoningMode::Drop);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call_abc");
        // arguments 必须是字符串化 JSON
        assert!(items[0]["arguments"].is_string());
    }

    #[test]
    fn assistant_web_search_call_replays_as_a_responses_input_item() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Custom {
                type_name: "web_search_call".into(),
                data: json!({
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "Milvus hybrid search"}
                }),
            }],
        };

        let items = message_to_items(&msg, ReasoningMode::Drop);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "web_search_call");
        assert_eq!(items[0]["id"], "ws_1");
        assert_eq!(items[0]["action"]["query"], "Milvus hybrid search");
    }

    #[test]
    fn only_completed_bounded_hosted_web_items_are_replayed() {
        for data in [
            json!({"id": "ws_failed", "status": "failed"}),
            json!({"id": "ws_pending", "status": "in_progress"}),
            json!({"status": "completed"}),
        ] {
            let message = Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Custom {
                    type_name: "web_search_call".into(),
                    data,
                }],
            };
            assert!(message_to_items(&message, ReasoningMode::Drop).is_empty());
        }

        let oversized = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Custom {
                type_name: "web_search_call".into(),
                data: json!({
                    "id": "ws_oversized",
                    "status": "completed",
                    "action": {"query": "x".repeat(512 * 1024)}
                }),
            }],
        };
        assert!(message_to_items(&oversized, ReasoningMode::Drop).is_empty());
    }

    #[test]
    fn tool_result_becomes_function_call_output_not_message() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_abc".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_def".into(),
                    content: "also ok".into(),
                    is_error: false,
                },
            ],
        };
        let items = message_to_items(&msg, ReasoningMode::Drop);
        // 一条内部消息里的多个 ToolResult 必须全部展开
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call_def");
    }

    #[test]
    fn user_image_is_emitted_as_input_image() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "look".into(),
                },
                ContentBlock::Image {
                    source: agent_contract::ImageSource {
                        kind: "base64".into(),
                        media_type: "image/webp".into(),
                        data: "UklGRg==".into(),
                    },
                },
            ],
        };
        let items = message_to_items(&message, ReasoningMode::Drop);
        assert_eq!(items[0]["content"][0]["type"], "input_text");
        assert_eq!(items[0]["content"][1]["type"], "input_image");
        assert_eq!(
            items[0]["content"][1]["image_url"],
            "data:image/webp;base64,UklGRg=="
        );
    }

    #[test]
    fn orphan_function_call_output_is_dropped() {
        let items = vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
            // 上一轮被中止，只剩结果没有调用
            json!({"type": "function_call_output", "call_id": "call_ghost", "output": "?"}),
        ];
        let sanitized = sanitize_input_items(items);
        assert_eq!(sanitized.len(), 2);
        assert!(sanitized.iter().all(|item| item["call_id"] != "call_ghost"));
    }

    #[test]
    fn custom_deepseek_gateway_preserves_paired_long_tool_evidence_and_drops_only_orphans() {
        let provider = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://gateway.example/deepseek/v1".into(),
        );
        let evidence = long_tool_evidence();
        let assistant = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_read".into(),
                name: "read_file".into(),
                input: json!({"path": "src/核心.rs"}),
            }],
        };
        let results = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_read".into(),
                    content: evidence.clone(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_orphan".into(),
                    content: "must not reach the provider".into(),
                    is_error: false,
                },
            ],
        };
        let mut completion = request(vec![assistant, results]);
        completion.model = "deepseek-v4-pro".into();

        let body = provider.build_body(&completion, false);
        let input = body["input"].as_array().expect("responses input");

        // thinking 模式下，缺 reasoning 的工具轮会在发送前补空 reasoning item。
        assert_eq!(
            input.len(),
            3,
            "empty reasoning + function call + its paired output"
        );
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["content"][0]["text"], "");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_read");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_read");
        assert_eq!(input[2]["output"].as_str(), Some(evidence.as_str()));
        assert!(input[2]["output"]
            .as_str()
            .is_some_and(|output| output.ends_with("__TOOL_EVIDENCE_TAIL_RESPONSES__")));
        assert!(!body.to_string().contains("call_orphan"));
        assert_eq!(provider.name(), "deepseek_responses");
    }

    #[test]
    fn orphan_reasoning_is_dropped_but_paired_reasoning_survives() {
        let items = vec![
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}),
            json!({"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"}),
            // 末尾落单的 reasoning：留着必 400
            json!({"type": "reasoning", "id": "rs_2", "encrypted_content": "y"}),
        ];
        let sanitized = sanitize_input_items(items);
        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0]["id"], "rs_1");
        assert_eq!(sanitized[1]["type"], "function_call");
    }

    #[test]
    fn reasoning_followed_only_by_user_message_is_dropped() {
        let items = vec![
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}),
            json!({"type": "message", "role": "user", "content": []}),
        ];
        assert_eq!(sanitize_input_items(items).len(), 1);
    }

    #[test]
    fn reasoning_signature_roundtrip() {
        let sig = encode_reasoning_signature("rs_1", "gAAAAAB+base64==");
        assert_eq!(
            decode_reasoning_signature(&sig),
            Some(("rs_1".into(), "gAAAAAB+base64==".into()))
        );
        // Anthropic 的 thinking signature 不该被误读成 Responses 的
        assert_eq!(decode_reasoning_signature("ErUBCkYIBRgCKk"), None);
    }

    #[test]
    fn reasoning_block_replays_before_its_product() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "think".into(),
                    signature: Some(encode_reasoning_signature("rs_1", "enc")),
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "a".into(),
                    input: json!({}),
                },
            ],
        };
        let items = sanitize_input_items(message_to_items(&msg, ReasoningMode::EncryptedReplay));
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["encrypted_content"], "enc");
        assert_eq!(items[1]["type"], "function_call");
    }

    #[test]
    fn reasoning_block_omitted_in_drop_mode() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                thinking: "think".into(),
                signature: Some(encode_reasoning_signature("rs_1", "enc")),
            }],
        };
        assert!(message_to_items(&msg, ReasoningMode::Drop).is_empty());
    }

    #[test]
    fn parses_heterogeneous_output_array() {
        let value = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "read_file", "arguments": "{\"path\":\"/a\"}"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "done"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3,
                      "input_tokens_details": {"cached_tokens": 4}}
        });
        let parsed = parse_responses_response(&value, ReasoningMode::EncryptedReplay).unwrap();
        assert_eq!(parsed.content.len(), 3);
        assert!(matches!(parsed.content[0], ContentBlock::Thinking { .. }));
        // 工具调用要取 call_id 而不是 item 的 id
        assert_eq!(parsed.content[1].tool_id(), Some("call_1"));
        assert_eq!(parsed.text(), "done");
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        assert_eq!(parsed.usage.input_tokens, 10);
        assert_eq!(parsed.usage.cache_read_tokens, Some(4));
        assert_eq!(parsed.usage.cache_write_tokens, None);
    }

    #[test]
    fn deepseek_responses_derives_cache_miss_tokens() {
        let value = json!({
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}]
            }],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 3,
                "input_tokens_details": {"cached_tokens": 80}
            }
        });
        let parsed =
            parse_responses_response_with_cache(&value, ReasoningMode::Drop, true).unwrap();
        assert_eq!(parsed.usage.cache_read_tokens, Some(80));
        assert_eq!(parsed.usage.cache_write_tokens, Some(20));

        let explicit = json!({
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 0,
                "prompt_cache_hit_tokens": 70,
                "prompt_cache_miss_tokens": 30,
                "input_tokens_details": {"cached_tokens": 999}
            }
        });
        let parsed =
            parse_responses_response_with_cache(&explicit, ReasoningMode::Drop, true).unwrap();
        assert_eq!(parsed.usage.cache_read_tokens, Some(70));
        assert_eq!(parsed.usage.cache_write_tokens, Some(30));
    }

    #[test]
    fn incomplete_due_to_length_maps_to_max_tokens() {
        let value = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": []
        });
        let parsed = parse_responses_response(&value, ReasoningMode::Drop).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn missing_output_is_an_error_not_a_silent_empty_turn() {
        assert!(
            parse_responses_response(&json!({"status": "completed"}), ReasoningMode::Drop).is_err()
        );
    }

    // ── 流式 ──────────────────────────────────────────────────

    fn run_stream(frames: &[&str]) -> Vec<StreamEvent> {
        let mut state = StreamState::default();
        frames
            .iter()
            .flat_map(|frame| parse_one_responses_event(frame, &mut state))
            .collect()
    }

    #[test]
    fn typed_events_map_to_stream_events() {
        let events = run_stream(&[
            r#"{"type":"response.created","sequence_number":0}"#,
            r#"{"type":"response.output_text.delta","sequence_number":1,"delta":"He"}"#,
            r#"{"type":"response.output_text.delta","sequence_number":2,"delta":"llo"}"#,
            r#"{"type":"response.completed","sequence_number":3,
                "response":{"status":"completed","output":[],
                            "usage":{"input_tokens":5,"output_tokens":2}}}"#,
        ]);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hello");
        assert!(matches!(events.last(), Some(StreamEvent::Stop { .. })));
    }

    #[test]
    fn reasoning_summary_stream_is_visible_once_and_separate_from_answer() {
        let events = run_stream(&[
            r#"{"type":"response.reasoning_summary_text.delta","sequence_number":1,"item_id":"rs_1","delta":"check "}"#,
            r#"{"type":"response.reasoning_summary_text.delta","sequence_number":2,"item_id":"rs_1","delta":"facts"}"#,
            r#"{"type":"response.output_item.done","sequence_number":3,
                "item":{"type":"reasoning","id":"rs_1","summary":[{"text":"check facts"}]}}"#,
            r#"{"type":"response.output_text.delta","sequence_number":4,"delta":"done"}"#,
        ]);
        let reasoning = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ReasoningDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "check facts");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::ReasoningDelta { .. }))
                .count(),
            2,
            "the completed item must not replay an already streamed summary"
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "done")));
    }

    #[test]
    fn completed_response_backfills_reasoning_when_deltas_are_absent() {
        let events = run_stream(&[
            r#"{"type":"response.completed","sequence_number":1,"response":{
                "status":"completed","output":[
                    {"type":"reasoning","id":"rs_1","summary":[{"text":"summary"}]},
                    {"type":"message","content":[]}
                ],"usage":{"input_tokens":1,"output_tokens":1}}}"#,
        ]);
        assert!(events.iter().any(
            |event| matches!(event, StreamEvent::ReasoningDelta { text } if text == "summary")
        ));
    }

    #[test]
    fn ark_summary_reasoning_roundtrips_without_dropping() {
        let response = serde_json::json!({
            "status": "completed",
            "output": [
                {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"先检查再动手"}]},
                {"type":"message","content":[{"type":"output_text","text":"done"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let parsed =
            parse_responses_response_with_cache(&response, ReasoningMode::SummaryReplay, false)
                .unwrap();
        assert!(matches!(
            parsed.content.as_slice(),
            [ContentBlock::Thinking { thinking, signature: None }, ContentBlock::Text { .. }]
                if thinking == "先检查再动手"
        ));

        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "先检查再动手".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "done".into(),
                },
            ],
        };
        let items = message_to_items(&msg, ReasoningMode::SummaryReplay);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["summary"][0]["type"], "summary_text");
        assert_eq!(items[0]["summary"][0]["text"], "先检查再动手");
    }

    #[test]
    fn ark_effort_mapping_filters_rejected_vocab() {
        let provider = ResponsesProvider::new_ark(
            "k".into(),
            "glm-5.3".into(),
            "https://ark.cn-beijing.volces.com/api/plan/v3".into(),
            "ark_agent",
        );
        let cases: Vec<(Option<&str>, Option<&str>, Option<&str>)> = vec![
            (Some("enabled"), None, Some("high")),
            (None, Some("low"), Some("low")),
            (None, Some("max"), Some("max")),
            (None, Some("none"), None),
            (None, Some("minimal"), None),
            (Some("disabled"), None, None),
            (Some("adaptive"), None, None),
        ];
        for (thinking, effort, expected) in cases {
            let request = CompletionRequest {
                model: "glm-5.3".into(),
                system: None,
                messages: vec![Message::user_text("ping")],
                tools: vec![],
                hosted_tools: vec![],
                max_tokens: 16,
                temperature: None,
                enable_caching: false,
                inference: agent_contract::InferenceOptions {
                    thinking: thinking.map(str::to_string),
                    reasoning_effort: effort.map(str::to_string),
                    verbosity: None,
                },
            };
            let body = provider.build_body(&request, false);
            let actual = body
                .get("reasoning")
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str);
            assert_eq!(actual, expected, "thinking={thinking:?} effort={effort:?}");
        }
    }

    fn run_stream_with_reasoning(reasoning: ReasoningMode, frames: &[&str]) -> Vec<StreamEvent> {
        let mut state = StreamState {
            reasoning,
            ..StreamState::default()
        };
        frames
            .iter()
            .flat_map(|frame| parse_one_responses_event(frame, &mut state))
            .collect()
    }

    #[test]
    fn deepseek_plaintext_reasoning_is_backfilled_from_the_completed_item() {
        // DeepSeek Responses 的明文 reasoning 放在 content[].reasoning_text，而不是
        // OpenAI 的 summary[]。流式路径若只收到 output_item.done（无 delta），
        // 也必须把明文 reasoning 提出来，否则下一轮回传缺失会 400。
        let events = run_stream_with_reasoning(
            ReasoningMode::PlaintextReplay,
            &[r#"{"type":"response.output_item.done","sequence_number":1,
                    "item":{"type":"reasoning","id":"rs_1",
                            "content":[{"type":"reasoning_text","text":"先想一下再回答"}]}}"#],
        );
        assert!(events.iter().any(
            |event| matches!(event, StreamEvent::ReasoningDelta { text } if text == "先想一下再回答")
        ));
    }

    #[test]
    fn deepseek_empty_reasoning_item_still_emits_a_delta_for_replay() {
        // 工具调用轮次可能只收到一个空 reasoning item：不能因为文本为空就当作
        // “没有 reasoning”，否则下一轮缺少空 reasoning_text 会 400。
        let events = run_stream_with_reasoning(
            ReasoningMode::PlaintextReplay,
            &[r#"{"type":"response.output_item.done","sequence_number":1,
                    "item":{"type":"reasoning","id":"rs_1",
                            "content":[{"type":"reasoning_text","text":""}]}}"#],
        );
        assert!(matches!(
            &events[..],
            [StreamEvent::ReasoningDelta { text }] if text.is_empty()
        ));
    }

    #[test]
    fn deepseek_plaintext_reasoning_is_backfilled_from_final_output() {
        let events = run_stream_with_reasoning(
            ReasoningMode::PlaintextReplay,
            &[
                r#"{"type":"response.completed","sequence_number":1,"response":{
                    "status":"completed","output":[
                        {"type":"reasoning","id":"rs_1",
                         "content":[{"type":"reasoning_text","text":"最终输出里的思维链"}]},
                        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"回答"}]}
                    ],"usage":{"input_tokens":1,"output_tokens":1}}}"#,
            ],
        );
        assert!(events.iter().any(
            |event| matches!(event, StreamEvent::ReasoningDelta { text } if text == "最终输出里的思维链")
        ));
    }

    #[test]
    fn tool_call_stream_uses_call_id_throughout() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file"}}"#,
            r#"{"type":"response.function_call_arguments.delta","sequence_number":2,
                "item_id":"fc_1","delta":"{\"path\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","sequence_number":3,
                "item_id":"fc_1","delta":"\"/a\"}"}"#,
            r#"{"type":"response.output_item.done","sequence_number":4,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1",
                        "name":"read_file","arguments":"{\"path\":\"/a\"}"}}"#,
        ]);
        assert!(matches!(&events[0], StreamEvent::ToolUseStart { id, .. } if id == "call_1"));
        assert!(matches!(&events[1], StreamEvent::ToolUseDelta { id, .. } if id == "call_1"));
        match events.last().unwrap() {
            StreamEvent::ToolUseComplete { id, input } => {
                assert_eq!(id, "call_1");
                // 以 .done 的完整 arguments 为准，而不是 delta 拼接
                assert_eq!(input["path"], "/a");
            }
            other => panic!("expected ToolUseComplete, got {other:?}"),
        }
    }

    #[test]
    fn web_search_stream_emits_hosted_events_and_never_a_local_tool_call() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"web_search_call","id":"ws_1","status":"in_progress",
                        "action":{"type":"search","query":"Milvus hybrid search"}}}"#,
            r#"{"type":"response.output_item.done","sequence_number":2,
                "item":{"type":"web_search_call","id":"ws_1","status":"completed",
                        "action":{"type":"search","query":"Milvus hybrid search"}}}"#,
            r#"{"type":"response.completed","sequence_number":3,
                "response":{"status":"completed",
                 "output":[{"type":"web_search_call","id":"ws_1","status":"completed",
                            "action":{"type":"search","query":"Milvus hybrid search"}}],
                 "usage":{"input_tokens":2,"output_tokens":1}}}"#,
        ]);

        assert!(matches!(
            &events[0],
            StreamEvent::HostedToolUse { id, name, input, .. }
                if id == "ws_1"
                    && name == "web_search"
                    && input["query"] == "Milvus hybrid search"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::HostedToolResult { id, name, output, is_error: false, .. }
                if id == "ws_1"
                    && name == "web_search"
                    && output["status"] == "completed"
        ));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolUseStart { .. }
                | StreamEvent::ToolUseDelta { .. }
                | StreamEvent::ToolUseComplete { .. }
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::HostedToolUse { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::HostedToolResult { .. }))
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Stop {
                reason: StopReason::EndTurn
            })
        ));
    }

    #[test]
    fn web_extractor_stream_is_reported_as_hosted_web_fetch() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.added","sequence_number":1,
                "item":{"type":"web_extractor_call","id":"we_1","status":"in_progress",
                        "goal":"summarize","urls":["https://example.com"]}}"#,
            r#"{"type":"response.output_item.done","sequence_number":2,
                "item":{"type":"web_extractor_call","id":"we_1","status":"completed",
                        "goal":"summarize","urls":["https://example.com"],"output":"summary"}}"#,
        ]);

        assert!(matches!(
            &events[0],
            StreamEvent::HostedToolUse { id, name, input, .. }
                if id == "we_1" && name == "web_fetch"
                    && input["urls"][0] == "https://example.com"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::HostedToolResult { id, name, output, is_error: false, .. }
                if id == "we_1" && name == "web_fetch"
                    && output["urls"][0] == "https://example.com"
        ));
    }

    #[test]
    fn replayed_sequence_numbers_are_deduped() {
        let mut state = StreamState::default();
        let frame = r#"{"type":"response.output_text.delta","sequence_number":1,"delta":"x"}"#;
        assert_eq!(parse_one_responses_event(frame, &mut state).len(), 1);
        // 断线重连重放同一帧
        assert_eq!(parse_one_responses_event(frame, &mut state).len(), 0);
    }

    #[test]
    fn unknown_events_are_ignored_not_fatal() {
        let events = run_stream(&[
            r#"{"type":"response.some_future_event","sequence_number":1,"whatever":true}"#,
            r#"{"type":"response.output_text.delta","sequence_number":2,"delta":"ok"}"#,
        ]);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn final_frame_backfills_tool_calls_never_announced() {
        // 部分兼容实现不发 output_item.added/done，只在终帧给出完整 output
        let events = run_stream(&[r#"{"type":"response.completed","sequence_number":1,
                "response":{"status":"completed",
                 "output":[{"type":"function_call","id":"fc_1","call_id":"call_1",
                            "name":"a","arguments":"{}"}],
                 "usage":{"input_tokens":1,"output_tokens":1}}}"#]);
        assert!(matches!(&events[0], StreamEvent::ToolUseStart { id, .. } if id == "call_1"));
        assert!(matches!(&events[1], StreamEvent::ToolUseComplete { .. }));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Stop {
                reason: StopReason::ToolUse
            })
        ));
    }

    #[test]
    fn tool_call_is_not_completed_twice() {
        let events = run_stream(&[
            r#"{"type":"response.output_item.done","sequence_number":1,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_1",
                        "name":"a","arguments":"{}"}}"#,
            r#"{"type":"response.completed","sequence_number":2,
                "response":{"status":"completed",
                 "output":[{"type":"function_call","id":"fc_1","call_id":"call_1",
                            "name":"a","arguments":"{}"}],"usage":{}}}"#,
        ]);
        let completes = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ToolUseComplete { .. }))
            .count();
        assert_eq!(completes, 1);
    }

    #[test]
    fn failure_frame_stops_the_stream() {
        let events = run_stream(&[r#"{"type":"response.failed","sequence_number":1,
                "response":{"error":{"message":"upstream exploded"}}}"#]);
        assert!(matches!(
            &events[0],
            StreamEvent::Stop { reason: StopReason::Other(message) } if message == "upstream exploded"
        ));
    }

    #[test]
    fn failure_frame_preserves_safe_error_classification() {
        let events = run_stream(&[r#"{"type":"response.failed","sequence_number":1,
                "response":{"error":{"message":"unsupported input item","type":"invalid_request_error",
                                       "code":"invalid_value","param":"input[37]"}}}"#]);
        assert!(matches!(
            &events[0],
            StreamEvent::Stop { reason: StopReason::Other(message) }
                if message == "unsupported input item (type=invalid_request_error, code=invalid_value, param=input[37])"
        ));
    }

    #[test]
    fn error_frames_support_compatible_message_shapes() {
        let cases = [
            (
                r#"{"type":"error","error":{"message":"nested error"}}"#,
                "nested error",
            ),
            (
                r#"{"type":"error","error":{"message":"bad input","type":"invalid_request_error","code":"invalid_value","param":"tools[0]"}}"#,
                "bad input (type=invalid_request_error, code=invalid_value, param=tools[0])",
            ),
            (
                r#"{"type":"error","message":"top-level error"}"#,
                "top-level error",
            ),
            (
                r#"{"type":"error","message":"top-level bad input","error_type":"invalid_request_error","code":"invalid_value","param":"input[2]"}"#,
                "top-level bad input (type=invalid_request_error, code=invalid_value, param=input[2])",
            ),
            (r#"{"type":"error"}"#, "request failed"),
        ];

        for (frame, expected) in cases {
            let events = run_stream(&[frame]);
            assert_eq!(events.len(), 1);
            assert!(matches!(
                &events[0],
                StreamEvent::Stop { reason: StopReason::Other(message) } if message == expected
            ));
        }
    }

    #[test]
    fn terminal_frames_emit_stop_only_once() {
        let completed_then_errors = run_stream(&[
            r#"{"type":"response.completed","sequence_number":1,
                "response":{"status":"completed","output":[],"usage":{}}}"#,
            r#"{"type":"response.failed","sequence_number":2,
                "response":{"error":{"message":"late failure"}}}"#,
            r#"{"type":"error","sequence_number":3,"message":"late error"}"#,
        ]);
        assert_eq!(
            completed_then_errors
                .iter()
                .filter(|event| matches!(event, StreamEvent::Stop { .. }))
                .count(),
            1
        );

        let failure_then_errors = run_stream(&[
            r#"{"type":"response.failed","sequence_number":1,
                "response":{"error":{"message":"first failure"}}}"#,
            r#"{"type":"error","sequence_number":2,"message":"late error"}"#,
            r#"{"type":"response.failed","sequence_number":3,
                "response":{"error":{"message":"later failure"}}}"#,
        ]);
        let stops: Vec<_> = failure_then_errors
            .iter()
            .filter(|event| matches!(event, StreamEvent::Stop { .. }))
            .collect();
        assert_eq!(stops.len(), 1);
        assert!(matches!(
            stops[0],
            StreamEvent::Stop { reason: StopReason::Other(message) } if message == "first failure"
        ));
    }

    #[test]
    fn capabilities_and_name() {
        let provider = provider("https://api.openai.com");
        assert_eq!(provider.name(), "openai_responses");
        assert!(provider.capabilities().supports_tool_use);
        assert_eq!(
            provider
                .with_max_context_tokens(400_000)
                .capabilities()
                .max_context_tokens,
            400_000
        );

        let deepseek = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-flash".into(),
            "https://api.deepseek.com".into(),
        );
        assert_eq!(deepseek.name(), "deepseek_responses");
        assert!(deepseek.capabilities().supports_prompt_caching);
        assert_eq!(deepseek.capabilities().max_context_tokens, 1_000_000);
        assert_eq!(deepseek.capabilities().max_output_tokens, 393_216);
        assert!(!deepseek.capabilities().supports_vision);

        let chat = ResponsesProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        assert_eq!(chat.capabilities().max_output_tokens, 393_216);
    }

    #[test]
    fn pdf_attachment_becomes_an_input_file() {
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::File {
                source: agent_contract::FileSource {
                    kind: "base64".into(),
                    name: "spec.pdf".into(),
                    media_type: "application/pdf".into(),
                    text: None,
                    data: Some("JVBERi0=".into()),
                },
            }],
        };
        let items = message_to_items(&message, ReasoningMode::Drop);
        assert_eq!(items[0]["content"][0]["type"], "input_file");
        assert_eq!(items[0]["content"][0]["filename"], "spec.pdf");
    }
}
