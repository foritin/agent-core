//! Anthropic Claude Provider。
//!
//! 参见 `01-llm-provider.html §4.1 §7`。实现 Messages API 转换、流式 SSE 解析
//! 与 prompt caching。错误消息绝不包含 api_key（V-PROV-02）。

use agent_contract::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, HostedToolFormat,
    HostedToolSpec, LlmProvider, Message, Role, StopReason, StreamEvent, Usage,
};
use agent_error::{Error, Result};
use serde_json::{json, Value};
use std::time::Duration;

use crate::dialect::{dialect_thinking_value, EffortWire, WireDialect};

/// DeepSeek V4 窗口语义；`deepseek-chat` 已实测被服务端别名到 v4-flash。
fn is_deepseek_v4_alias(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("deepseek-v4-") || model == "deepseek-chat"
}

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const MAX_RETRIES: u32 = 10;
const MAX_BACKOFF: Duration = Duration::from_secs(15);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(not(test))]
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
#[cfg(test)]
const RETRY_BASE_DELAY: Duration = Duration::from_millis(1);

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
    emit_explicit_cache_control: bool,
    deepseek_automatic_cache: bool,
    max_context_tokens: u32,
    max_output_tokens: u32,
    supports_vision: bool,
    provider_name: &'static str,
    /// 厂商/模型族方言；None 保持与改造前一致的通用 Anthropic 行为。
    dialect: Option<WireDialect>,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        if api_key.is_empty() {
            return Err(Error::AuthFailed("anthropic api_key is empty".into()));
        }
        Ok(Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .user_agent(concat!("R-Code/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("anthropic provider client"),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
            emit_explicit_cache_control: true,
            deepseek_automatic_cache: false,
            max_context_tokens: 200_000,
            max_output_tokens: 0,
            supports_vision: true,
            provider_name: "anthropic",
            dialect: None,
        })
    }

    /// 应用实测冻结的厂商方言；UA 覆盖会重建 client。
    pub(crate) fn with_dialect(mut self, dialect: WireDialect) -> Self {
        if let Some(user_agent) = dialect.user_agent {
            self.client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .user_agent(user_agent)
                .build()
                .expect("anthropic provider client with dialect ua");
        }
        self.emit_explicit_cache_control = dialect.emit_cache_control;
        self.dialect = Some(dialect);
        self
    }

    /// DeepSeek 的 Anthropic 兼容口使用服务端自动前缀缓存；`cache_control` 仅被
    /// 忽略，不是开启缓存的条件。因此保留 DeepSeek 身份并避免注入无效字段。
    pub fn new_deepseek(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let is_v4 = is_deepseek_v4_alias(&model);
        let lowercase_model = model.trim().to_ascii_lowercase();
        let mut provider = Self::new(api_key, model, base_url)?;
        provider.emit_explicit_cache_control = false;
        provider.deepseek_automatic_cache = true;
        provider.max_context_tokens = 1_000_000;
        // DeepSeek V4 的单次输出上限是 393_216（API 报错口径）；非 V4 未声明。
        provider.max_output_tokens = if is_v4 { 393_216 } else { 0 };
        // 目录级真值（deepseek.rs）：只有 vision 实验模型支持图片输入。
        provider.supports_vision =
            crate::deepseek::deepseek_model_supports_vision(&lowercase_model);
        provider.provider_name = "deepseek_anthropic";
        Ok(provider)
    }

    /// Kimi For Coding 使用 Anthropic Messages 线路，但上下文窗口是 256 Ki tokens，
    /// 不能沿用 Anthropic 官方的 200K 默认值，否则运行时会过早压缩会话。
    pub fn new_kimi_coding(
        api_key: String,
        model: String,
        base_url: Option<String>,
    ) -> Result<Self> {
        let mut provider = Self::new(api_key, model, base_url)?;
        provider.max_context_tokens = 262_144;
        provider.provider_name = "kimi_coding";
        Ok(provider)
    }

    fn messages_url(&self) -> String {
        let base_url = self.base_url.trim_end_matches('/');
        let api_root = if base_url.ends_with("/v1") {
            base_url.to_string()
        } else {
            format!("{base_url}/v1")
        };
        format!("{api_root}/messages")
    }

    /// 将内部请求转为 Anthropic Messages API 请求体。
    fn build_request_body(&self, request: &CompletionRequest) -> Value {
        let messages: Vec<Value> = request.messages.iter().map(message_to_anthropic).collect();

        let mut body = json!({
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        });

        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }
        if let Some(dialect) = &self.dialect {
            // 实测冻结的 Ark/Kimi 方言：thinking 只发词表内值，temperature 按规则
            // 省略，effort 只按 EffortWire::Nested 嵌进 thinking 对象。
            let thinking = dialect_thinking_value(dialect, request.inference.thinking.as_deref());
            let omit_temperature = dialect.omit_temperature
                || (dialect.omit_temperature_when_thinking && matches!(thinking, Some("enabled")));
            if !omit_temperature {
                if let Some(temp) = request.temperature {
                    body["temperature"] = json!(temp);
                }
            }
            if let Some(thinking) = thinking {
                if dialect.effort_wire == EffortWire::Nested {
                    let effort = request
                        .inference
                        .reasoning_effort
                        .as_deref()
                        .filter(|value| {
                            dialect.effort_vocab.is_empty() || dialect.effort_vocab.contains(value)
                        });
                    if let Some(effort) = effort {
                        body["thinking"] = json!({ "type": thinking, "effort": effort });
                    } else {
                        body["thinking"] = json!({ "type": thinking });
                    }
                } else {
                    body["thinking"] = json!({ "type": thinking });
                }
            }
        } else {
            if let Some(temp) = request.temperature {
                body["temperature"] = json!(temp);
            }
            let is_deepseek_v4 =
                self.deepseek_automatic_cache && is_deepseek_v4_alias(&request.model);
            if let Some(thinking) = request.inference.thinking.as_deref() {
                // `adaptive` is an R-Code-side governor marker for DeepSeek.  DeepSeek's
                // Anthropic compatibility endpoint accepts enabled/disabled, whereas Anthropic
                // itself uses adaptive thinking.  Translate only for providers constructed with
                // DeepSeek identity so Claude and other compatible endpoints retain their
                // native semantics.
                let thinking = if is_deepseek_v4 && thinking == "adaptive" {
                    "enabled"
                } else {
                    thinking
                };
                body["thinking"] = json!({ "type": thinking });
            } else if request.inference.reasoning_effort.is_some()
                && request.model.to_ascii_lowercase().contains("claude")
            {
                // Claude 4.6+ / 5 的 effort 必须与 adaptive thinking 一起使用。
                body["thinking"] = json!({ "type": "adaptive" });
            }
            let effort = if is_deepseek_v4 {
                match request.inference.thinking.as_deref() {
                    // The off switch wins over a stale effort saved by an earlier UI selection.
                    Some("disabled") => None,
                    Some("enabled" | "adaptive") => request
                        .inference
                        .reasoning_effort
                        .as_deref()
                        .or(Some("high")),
                    _ => request.inference.reasoning_effort.as_deref(),
                }
            } else {
                request.inference.reasoning_effort.as_deref()
            };
            if let Some(effort) = effort {
                body["output_config"] = json!({ "effort": effort });
            }
        }
        let mut tools = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect::<Vec<_>>();
        tools.extend(
            request
                .hosted_tools
                .iter()
                .filter_map(hosted_tool_to_anthropic),
        );
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if request.enable_caching && self.emit_explicit_cache_control {
            apply_prompt_caching(&mut body);
        }
        body
    }

    /// 非流式完成。
    async fn do_complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let body = self.build_request_body(&request);
        let resp = self.send_with_retry(&body).await?;

        let v: Value = resp.json().await.map_err(|e| {
            Error::Provider(format!(
                "invalid response json: {}",
                sanitize_http_err(&e, &self.api_key)
            ))
        })?;
        parse_anthropic_response(&v, self.deepseek_automatic_cache)
    }

    /// 流式完成（SSE）。
    async fn do_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let mut body = self.build_request_body(&request);
        body["stream"] = json!(true);
        let resp = self.send_with_retry(&body).await?;

        // F-robust-04：与 openai 线路共享同一 SSE 空闲 watchdog。半开连接
        //（网关停发 body）在 DEFAULT_STREAM_IDLE_TIMEOUT 内无新字节即终止，
        // 否则只靠 agent_loop 的 10 分钟兜底，单故障窗口过长。
        let byte_stream = crate::openai::watch_sse_idle(
            resp.bytes_stream(),
            crate::openai::DEFAULT_STREAM_IDLE_TIMEOUT,
        );
        let stream = parse_sse_stream(
            byte_stream,
            self.deepseek_automatic_cache,
            self.api_key.clone(),
        );
        Ok(Box::pin(stream))
    }

    /// 只在连接与响应头阶段重试。请求 JSON 先序列化一次并在每次尝试中复用，
    /// 保证重放字节一致；SSE body 开始后由 agent 层根据错误事件决定是否整轮重放。
    async fn send_with_retry(&self, body: &Value) -> Result<reqwest::Response> {
        let body_bytes = serde_json::to_vec(body)
            .map_err(|error| Error::Provider(format!("invalid request json: {error}")))?;
        let url = self.messages_url();
        let mut attempt = 0_u32;
        let mut retry_after = None;

        loop {
            attempt += 1;
            if attempt > 1 {
                let delay = backoff_delay(attempt - 1, retry_after);
                tracing::debug!(
                    provider = self.provider_name,
                    attempt,
                    ?delay,
                    "provider request failed before response body, retrying"
                );
                thread_sleep(delay).await;
            }
            let (error, next_retry_after) = match self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .body(body_bytes.clone())
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response);
                    }
                    let retry_after = parse_retry_after(response.headers());
                    let text = read_error_body(response).await;
                    let error = map_api_error_with_key(status.as_u16(), &text, &self.api_key);
                    if !retryable_status(status.as_u16()) {
                        return Err(error);
                    }
                    (error, retry_after)
                }
                Err(error) => (
                    Error::Provider(sanitize_http_err(&error, &self.api_key)),
                    None,
                ),
            };

            if attempt > MAX_RETRIES {
                return Err(error);
            }
            retry_after = next_retry_after;
        }
    }
}

/// Anthropic 与 DeepSeek Anthropic 兼容口都把服务端工具放在 `tools` 数组，
/// 但它们带版本化 `type`，且不会产生需要客户端执行的普通 `tool_use`。
fn hosted_tool_to_anthropic(tool: &HostedToolSpec) -> Option<Value> {
    match tool {
        HostedToolSpec::WebSearch {
            format: HostedToolFormat::Standard,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(anthropic_web_tool(
            "web_search_20250305",
            "web_search",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        HostedToolSpec::WebFetch {
            format: HostedToolFormat::Standard,
            max_uses,
            allowed_domains,
            blocked_domains,
        } => Some(anthropic_web_tool(
            "web_fetch_20250910",
            "web_fetch",
            *max_uses,
            allowed_domains,
            blocked_domains,
        )),
        // Provider-specific formats must never be sent to an Anthropic-compatible endpoint.
        _ => None,
    }
}

fn anthropic_web_tool(
    tool_type: &str,
    name: &str,
    max_uses: Option<u32>,
    allowed_domains: &[String],
    blocked_domains: &[String],
) -> Value {
    let mut value = json!({"type": tool_type, "name": name});
    if let Some(max_uses) = max_uses {
        value["max_uses"] = json!(max_uses);
    }
    if !allowed_domains.is_empty() {
        value["allowed_domains"] = json!(allowed_domains);
    }
    if !blocked_domains.is_empty() {
        value["blocked_domains"] = json!(blocked_domains);
    }
    value
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.do_complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        self.do_stream(request).await
    }

    fn capabilities(&self) -> Capabilities {
        if let Some(dialect) = &self.dialect {
            return Capabilities {
                supports_streaming: true,
                supports_tool_use: true,
                supports_vision: dialect.supports_vision,
                supports_prompt_caching: true,
                max_context_tokens: dialect.max_context_tokens,
                max_output_tokens: dialect.max_output_tokens,
            };
        }
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: self.supports_vision,
            supports_prompt_caching: true,
            max_context_tokens: self.max_context_tokens,
            max_output_tokens: self.max_output_tokens,
        }
    }

    fn echoes_reasoning(&self) -> bool {
        self.dialect
            .as_ref()
            .is_some_and(|dialect| dialect.echo_reasoning)
    }

    fn name(&self) -> &str {
        self.provider_name
    }
}

// ── 请求转换 ──────────────────────────────────────────────────

/// 将内部 Message 转为 Anthropic 消息 JSON。
pub fn message_to_anthropic(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = msg.content.iter().map(content_block_to_anthropic).collect();
    json!({ "role": role, "content": content })
}

fn content_block_to_anthropic(b: &ContentBlock) -> Value {
    match b {
        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut v = json!({ "type": "thinking", "thinking": thinking });
            if let Some(sig) = signature {
                v["signature"] = json!(sig);
            }
            v
        }
        ContentBlock::ToolUse { id, name, input } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            });
            if *is_error {
                v["is_error"] = json!(true);
            }
            v
        }
        ContentBlock::Image { source } => json!({ "type": "image", "source": source }),
        // 引用块必须在进入适配器前物化（complete/stream 入口已 fail closed 断言）。
        // 这里只作为直接调用 message_to_anthropic 的最后防线，输出可见错误标记，
        // 绝不伪造成可被模型误解的正文。
        ContentBlock::Attachment { source } => json!({
            "type": "text",
            "text": format!(
                "[UNRESOLVED ATTACHMENT {} — materialization skipped; this must not be dispatched]",
                source.attachment_id
            ),
        }),
        ContentBlock::File { source } if source.media_type.starts_with("image/") => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": source.media_type,
                "data": source.data.as_deref().unwrap_or_default(),
            }
        }),
        ContentBlock::File { source } if source.media_type == "application/pdf" => json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": source.media_type,
                "data": source.data.as_deref().unwrap_or_default(),
            },
            "title": source.name,
        }),
        ContentBlock::File { source } => json!({
            "type": "text",
            "text": format!(
                "\n\n--- Attached file: {} ({}) ---\n{}\n--- End attached file: {} ---",
                source.name,
                source.media_type,
                source.text.as_deref().unwrap_or_default(),
                source.name,
            )
        }),
        ContentBlock::Custom { type_name, data } => {
            // Provider-hosted blocks must be replayed byte-for-byte in mixed client/server
            // tool turns and after `pause_turn`. Other product extensions still degrade to
            // visible text because the public provider layer does not interpret them.
            if is_anthropic_hosted_content_type(type_name) {
                let mut block = data.as_object().cloned().unwrap_or_default();
                block.insert("type".to_string(), Value::String(type_name.clone()));
                Value::Object(block)
            } else {
                json!({
                    "type": "text",
                    "text": format!("[custom: {} {}]", type_name, data)
                })
            }
        }
    }
}

fn is_anthropic_hosted_content_type(kind: &str) -> bool {
    kind == "server_tool_use" || kind.ends_with("_tool_result")
}

fn hosted_content_block(block: &Value) -> Option<ContentBlock> {
    let mut data = block.as_object()?.clone();
    let type_name = data.remove("type")?.as_str()?.to_string();
    is_anthropic_hosted_content_type(&type_name).then_some(ContentBlock::Custom {
        type_name,
        data: Value::Object(data),
    })
}

/// 对 system + tools 添加 cache_control（prompt caching）。
fn apply_prompt_caching(body: &mut Value) {
    if let Some(system) = body.get_mut("system") {
        if let Some(s) = system.as_str() {
            *system = json!([{
                "type": "text",
                "text": s,
                "cache_control": { "type": "ephemeral" }
            }]);
        }
    }
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({ "type": "ephemeral" });
        }
    }
}

// ── 响应解析 ──────────────────────────────────────────────────

fn parse_anthropic_response(
    v: &Value,
    deepseek_automatic_cache: bool,
) -> Result<CompletionResponse> {
    let content_arr = v
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| Error::Provider("response missing content array".into()))?;

    let mut content = Vec::new();
    for block in content_arr {
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match kind {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                content.push(ContentBlock::Text { text });
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                content.push(ContentBlock::ToolUse { id, name, input });
            }
            "thinking" => {
                let thinking = block
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let signature = block
                    .get("signature")
                    .and_then(|t| t.as_str())
                    .map(String::from);
                content.push(ContentBlock::Thinking {
                    thinking,
                    signature,
                });
            }
            "server_tool_use" => {
                if let Some(block) = hosted_content_block(block) {
                    content.push(block);
                }
            }
            kind if kind.ends_with("_tool_result") => {
                if let Some(block) = hosted_content_block(block) {
                    content.push(block);
                }
            }
            _ => {}
        }
    }

    let stop_reason = v
        .get("stop_reason")
        .and_then(|t| t.as_str())
        .map(parse_stop_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = v
        .get("usage")
        .map(|usage| parse_anthropic_usage(usage, deepseek_automatic_cache))
        .unwrap_or_default();

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

fn parse_anthropic_usage(usage: &Value, deepseek_automatic_cache: bool) -> Usage {
    let uncached_input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let cache_read_tokens = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .map(|tokens| tokens as u32);
    let cache_creation_tokens = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .map(|tokens| tokens as u32);
    let has_cache_metrics = cache_read_tokens.is_some() || cache_creation_tokens.is_some();

    // Anthropic 的 input_tokens 不包含 cache read/create。DeepSeek 的 Anthropic
    // 兼容口也沿用该形状，但它使用自动缓存，cache_creation 恒为 0；对 DeepSeek
    // 专属 provider 将其归一成与 Chat/Responses 一致的「总输入 + hit/miss」。
    let (input_tokens, cache_write_tokens) = if deepseek_automatic_cache && has_cache_metrics {
        let created = cache_creation_tokens.unwrap_or(0);
        (
            uncached_input_tokens
                .saturating_add(cache_read_tokens.unwrap_or(0))
                .saturating_add(created),
            Some(uncached_input_tokens.saturating_add(created)),
        )
    } else {
        (uncached_input_tokens, cache_creation_tokens)
    };

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

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        other => StopReason::Other(other.to_string()),
    }
}

fn retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500..=599)
}

fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(after) = retry_after {
        return after.min(MAX_RETRY_AFTER);
    }
    RETRY_BASE_DELAY
        .saturating_mul(2_u32.saturating_pow(attempt.saturating_sub(1)))
        .min(MAX_BACKOFF)
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// agent-llm 的生产依赖不包含 Tokio；用一个可被 drop 的轻量 future 完成退避。
async fn thread_sleep(delay: Duration) {
    let (sender, receiver) = std::sync::mpsc::channel::<()>();
    let waker_slot: std::sync::Arc<std::sync::Mutex<Option<std::task::Waker>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let thread_waker = waker_slot.clone();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        if let Some(waker) = thread_waker.lock().unwrap().take() {
            waker.wake();
        }
        let _ = sender.send(());
    });
    let mut done = false;
    futures::future::poll_fn(move |context| {
        if done {
            return std::task::Poll::Ready(());
        }
        *waker_slot.lock().unwrap() = Some(context.waker().clone());
        match receiver.try_recv() {
            Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                done = true;
                std::task::Poll::Ready(())
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => std::task::Poll::Pending,
        }
    })
    .await
}

async fn read_error_body(response: reqwest::Response) -> String {
    let body = response.text();
    futures::pin_mut!(body);
    let timeout = thread_sleep(ERROR_BODY_READ_TIMEOUT);
    futures::pin_mut!(timeout);
    match futures::future::select(&mut body, &mut timeout).await {
        futures::future::Either::Left((result, _)) => result.unwrap_or_default(),
        futures::future::Either::Right(_) => String::new(),
    }
}

/// 将 HTTP 状态码 + 响应体映射为稳定错误类别，绝不包含 api_key。
fn map_api_error_with_key(status: u16, body: &str, api_key: &str) -> Error {
    match status {
        401 | 403 => Error::AuthFailed("authentication failed (check api_key)".into()),
        429 => Error::RateLimited { retry_after: 0 },
        404 => Error::ModelNotFound(format!("model not found (status {status})")),
        _ => Error::ApiError {
            status,
            message: sanitize_error_text(body, api_key),
        },
    }
}

#[cfg(test)]
fn map_api_error(status: u16, body: &str) -> Error {
    map_api_error_with_key(status, body, "")
}

/// 先精确移除当前 provider 的完整 key，再兜底移除任何 sk-/sk-ant- token。
/// 兜底扫描消费整个 token，而不是只替换前缀，避免把可重建的密钥后缀写入日志。
fn sanitize_error_text(value: &str, api_key: &str) -> String {
    let exact_redacted = if api_key.is_empty() {
        value.to_string()
    } else {
        value.replace(api_key, "***")
    };
    redact_api_key_tokens(&exact_redacted)
        .chars()
        .take(512)
        .collect()
}

fn redact_api_key_tokens(value: &str) -> String {
    let mut remaining = value;
    let mut redacted = String::with_capacity(value.len());
    while !remaining.is_empty() {
        if starts_with_ascii_ignore_case(remaining, "sk-ant-")
            || starts_with_ascii_ignore_case(remaining, "sk-")
        {
            let token_len = secret_token_len(remaining);
            if token_len >= "sk-".len() {
                redacted.push_str("***");
                remaining = &remaining[token_len..];
                continue;
            }
        }

        let ch = remaining
            .chars()
            .next()
            .expect("remaining error text must contain a character");
        redacted.push(ch);
        remaining = &remaining[ch.len_utf8()..];
    }
    redacted
}

fn starts_with_ascii_ignore_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

fn secret_token_len(value: &str) -> usize {
    value
        .char_indices()
        .take_while(|(_, ch)| {
            ch.is_ascii_alphanumeric()
                || matches!(ch, '-' | '_' | '.' | '~' | '+' | '/' | '=' | '%')
        })
        .map(|(index, ch)| index + ch.len_utf8())
        .last()
        .unwrap_or(0)
}

/// 网络错误信息必须使用当前 provider 的真实 key 脱敏。
fn sanitize_http_err<E: std::fmt::Display>(e: &E, api_key: &str) -> String {
    sanitize_error_text(&e.to_string(), api_key)
}

// ── SSE 流解析 ────────────────────────────────────────────────

fn parse_sse_stream(
    byte_stream: impl futures::Stream<
        Item = std::result::Result<bytes::Bytes, crate::openai::SseChunkError>,
    >,
    deepseek_automatic_cache: bool,
    api_key: String,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;

    // Keep raw bytes until a complete SSE frame is available. Decoding every transport chunk
    // independently would replace a multi-byte UTF-8 character when it straddles two chunks.
    let mut buffer = Vec::new();
    let mut state = AnthropicStreamState {
        deepseek_automatic_cache,
        api_key,
        ..AnthropicStreamState::default()
    };

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(c) => c,
                Err(error) => {
                    let error = sanitize_http_err(&error, &state.api_key);
                    tracing::debug!(error = %error, "provider stream transport error");
                    return vec![StreamEvent::Stop {
                        reason: StopReason::Other(format!(
                            "api_error: stream transport error: {error}"
                        )),
                    }];
                }
            };
            buffer.extend_from_slice(&chunk);

            let mut events = Vec::new();
            while let Some(frame_end) = find_sse_frame_end(&buffer) {
                let raw = buffer.drain(..frame_end).collect::<Vec<_>>();
                let raw = match std::str::from_utf8(&raw) {
                    Ok(raw) => raw,
                    Err(error) => {
                        tracing::debug!(%error, "provider stream SSE frame is not valid UTF-8");
                        continue;
                    }
                };
                if let Some(ev) = parse_one_sse(raw, &mut state) {
                    events.push(ev);
                }
            }
            events
        })
        .flat_map(futures::stream::iter)
}

fn find_sse_frame_end(buffer: &[u8]) -> Option<usize> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| position + 2);
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4);
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(end), None) | (None, Some(end)) => Some(end),
        (None, None) => None,
    }
}

#[derive(Default)]
struct AnthropicStreamState {
    /// Content block index is the stable association key used by Anthropic SSE deltas.
    tools: std::collections::HashMap<u64, PendingAnthropicTool>,
    deepseek_automatic_cache: bool,
    /// Kept only inside the provider stream so every surfaced/logged SSE error can redact the
    /// exact configured credential before applying the generic token fallback.
    api_key: String,
    /// A message_delta can contain usage and stop_reason together. Emit usage first and
    /// defer the stop event until message_stop so neither signal is lost.
    pending_stop: Option<StopReason>,
}

struct PendingAnthropicTool {
    id: String,
    name: String,
    input_json: String,
    hosted: bool,
}

fn parse_one_sse(raw: &str, state: &mut AnthropicStreamState) -> Option<StreamEvent> {
    let mut event_type = String::new();
    let mut data = String::new();
    for raw_line in raw.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let Some((field, value)) = line.split_once(':') else {
            continue;
        };
        // The SSE grammar permits no space after `:` and strips at most one optional space.
        // Kimi For Coding emits `event:message_start` / `data:{...}` in this valid form.
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event_type = value.trim().to_string(),
            "data" => {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value);
            }
            _ => {}
        }
    }
    if data.is_empty() {
        return None;
    }
    let v: Value = match serde_json::from_str(&data) {
        Ok(value) => value,
        Err(error) => {
            let error = sanitize_error_text(&error.to_string(), &state.api_key);
            tracing::debug!(error = %error, "provider stream JSON parse error");
            return None;
        }
    };

    match event_type.as_str() {
        "error" => Some(StreamEvent::Stop {
            reason: StopReason::Other(sse_error_detail(&v, &state.api_key)),
        }),
        "message_start" => {
            let usage = v.pointer("/message/usage")?;
            Some(StreamEvent::Usage(parse_anthropic_usage(
                usage,
                state.deepseek_automatic_cache,
            )))
        }
        "content_block_start" => {
            let block = v.get("content_block")?;
            let kind = block.get("type")?.as_str()?;
            if matches!(kind, "tool_use" | "server_tool_use") {
                let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                let id = block.get("id")?.as_str()?.to_string();
                let name = block.get("name")?.as_str()?.to_string();
                let hosted = kind == "server_tool_use";
                state.tools.insert(
                    index,
                    PendingAnthropicTool {
                        id: id.clone(),
                        name: name.clone(),
                        input_json: String::new(),
                        hosted,
                    },
                );
                (!hosted).then_some(StreamEvent::ToolUseStart { id, name })
            } else if kind.ends_with("_tool_result") {
                let id = block.get("tool_use_id")?.as_str()?.to_string();
                let content = block.get("content").cloned().unwrap_or(Value::Null);
                let is_error = content
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.ends_with("_error"));
                Some(StreamEvent::HostedToolResult {
                    id,
                    name: kind
                        .strip_suffix("_tool_result")
                        .unwrap_or(kind)
                        .to_string(),
                    output: public_hosted_tool_result(kind, &content, &state.api_key),
                    is_error,
                    provider_content: Some(block.clone()),
                })
            } else if kind == "thinking" {
                block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| StreamEvent::ReasoningDelta {
                        text: text.to_string(),
                    })
            } else {
                None
            }
        }
        "content_block_delta" => {
            let delta = v.get("delta")?;
            let kind = delta.get("type")?.as_str()?;
            match kind {
                "text_delta" => {
                    let text = delta.get("text")?.as_str()?.to_string();
                    Some(StreamEvent::TextDelta { text })
                }
                "thinking_delta" => {
                    let text = delta.get("thinking")?.as_str()?.to_string();
                    (!text.is_empty()).then_some(StreamEvent::ReasoningDelta { text })
                }
                "input_json_delta" => {
                    let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                    let partial = delta.get("partial_json")?.as_str()?.to_string();
                    let tool = state.tools.get_mut(&index)?;
                    tool.input_json.push_str(&partial);
                    (!tool.hosted).then(|| StreamEvent::ToolUseDelta {
                        id: tool.id.clone(),
                        input_json: partial,
                    })
                }
                _ => None,
            }
        }
        "content_block_stop" => {
            let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
            let tool = state.tools.remove(&index)?;
            let input = if tool.input_json.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&tool.input_json).unwrap_or(Value::Null)
            };
            if tool.hosted {
                let provider_content = json!({
                    "type": "server_tool_use",
                    "id": tool.id,
                    "name": tool.name,
                    "input": input,
                });
                Some(StreamEvent::HostedToolUse {
                    id: provider_content["id"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    name: provider_content["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    input: provider_content["input"].clone(),
                    provider_content: Some(provider_content),
                })
            } else {
                Some(StreamEvent::ToolUseComplete { id: tool.id, input })
            }
        }
        "message_delta" => {
            let stop = v
                .pointer("/delta/stop_reason")
                .and_then(|t| t.as_str())
                .map(parse_stop_reason);
            let output = v
                .pointer("/usage/output_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32;
            if let Some(reason) = stop {
                if output > 0 {
                    state.pending_stop = Some(reason);
                    Some(StreamEvent::Usage(Usage {
                        input_tokens: 0,
                        output_tokens: output,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    }))
                } else {
                    Some(StreamEvent::Stop { reason })
                }
            } else {
                Some(StreamEvent::Usage(Usage {
                    input_tokens: 0,
                    output_tokens: output,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                }))
            }
        }
        "message_stop" => state
            .pending_stop
            .take()
            .map(|reason| StreamEvent::Stop { reason }),
        _ => None,
    }
}

fn sse_error_detail(value: &Value, api_key: &str) -> String {
    let error = value.get("error").unwrap_or(value);
    let error_type = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("stream_error");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let detail = if message.is_empty() {
        error_type.to_string()
    } else {
        format!("{error_type}: {message}")
    };
    sanitize_error_text(&detail, api_key)
}

/// Search payloads contain encrypted page content that is required only for provider-side
/// continuation. Persisting it in UI tool logs would be noisy and unnecessarily sensitive, so
/// expose source metadata and errors only.
fn public_hosted_tool_result(kind: &str, content: &Value, api_key: &str) -> Value {
    if kind != "web_search_tool_result" {
        return json!({ "status": "completed" });
    }
    if let Some(results) = content.as_array() {
        return json!({
            "sources": results
                .iter()
                .map(|result| json!({
                    "title": result.get("title").cloned().unwrap_or(Value::Null),
                    "url": result.get("url").cloned().unwrap_or(Value::Null),
                    "page_age": result.get("page_age").cloned().unwrap_or(Value::Null),
                }))
                .collect::<Vec<_>>()
        });
    }
    json!({
        "error_code": content
            .get("error_code")
            .map(|value| sanitize_json_error_value(value, api_key))
            .unwrap_or(Value::Null)
    })
}

fn sanitize_json_error_value(value: &Value, api_key: &str) -> Value {
    match value {
        Value::String(value) => Value::String(sanitize_error_text(value, api_key)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| sanitize_json_error_value(value, api_key))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), sanitize_json_error_value(value, api_key)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system: None,
            messages: vec![Message::user_text("hello")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 32,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        }
    }

    fn long_tool_evidence() -> String {
        let mut evidence =
            "path=src/核心.rs; command=cargo test; exit=0; 证据=完整\n".repeat(4_096);
        evidence.push_str("__TOOL_EVIDENCE_TAIL_ANTHROPIC__");
        assert!(evidence.len() > 100_000);
        evidence
    }

    fn assert_credentials_absent(value: &str, credentials: &[&str]) {
        for credential in credentials {
            assert!(
                !value.contains(credential),
                "credential fragment {credential:?} leaked in {value:?}"
            );
        }
    }

    async fn read_test_http_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
        use tokio::io::AsyncReadExt;

        let mut request = Vec::new();
        let mut buffer = [0_u8; 2048];
        loop {
            let read = socket.read(&mut buffer).await.expect("read test request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, value)| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        request
    }

    fn test_http_request_body(request: &[u8]) -> &[u8] {
        request
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .map(|header_end| &request[header_end + 4..])
            .unwrap_or_default()
    }

    async fn serve_http_response_once(
        status: &str,
        body: String,
    ) -> (String, tokio::task::JoinHandle<()>) {
        serve_http_response_times(status, body, 1).await
    }

    async fn serve_http_response_times(
        status: &str,
        body: String,
        times: usize,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test HTTP listener");
        let address = listener.local_addr().expect("read test listener address");
        let status = status.to_string();
        let server = tokio::spawn(async move {
            for _ in 0..times {
                let (mut socket, _) = listener.accept().await.expect("accept test request");
                let _request = read_test_http_request(&mut socket).await;
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write test response");
            }
        });
        (format!("http://{address}"), server)
    }

    async fn serve_disconnect_once() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind disconnect listener");
        let address = listener.local_addr().expect("read test listener address");
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept test request");
            drop(socket);
        });
        (format!("http://{address}"), server)
    }

    #[test]
    fn messages_url_accepts_root_or_v1_base_url() {
        let from_root = AnthropicProvider::new(
            "sk-ant-test".into(),
            "test-model".into(),
            Some("https://api.example.com".into()),
        )
        .unwrap();
        let from_v1 = AnthropicProvider::new(
            "sk-ant-test".into(),
            "test-model".into(),
            Some("https://api.example.com/v1/".into()),
        )
        .unwrap();
        let deepseek = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();

        assert_eq!(
            from_root.messages_url(),
            "https://api.example.com/v1/messages"
        );
        assert_eq!(
            from_v1.messages_url(),
            "https://api.example.com/v1/messages"
        );
        assert_eq!(
            deepseek.messages_url(),
            "https://api.deepseek.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn deepseek_v4_and_chat_alias_declare_max_output() {
        let v4 = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        assert_eq!(v4.capabilities().max_output_tokens, 393_216);

        let chat = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-chat".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        assert_eq!(chat.capabilities().max_output_tokens, 393_216);
    }

    #[test]
    fn claude_effort_enables_adaptive_thinking() {
        let provider =
            AnthropicProvider::new("sk-ant-test".into(), "claude-opus-4-6".into(), None).unwrap();
        let request = CompletionRequest {
            model: "claude-opus-4-6".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 1024,
            temperature: None,
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: None,
                reasoning_effort: Some("high".into()),
                verbosity: None,
            },
        };

        let body = provider.build_request_body(&request);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn deepseek_anthropic_maps_local_adaptive_marker_to_native_enabled() {
        let provider = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        let mut request = test_completion_request();
        request.model = "deepseek-v4-pro".into();
        request.inference.thinking = Some("adaptive".into());

        let body = provider.build_request_body(&request);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["output_config"]["effort"], "high");
        assert!(!body.to_string().contains("adaptive"));
    }

    #[test]
    fn deepseek_anthropic_preserves_explicit_disabled_high_and_max_modes() {
        let provider = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        let mut request = test_completion_request();
        request.model = "deepseek-v4-pro".into();

        request.inference.thinking = Some("disabled".into());
        request.inference.reasoning_effort = Some("max".into());
        let disabled = provider.build_request_body(&request);
        assert_eq!(disabled["thinking"]["type"], "disabled");
        assert!(disabled.get("output_config").is_none());

        request.inference.thinking = Some("enabled".into());
        for effort in ["high", "max"] {
            request.inference.reasoning_effort = Some(effort.into());
            let body = provider.build_request_body(&request);
            assert_eq!(body["thinking"]["type"], "enabled");
            assert_eq!(body["output_config"]["effort"], effort);
        }
    }

    #[test]
    fn anthropic_keeps_native_adaptive_thinking() {
        let provider =
            AnthropicProvider::new("sk-ant-test".into(), "claude-opus-4-6".into(), None).unwrap();
        let mut request = test_completion_request();
        request.model = "claude-opus-4-6".into();
        request.inference.thinking = Some("adaptive".into());

        let body = provider.build_request_body(&request);
        assert_eq!(body["thinking"]["type"], "adaptive");
    }

    #[test]
    fn deepseek_anthropic_uses_automatic_cache_without_cache_control() {
        let deepseek = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        let anthropic = AnthropicProvider::new(
            "sk-test".into(),
            "claude-sonnet-5".into(),
            Some("https://api.anthropic.com".into()),
        )
        .unwrap();
        let request = CompletionRequest {
            model: "deepseek-v4-pro".into(),
            system: Some("stable system".into()),
            messages: vec![Message::user_text("hello")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 1024,
            temperature: None,
            enable_caching: true,
            inference: Default::default(),
        };

        let deepseek_body = deepseek.build_request_body(&request);
        let anthropic_body = anthropic.build_request_body(&request);
        assert_eq!(deepseek_body["system"], "stable system");
        assert!(!deepseek_body.to_string().contains("cache_control"));
        assert_eq!(
            anthropic_body["system"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(deepseek.name(), "deepseek_anthropic");
        assert!(deepseek.capabilities().supports_prompt_caching);
        assert_eq!(deepseek.capabilities().max_context_tokens, 1_000_000);
        assert!(!deepseek.capabilities().supports_vision);
    }

    #[test]
    fn message_start_stream_usage_includes_cache_tokens() {
        let mut state = AnthropicStreamState::default();
        let frame = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{",
            "\"input_tokens\":100,\"output_tokens\":0,",
            "\"cache_read_input_tokens\":80,",
            "\"cache_creation_input_tokens\":20}}}\n\n"
        );
        let Some(StreamEvent::Usage(usage)) = parse_one_sse(frame, &mut state) else {
            panic!("expected usage event");
        };
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(20));
    }

    #[test]
    fn kimi_sse_fields_without_spaces_preserve_text_after_thinking() {
        let mut state = AnthropicStreamState::default();
        let message_start = concat!(
            "event:message_start\n",
            "data:{\"type\":\"message_start\",\"message\":{\"usage\":{",
            "\"input_tokens\":12,\"output_tokens\":0}}}\n\n"
        );
        assert!(matches!(
            parse_one_sse(message_start, &mut state),
            Some(StreamEvent::Usage(Usage {
                input_tokens: 12,
                ..
            }))
        ));

        let thinking_delta = concat!(
            "event:content_block_delta\n",
            "data:{\"type\":\"content_block_delta\",\"index\":0,",
            "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"checking\"}}\n\n"
        );
        assert!(matches!(
            parse_one_sse(thinking_delta, &mut state),
            Some(StreamEvent::ReasoningDelta { text }) if text == "checking"
        ));

        let text_delta = concat!(
            "event:content_block_delta\n",
            "data:{\"type\":\"content_block_delta\",\"index\":1,",
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"Kimi replied\"}}\n\n"
        );
        assert!(matches!(
            parse_one_sse(text_delta, &mut state),
            Some(StreamEvent::TextDelta { text }) if text == "Kimi replied"
        ));
    }

    #[tokio::test]
    async fn sse_stream_handles_crlf_and_utf8_split_across_transport_chunks() {
        use futures::StreamExt as _;

        let frame = concat!(
            "event:content_block_delta\r\n",
            "data:{\"type\":\"content_block_delta\",\"index\":0,",
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"你好\"}}\r\n\r\n"
        );
        let split = frame.find('你').expect("Chinese text is present") + 1;
        let bytes = frame.as_bytes();
        let chunks = futures::stream::iter(vec![
            Ok::<_, crate::openai::SseChunkError>(bytes::Bytes::copy_from_slice(&bytes[..split])),
            Ok::<_, crate::openai::SseChunkError>(bytes::Bytes::copy_from_slice(&bytes[split..])),
        ]);

        let events = parse_sse_stream(chunks, false, String::new())
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::TextDelta { text }] if text == "你好"
        ));
    }

    #[test]
    fn deepseek_anthropic_usage_normalizes_total_input_and_cache_miss() {
        let cold = parse_anthropic_usage(
            &json!({
                "input_tokens": 169,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0
            }),
            true,
        );
        assert_eq!(cold.input_tokens, 169);
        assert_eq!(cold.cache_read_tokens, Some(0));
        assert_eq!(cold.cache_write_tokens, Some(169));

        let hit = parse_anthropic_usage(
            &json!({
                "input_tokens": 41,
                "output_tokens": 0,
                "cache_read_input_tokens": 128,
                "cache_creation_input_tokens": 0
            }),
            true,
        );
        assert_eq!(hit.input_tokens, 169);
        assert_eq!(hit.cache_read_tokens, Some(128));
        assert_eq!(hit.cache_write_tokens, Some(41));
    }

    #[test]
    fn message_delta_preserves_usage_when_stop_reason_is_in_the_same_frame() {
        let mut state = AnthropicStreamState::default();
        let delta = concat!(
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",",
            "\"delta\":{\"stop_reason\":\"end_turn\"},",
            "\"usage\":{\"output_tokens\":7}}\n\n"
        );
        let Some(StreamEvent::Usage(usage)) = parse_one_sse(delta, &mut state) else {
            panic!("expected usage before stop");
        };
        assert_eq!(usage.output_tokens, 7);

        let stop = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(matches!(
            parse_one_sse(stop, &mut state),
            Some(StreamEvent::Stop {
                reason: StopReason::EndTurn
            })
        ));
    }

    #[test]
    fn hosted_web_tools_use_the_anthropic_server_tool_schemas() {
        let provider = AnthropicProvider::new(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://api.deepseek.com/anthropic".into()),
        )
        .unwrap();
        let request = CompletionRequest {
            model: "deepseek-v4-pro".into(),
            system: None,
            messages: vec![Message::user_text("search Rust news")],
            tools: vec![],
            hosted_tools: vec![HostedToolSpec::web_search(), HostedToolSpec::web_fetch()],
            max_tokens: 1024,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };

        let body = provider.build_request_body(&request);
        assert_eq!(body["tools"][0]["type"], "web_search_20250305");
        assert_eq!(body["tools"][0]["name"], "web_search");
        assert_eq!(body["tools"][0]["max_uses"], 5);
        assert!(body["tools"][0].get("input_schema").is_none());
        assert_eq!(body["tools"][1]["type"], "web_fetch_20250910");
        assert_eq!(body["tools"][1]["name"], "web_fetch");
        assert_eq!(body["tools"][1]["max_uses"], 5);
    }

    #[test]
    fn hosted_web_search_stream_is_observable_without_exposing_encrypted_content() {
        let mut state = AnthropicStreamState::default();
        let start = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,",
            "\"content_block\":{\"type\":\"server_tool_use\",",
            "\"id\":\"srvtoolu_1\",\"name\":\"web_search\"}}\n\n"
        );
        assert!(parse_one_sse(start, &mut state).is_none());

        let delta = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,",
            "\"delta\":{\"type\":\"input_json_delta\",",
            "\"partial_json\":\"{\\\"query\\\":\\\"Rust 2026\\\"}\"}}\n\n"
        );
        assert!(parse_one_sse(delta, &mut state).is_none());

        let stop = concat!(
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n"
        );
        assert!(matches!(
            parse_one_sse(stop, &mut state),
            Some(StreamEvent::HostedToolUse {
                id, name, input, ..
            })
                if id == "srvtoolu_1"
                    && name == "web_search"
                    && input["query"] == "Rust 2026"
        ));

        let result = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":2,",
            "\"content_block\":{\"type\":\"web_search_tool_result\",",
            "\"tool_use_id\":\"srvtoolu_1\",\"content\":[{",
            "\"type\":\"web_search_result\",\"title\":\"Rust\",",
            "\"url\":\"https://www.rust-lang.org\",",
            "\"encrypted_content\":\"provider-private\"}]}}\n\n"
        );
        let Some(StreamEvent::HostedToolResult {
            output,
            provider_content,
            ..
        }) = parse_one_sse(result, &mut state)
        else {
            panic!("expected hosted tool result");
        };
        assert_eq!(output["sources"][0]["url"], "https://www.rust-lang.org");
        assert!(!output.to_string().contains("provider-private"));
        assert!(provider_content
            .expect("provider continuation block")
            .to_string()
            .contains("provider-private"));
    }

    #[test]
    fn hosted_provider_blocks_roundtrip_for_protocol_continuation() {
        let raw = json!({
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{
                "type": "web_search_result",
                "url": "https://www.rust-lang.org",
                "encrypted_content": "provider-private",
            }],
        });
        let block = hosted_content_block(&raw).expect("hosted custom block");
        let message = Message {
            role: Role::Assistant,
            content: vec![block],
        };

        assert_eq!(message_to_anthropic(&message)["content"][0], raw);
    }

    #[test]
    fn message_conversion_preserves_text() {
        let m = Message::user_text("hello");
        let v = message_to_anthropic(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hello");
    }

    #[test]
    fn tool_result_conversion_includes_is_error() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "fail".into(),
                is_error: true,
            }],
        };
        let v = message_to_anthropic(&m);
        assert_eq!(v["content"][0]["is_error"], true);
    }

    #[test]
    fn custom_deepseek_gateway_preserves_paired_long_tool_evidence() {
        let provider = AnthropicProvider::new_deepseek(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            Some("https://gateway.example/deepseek/anthropic".into()),
        )
        .expect("deepseek anthropic provider");
        let evidence = long_tool_evidence();
        let mut request = test_completion_request();
        request.model = "deepseek-v4-pro".into();
        request.messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_read".into(),
                    name: "read_file".into(),
                    input: json!({"path": "src/核心.rs"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_read".into(),
                    content: evidence.clone(),
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body(&request);
        let messages = body["messages"].as_array().expect("anthropic messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"][0]["type"], "tool_use");
        assert_eq!(messages[0]["content"][0]["id"], "toolu_read");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "toolu_read");
        assert_eq!(
            messages[1]["content"][0]["content"].as_str(),
            Some(evidence.as_str())
        );
        assert!(messages[1]["content"][0]["content"]
            .as_str()
            .is_some_and(|content| content.ends_with("__TOOL_EVIDENCE_TAIL_ANTHROPIC__")));
        assert_eq!(provider.name(), "deepseek_anthropic");
    }

    #[test]
    fn standalone_anthropic_tool_result_is_not_silently_clipped_or_deleted() {
        // Anthropic 请求构造不会像 Chat/Responses 那样删除孤立结果；原子工具轮次
        // 必须由上游保证。这里固定“Provider 不静默损失证据”的边界语义。
        let evidence = long_tool_evidence();
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_orphan".into(),
                content: evidence.clone(),
                is_error: true,
            }],
        };

        let converted = message_to_anthropic(&message);
        assert_eq!(converted["content"][0]["tool_use_id"], "toolu_orphan");
        assert_eq!(converted["content"][0]["content"], evidence);
        assert_eq!(converted["content"][0]["is_error"], true);
    }

    #[test]
    fn unsupported_block_degrades_to_placeholder() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::Custom {
                type_name: "file_ref".into(),
                data: json!({"path": "/a"}),
            }],
        };
        let v = message_to_anthropic(&m);
        assert_eq!(v["content"][0]["type"], "text");
        assert!(v["content"][0]["text"].as_str().unwrap().contains("custom"));
    }

    #[test]
    fn pdf_attachment_becomes_anthropic_document_input() {
        let m = Message {
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
        let value = message_to_anthropic(&m);
        assert_eq!(value["content"][0]["type"], "document");
        assert_eq!(value["content"][0]["title"], "spec.pdf");
    }

    #[test]
    fn parse_response_text_and_usage() {
        let v = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 3 }
        });
        let r = parse_anthropic_response(&v, false).unwrap();
        assert_eq!(r.text(), "hi");
        assert_eq!(r.usage.output_tokens, 3);
    }

    #[test]
    fn sanitizer_redacts_exact_and_sk_credentials_through_punctuation_boundaries() {
        let configured_key = "opaque:provider/credential?tail=CONFIGURED_TAIL";
        let input = format!(
            "configured=[{configured_key}], generic=(sk-basic_BASIC_TAIL), \
             anthropic='sk-ant-secret.ANTHROPIC_TAIL+/%='; uppercase=<SK-UPPER_UPPER_TAIL>"
        );

        let sanitized = sanitize_error_text(&input, configured_key);

        assert_eq!(
            sanitized,
            "configured=[***], generic=(***), anthropic='***'; uppercase=<***>"
        );
        assert_credentials_absent(
            &sanitized,
            &[
                configured_key,
                "CONFIGURED_TAIL",
                "BASIC_TAIL",
                "ANTHROPIC_TAIL",
                "UPPER_TAIL",
            ],
        );
    }

    #[test]
    fn api_error_message_redacts_nonstandard_configured_key_and_generic_fallback() {
        let configured_key = "custom-provider-credential-CONFIGURED_HTTP_TAIL";
        let body = format!(
            "provider returned [{configured_key}], fallback sk-ant-STATUS_FALLBACK_TAIL; safe"
        );

        let Error::ApiError { status, message } =
            map_api_error_with_key(500, &body, configured_key)
        else {
            panic!("expected API error");
        };

        assert_eq!(status, 500);
        assert_eq!(message, "provider returned [***], fallback ***; safe");
        assert_credentials_absent(
            &message,
            &[
                configured_key,
                "CONFIGURED_HTTP_TAIL",
                "STATUS_FALLBACK_TAIL",
            ],
        );
    }

    #[test]
    fn sse_error_event_redacts_credentials_and_suffixes() {
        let configured_key = "opaque-sse-credential-CONFIGURED_SSE_TAIL";
        let payload = json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": format!(
                    "bad [{configured_key}], fallback sk-SSE_FALLBACK_TAIL; safe"
                )
            }
        });
        let frame = format!("event: error\ndata: {payload}\n\n");
        let mut state = AnthropicStreamState {
            api_key: configured_key.into(),
            ..AnthropicStreamState::default()
        };

        let Some(StreamEvent::Stop {
            reason: StopReason::Other(detail),
        }) = parse_one_sse(&frame, &mut state)
        else {
            panic!("expected sanitized SSE error event");
        };

        assert_eq!(
            detail,
            "invalid_request_error: bad [***], fallback ***; safe"
        );
        assert_credentials_absent(
            &detail,
            &[configured_key, "CONFIGURED_SSE_TAIL", "SSE_FALLBACK_TAIL"],
        );
    }

    #[test]
    fn hosted_tool_warning_fields_are_recursively_sanitized() {
        let configured_key = "opaque-warning-credential-CONFIGURED_WARNING_TAIL";
        let content = json!({
            "error_code": {
                "message": format!("{configured_key}, sk-WARNING_FALLBACK_TAIL; safe"),
                "details": [format!("again {configured_key}"), 7]
            }
        });

        let output = public_hosted_tool_result("web_search_tool_result", &content, configured_key);
        let rendered = output.to_string();

        assert_eq!(output["error_code"]["message"], "***, ***; safe");
        assert_eq!(output["error_code"]["details"][0], "again ***");
        assert_eq!(output["error_code"]["details"][1], 7);
        assert_credentials_absent(
            &rendered,
            &[
                configured_key,
                "CONFIGURED_WARNING_TAIL",
                "WARNING_FALLBACK_TAIL",
            ],
        );
    }

    #[test]
    fn retry_policy_covers_transient_statuses_and_bounds_retry_after() {
        assert!(retryable_status(408));
        assert!(retryable_status(429));
        assert!(retryable_status(500));
        assert!(retryable_status(599));
        assert!(!retryable_status(400));
        assert!(!retryable_status(404));

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "120".parse().unwrap());
        let retry_after = parse_retry_after(&headers);
        assert_eq!(retry_after, Some(Duration::from_secs(120)));
        assert_eq!(backoff_delay(1, retry_after), MAX_RETRY_AFTER);
    }

    #[tokio::test]
    async fn complete_retries_transient_statuses_with_the_same_serialized_body() {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind retry fixture");
        let address = listener.local_addr().expect("read retry fixture address");
        let responses = [
            (
                "500 Internal Server Error",
                "",
                r#"{"type":"error","error":{"message":"temporary"}}"#,
            ),
            (
                "429 Too Many Requests",
                "Retry-After: 0\r\n",
                r#"{"type":"error","error":{"message":"slow down"}}"#,
            ),
            (
                "200 OK",
                "",
                r#"{"content":[{"type":"text","text":"recovered"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":1}}"#,
            ),
        ];
        let server = tokio::spawn(async move {
            let mut bodies = Vec::new();
            for (status, extra_headers, body) in responses {
                let (mut socket, _) = listener.accept().await.expect("accept retry request");
                let request = read_test_http_request(&mut socket).await;
                bodies.push(test_http_request_body(&request).to_vec());
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write retry response");
            }
            bodies
        });
        let provider = AnthropicProvider::new(
            "test-key".into(),
            "test-model".into(),
            Some(format!("http://{address}")),
        )
        .expect("construct provider");

        let response = provider
            .complete(test_completion_request())
            .await
            .expect("transient failures should recover");
        let bodies = server.await.expect("retry fixture must finish");

        assert_eq!(response.text(), "recovered");
        assert_eq!(bodies.len(), 3);
        assert!(bodies.iter().all(|body| body == &bodies[0]));
        let request: Value = serde_json::from_slice(&bodies[0]).expect("request body is JSON");
        assert_eq!(request["model"], "test-model");
        assert_eq!(request["stream"], Value::Null);
    }

    #[tokio::test]
    async fn complete_redacts_credentials_from_http_status_body() {
        let configured_key = "opaque-http-credential-CONFIGURED_STATUS_TAIL";
        let body = format!("provider [{configured_key}], fallback sk-ant-HTTP_FALLBACK_TAIL; safe");
        let (base_url, server) = serve_http_response_times(
            "500 Internal Server Error",
            body,
            (MAX_RETRIES + 1) as usize,
        )
        .await;
        let provider =
            AnthropicProvider::new(configured_key.into(), "test-model".into(), Some(base_url))
                .expect("construct provider");

        let error = provider
            .complete(test_completion_request())
            .await
            .expect_err("HTTP 500 must fail");
        server.await.expect("HTTP fixture must finish");
        let Error::ApiError { status, message } = error else {
            panic!("expected API error");
        };

        assert_eq!(status, 500);
        assert_credentials_absent(
            &message,
            &[
                configured_key,
                "CONFIGURED_STATUS_TAIL",
                "HTTP_FALLBACK_TAIL",
            ],
        );
        assert!(message.contains("provider [***], fallback ***; safe"));
    }

    #[tokio::test]
    async fn complete_redacts_credentials_from_json_decode_errors() {
        let configured_key = "opaque-json-credential-CONFIGURED_DECODE_TAIL";
        let invalid_json = format!("{{\"secret\":\"{configured_key}\"");
        let (base_url, server) = serve_http_response_once("200 OK", invalid_json).await;
        let provider = AnthropicProvider::new(
            configured_key.into(),
            "test-model".into(),
            Some(format!("{base_url}/{configured_key}")),
        )
        .expect("construct provider");

        let error = provider
            .complete(test_completion_request())
            .await
            .expect_err("invalid JSON must fail");
        server.await.expect("HTTP fixture must finish");
        let Error::Provider(message) = error else {
            panic!("expected provider decode error");
        };

        assert!(message.starts_with("invalid response json:"));
        assert_credentials_absent(&message, &[configured_key, "CONFIGURED_DECODE_TAIL"]);
    }

    #[tokio::test]
    async fn complete_redacts_credentials_from_transport_errors() {
        let configured_key = "opaque-transport-credential-CONFIGURED_TRANSPORT_TAIL";
        let (base_url, server) = serve_disconnect_once().await;
        let provider = AnthropicProvider::new(
            configured_key.into(),
            "test-model".into(),
            Some(format!("{base_url}/{configured_key}")),
        )
        .expect("construct provider");

        let error = provider
            .complete(test_completion_request())
            .await
            .expect_err("disconnected transport must fail");
        server.await.expect("disconnect fixture must finish");
        let Error::Provider(message) = error else {
            panic!("expected provider transport error");
        };

        assert_credentials_absent(&message, &[configured_key, "CONFIGURED_TRANSPORT_TAIL"]);
    }

    #[test]
    fn map_api_error_never_leaks_key() {
        let e = map_api_error(500, "some sk-ant-SECRET text");
        let msg = e.to_string();
        assert!(!msg.contains("sk-ant-SECRET"));
        assert!(msg.contains("***"));
    }

    #[test]
    fn map_api_error_401_is_auth() {
        let e = map_api_error(401, "bad");
        assert!(matches!(e, Error::AuthFailed(_)));
    }

    #[test]
    fn map_api_error_429_is_rate_limited() {
        let e = map_api_error(429, "slow down");
        assert!(matches!(e, Error::RateLimited { .. }));
    }

    #[test]
    fn kimi_anthropic_dialect_omits_temperature_and_nests_effort() {
        use crate::dialect::{dialect_for, DialectPort};
        let provider = AnthropicProvider::new(
            "k".into(),
            "k3-256k".into(),
            Some("https://api.kimi.com/coding".into()),
        )
        .unwrap()
        .with_dialect(
            dialect_for("kimi_coding", "k3-256k", DialectPort::AnthropicMessages)
                .expect("kimi anthropic dialect"),
        );
        let mut request = test_completion_request();
        request.model = "k3-256k".into();
        request.temperature = Some(0.1);
        request.inference.thinking = Some("adaptive".into());
        request.inference.reasoning_effort = Some("high".into());
        let body = provider.build_request_body(&request);
        assert!(body.get("temperature").is_none());
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["effort"], "high");
    }

    #[test]
    fn ark_anthropic_dialect_passthrough_thinking_and_drops_effort() {
        use crate::dialect::{dialect_for, DialectPort};
        let provider = AnthropicProvider::new(
            "k".into(),
            "ark-code-latest".into(),
            Some("https://ark.cn-beijing.volces.com/api/coding".into()),
        )
        .unwrap()
        .with_dialect(
            dialect_for(
                "ark_coding",
                "ark-code-latest",
                DialectPort::AnthropicMessages,
            )
            .expect("ark anthropic dialect"),
        );
        let mut request = test_completion_request();
        request.model = "ark-code-latest".into();
        request.temperature = Some(0.3);
        request.inference.thinking = Some("adaptive".into());
        request.inference.reasoning_effort = Some("high".into());
        let body = provider.build_request_body(&request);
        assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert!(body.get("output_config").is_none());
    }
}
