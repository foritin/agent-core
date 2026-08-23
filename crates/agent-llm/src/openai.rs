//! OpenAI 兼容 Provider（Chat Completions API）。
//!
//! 参见 `01-llm-provider.html §4.2`。system 作为消息角色注入；不支持 prompt
//! caching。错误信息不含 api_key。

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use agent_contract::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, HostedToolFormat,
    HostedToolSpec, LlmProvider, Message, Role, StopReason, StreamEvent, ToolSpec, Usage,
};
use agent_error::{Error, Result};
use serde_json::{json, Value};

use crate::dialect::{dialect_thinking_value, EffortWire, WireDialect};
use crate::url::openai_api_root;

const MAX_ERROR_MESSAGE_CHARS: usize = 240;
const MAX_ERROR_METADATA_CHARS: usize = 64;

// ── 连接层重试（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）──────────
//
// 对齐 Reasonix `internal/provider/retry.go`：连接 + header 阶段失败时指数
// 退避重试；body 流一旦开始就不重试（模型可能已产出 token）。重试复用同一
// 请求体（`build_body` 结果），保证重试字节与首试逐字节一致，不破坏
// DeepSeek 前缀缓存（PRD §4 原则 8）。

/// 首试之后最多重试次数（总计最多 MAX_RETRIES + 1 次尝试）。
const MAX_RETRIES: u32 = 10;
/// 指数退避封顶：500ms * 2^(n-1)，最大 15s。
const MAX_BACKOFF: Duration = Duration::from_secs(15);
/// 服务端 `Retry-After` 的等待上限（限流窗口通常比自身退避封顶更长，长等
/// 待可被取消，不会把用户锁死）。
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
/// 读取非 2xx 错误体的超时：网关在半开连接上发完 header 后可能停住 body，
/// 没有该 deadline 重试循环会在 io 读取上无限阻塞。
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(10);

// ── 流空闲 watchdog（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）──────
//
// 对齐 Reasonix `openai.go:43-50`（defaultStreamIdleTimeout = 120s）：SSE 流
// 超过该时长无任何新字节，视为半开 TCP 连接（代理切换、服务端静默），主动
// 关闭连接并以可恢复错误终止流。与"连接断开"（瞬时传输错误）区分，调用方
// 可据此用同一冻结请求重放。
pub(crate) const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// 流空闲超时终止时发出的 `StopReason::Other` 标记，agent 层据此走恢复路径。
pub(crate) const STREAM_IDLE_TIMEOUT_REASON: &str = "stream_idle_timeout";

// ── 悬空工具调用对的占位结果（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）
//
// 对齐 Reasonix `provider.go` 的 `interruptedToolResult`：assistant tool_calls
// 必须有对应 tool 结果，否则 DeepSeek 直接 400
// （"An assistant message with 'tool_calls' must be followed by tool messages…"）。
const INTERRUPTED_TOOL_RESULT_PLACEHOLDER: &str =
    "[no result: the previous turn was interrupted before this tool call completed]";

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// 配置的默认模型（请求可覆盖）。
    #[allow(dead_code)]
    model: String,
    /// DeepSeek 自定义网关仍需显式请求流式 usage；不能只靠官方域名判断。
    force_stream_usage: bool,
    /// 厂商/模型族方言；None 保持与改造前一致的通用 OpenAI 行为。
    dialect: Option<WireDialect>,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .user_agent(concat!("R-Code/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("openai provider client"),
            api_key,
            base_url,
            model,
            force_stream_usage: false,
            dialect: None,
        }
    }

    /// 强制请求流末尾的 usage 帧。仅供明确保留了 DeepSeek 身份的包装层使用；
    /// 普通 OpenAI 兼容网关仍维持按已知官方域名启用的保守行为。
    pub(crate) fn with_stream_usage(mut self) -> Self {
        self.force_stream_usage = true;
        self
    }

    /// 应用实测冻结的厂商方言；UA 覆盖会重建 client。
    pub(crate) fn with_dialect(mut self, dialect: WireDialect) -> Self {
        if let Some(user_agent) = dialect.user_agent {
            self.client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .user_agent(user_agent)
                .build()
                .expect("openai provider client with dialect ua");
        }
        self.dialect = Some(dialect);
        self
    }

    fn completions_url(&self) -> String {
        // 版本段由 base_url 自带（/v1、/v3、/v4、/compatible-mode/v1 …），
        // 只有裸域名才补 /v1。规则与理由见 `crate::url`。
        format!("{}/chat/completions", openai_api_root(&self.base_url))
    }

    /// 官方 OpenAI 端点与 DeepSeek 端点支持在流末额外返回 usage 帧；其余
    /// OpenAI 兼容接口不假定支持该扩展。
    ///
    /// DeepSeek 的自动前缀缓存字段（`prompt_cache_hit_tokens` /
    /// `prompt_cache_miss_tokens`）只在 usage 帧中出现，不带
    /// `stream_options.include_usage` 则真实流式会话收不到 usage 帧
    /// （https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §3 A14，P0-B 前置）。
    fn supports_stream_usage(&self) -> bool {
        self.force_stream_usage
            || self
                .dialect
                .as_ref()
                .is_some_and(|dialect| dialect.force_stream_usage)
            || reqwest::Url::parse(self.base_url.trim())
                .ok()
                .and_then(|url| {
                    url.host_str().map(|host| {
                        host.eq_ignore_ascii_case("api.openai.com")
                            || host.eq_ignore_ascii_case("api.deepseek.com")
                    })
                })
                .unwrap_or(false)
    }

    fn build_body(&self, request: &CompletionRequest, stream: bool) -> Value {
        let mut messages: Vec<Value> = Vec::new();

        // fail closed：未物化的附件引用不得进入最终序列化（降级为占位文本
        // 会静默丢图）。build_body 是无 IO 的纯构造，这里以断言失败 panic 之外
        // 的方式不可行——调用方（complete/stream）已在入口先行校验，此处再以
        // debug_assert 兜底，保证任何新调用路径都不会静默吞掉引用。
        debug_assert!(
            crate::assert_no_unresolved_attachments(&request.messages).is_ok(),
            "unresolved ContentBlock::Attachment reached openai build_body"
        );

        // system 作为角色注入
        if let Some(system) = &request.system {
            messages.push(json!({ "role": "system", "content": system }));
        }

        let model_lower = request.model.to_ascii_lowercase();
        // `adaptive` 是 R-Code 的本地策略标记：DeepSeek 走原有 governor 语义，
        // 实测冻结的 Ark/Kimi 方言按方言表翻译；未命中的通用网关保持旧行为。
        let dialect_echo = self
            .dialect
            .as_ref()
            .is_some_and(|dialect| dialect.echo_reasoning);
        let (thinking, reasoning_effort, dialect_omit_temp_for_thinking) = if let Some(dialect) =
            &self.dialect
        {
            let thinking = dialect_thinking_value(dialect, request.inference.thinking.as_deref());
            let reasoning_effort = match dialect.effort_wire {
                EffortWire::TopLevel => {
                    request
                        .inference
                        .reasoning_effort
                        .as_deref()
                        .filter(|value| {
                            dialect.effort_vocab.is_empty() || dialect.effort_vocab.contains(value)
                        })
                }
                _ => None,
            };
            (
                thinking,
                reasoning_effort,
                dialect.omit_temperature_when_thinking && matches!(thinking, Some("enabled")),
            )
        } else {
            let is_deepseek = self.force_stream_usage || model_lower.contains("deepseek");
            // 实测：deepseek-chat 已被服务端别名到 deepseek-v4-flash。
            let is_deepseek_v4 = is_deepseek
                && (model_lower.contains("deepseek-v4") || model_lower == "deepseek-chat");
            let thinking = request.inference.thinking.as_deref().map(|thinking| {
                if is_deepseek_v4 && thinking == "adaptive" {
                    "enabled"
                } else {
                    thinking
                }
            });
            let reasoning_effort = if is_deepseek_v4 {
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
            (
                thinking,
                reasoning_effort,
                is_deepseek && matches!(thinking, Some("enabled")),
            )
        };

        let deepseek_thinking = self.dialect.is_none()
            && (self.force_stream_usage || model_lower.contains("deepseek"))
            && matches!(thinking, Some("enabled"));
        let omit_temperature = self
            .dialect
            .as_ref()
            .is_some_and(|dialect| dialect.omit_temperature)
            || dialect_omit_temp_for_thinking
            || deepseek_thinking;

        // 发送前修复悬空工具调用对（Reasonix `SanitizeToolPairing`）：健康历史
        // 零拷贝透传（返回 None），只有存在未配对 tool_call / 孤儿 ToolResult
        // 时才构造修复副本。DeepSeek 对未配对的 assistant tool_calls 直接 400
        // （https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）。
        let sanitized = sanitize_tool_pairing(&request.messages);
        match &sanitized {
            Some(msgs) => {
                for message in msgs {
                    messages.extend(messages_to_openai(message, dialect_echo));
                }
            }
            None => {
                for message in &request.messages {
                    messages.extend(messages_to_openai(message, dialect_echo));
                }
            }
        }

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });
        if let Some(temp) = request.temperature.filter(|_| !omit_temperature) {
            body["temperature"] = json!(temp);
        }
        if let Some(thinking) = thinking {
            if model_lower.contains("qwen") {
                body["enable_thinking"] = json!(thinking != "disabled");
            } else {
                body["thinking"] = json!({ "type": thinking });
            }
        }
        if let Some(effort) = reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        if let Some(verbosity) = request.inference.verbosity.as_deref() {
            body["verbosity"] = json!(verbosity);
        }
        if stream {
            body["stream"] = json!(true);
            if self.supports_stream_usage() {
                body["stream_options"] = json!({ "include_usage": true });
            }
        }
        let mut tools = request.tools.iter().map(tool_to_openai).collect::<Vec<_>>();
        tools.extend(
            request
                .hosted_tools
                .iter()
                .filter_map(hosted_tool_to_openai),
        );
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        // 键恒发（Reasonix `openai.go:688-735`，https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5
        // P1-F）：thinking 模式对 assistant tool_calls 轮恒发 `reasoning_content`
        // 键（空串可接受，缺键 DeepSeek 400 "must be passed back"）；tool 消息
        // 恒发 `name` 键（空值可接受，严格兼容口缺键 400）。
        apply_key_emission(
            body["messages"].as_array_mut().expect("messages array"),
            deepseek_thinking,
            request,
        );
        body
    }
}

/// Chat Completions has no standard hosted-web schema. OpenRouter defines explicit server-tool
/// types on this endpoint; every other format stays out instead of masquerading as a function.
fn hosted_tool_to_openai(tool: &HostedToolSpec) -> Option<Value> {
    match tool {
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
        _ => None,
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

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let body = self.build_body(&request, false);
        let resp =
            send_with_retry(&self.client, &self.completions_url(), &self.api_key, &body).await?;
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("invalid response json: {e}")))?;
        parse_openai_response(&v)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        crate::assert_no_unresolved_attachments(&request.messages)?;
        let mut body = self.build_body(&request, true);
        let resp = match send_with_retry(
            &self.client,
            &self.completions_url(),
            &self.api_key,
            &body,
        )
        .await
        {
            Ok(resp) => resp,
            Err(error) if self.force_stream_usage && stream_options_rejected(&error) => {
                // 某些旧代理实现了 DeepSeek Chat/SSE，却拒绝 OpenAI 的 stream_options。
                // 首次请求在参数校验阶段失败，移除该扩展后兼容重试；此时只能放弃
                // 本轮 usage 观测，正文生成仍可继续。
                body.as_object_mut()
                    .expect("OpenAI request body must be an object")
                    .remove("stream_options");
                send_with_retry(&self.client, &self.completions_url(), &self.api_key, &body).await?
            }
            Err(error) => return Err(error),
        };
        // 流空闲 watchdog：SSE 流超过 DEFAULT_STREAM_IDLE_TIMEOUT 无新字节即
        // 视为流死（半开 TCP），主动关闭连接并以可恢复错误终止（P1-E）。
        Ok(Box::pin(parse_openai_sse_with_idle_timeout(
            resp.bytes_stream(),
            DEFAULT_STREAM_IDLE_TIMEOUT,
        )))
    }

    fn capabilities(&self) -> Capabilities {
        if let Some(dialect) = &self.dialect {
            return Capabilities {
                supports_streaming: true,
                supports_tool_use: true,
                supports_vision: dialect.supports_vision,
                supports_prompt_caching: false,
                max_context_tokens: dialect.max_context_tokens,
                max_output_tokens: dialect.max_output_tokens,
            };
        }
        Capabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
            supports_prompt_caching: false,
            max_context_tokens: 128_000,
            max_output_tokens: 0,
        }
    }

    fn echoes_reasoning(&self) -> bool {
        self.dialect
            .as_ref()
            .is_some_and(|dialect| dialect.echo_reasoning)
    }

    fn name(&self) -> &str {
        "openai"
    }
}

fn stream_options_rejected(error: &Error) -> bool {
    matches!(
        error,
        Error::ApiError { status: 400 | 422, message }
            if {
                let message = message.to_ascii_lowercase();
                message.contains("stream_options")
                    || message.contains("stream options")
                    || message.contains("include_usage")
            }
    )
}

// ── 连接层重试（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）──────────

/// 无 tokio 依赖的延时 future（crate 不依赖 tokio，futures 默认 features 也
/// 不含 channel）：短暂线程 sleep 后通过共享 waker 槽唤醒等待方。
/// 仅退避/错误体超时路径使用（失败时最多 spawn MAX_RETRIES 个短暂线程）；
/// abort 时 future 被 drop，线程发送失败后自行退出，不泄漏。
async fn thread_sleep(delay: Duration) {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let waker_slot: std::sync::Arc<std::sync::Mutex<Option<std::task::Waker>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let (slot_for_thread, tx_for_thread) = (waker_slot.clone(), tx.clone());
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        if let Some(waker) = slot_for_thread.lock().unwrap().take() {
            waker.wake();
        }
        let _ = tx_for_thread.send(());
    });
    let mut done = false;
    futures::future::poll_fn(move |cx| {
        use std::task::Poll;
        if done {
            return Poll::Ready(());
        }
        *waker_slot.lock().unwrap() = Some(cx.waker().clone());
        match rx.try_recv() {
            Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                done = true;
                Poll::Ready(())
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Poll::Pending,
        }
    })
    .await
}

/// 只有 408 / 429 / 5xx 可能被退避重试救回；其余 4xx（400/401/402/422 …）
/// 是调用方/配置问题，重试无意义（Reasonix `RetryableStatus`）。
fn retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500..=599)
}

/// 指数退避：500ms * 2^(attempt-1) 封顶 MAX_BACKOFF；服务端给了
/// `Retry-After` 则优先尊重它（封顶 MAX_RETRY_AFTER）。
/// 未加抖动（无 rand 依赖；Reasonix 的 ±250ms jitter 非必需）。
fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(after) = retry_after {
        return after.min(MAX_RETRY_AFTER);
    }
    let base = Duration::from_millis(500 * 2u64.pow(attempt.saturating_sub(1)));
    base.min(MAX_BACKOFF)
}

/// 解析 `Retry-After` 头（RFC 9110 delta-seconds 形式；HTTP-date 形式需要
/// httpdate 依赖，目前不支持，返回 None 走自身退避）。
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    value.parse::<u64>().ok().map(Duration::from_secs)
}

/// 读取非 2xx 错误体，带 ERROR_BODY_READ_TIMEOUT 硬超时：网关在半开连接上
/// 发完 header 后可能停住 body，没有该 deadline 重试循环会无限阻塞。
async fn read_error_body(resp: reqwest::Response) -> String {
    let text_fut = resp.text();
    futures::pin_mut!(text_fut);
    let timeout = thread_sleep(ERROR_BODY_READ_TIMEOUT);
    futures::pin_mut!(timeout);
    match futures::future::select(&mut text_fut, &mut timeout).await {
        futures::future::Either::Left((result, _)) => result.unwrap_or_default(),
        futures::future::Either::Right(_) => String::new(),
    }
}

/// 发送请求并在连接 + header 阶段按需重试：传输错误与可重试状态码
/// （408/429/5xx）指数退避重试（最多 MAX_RETRIES 次），`Retry-After` 优先；
/// 4xx（含 AuthFailed）不重试；abort 通过 drop future 立即生效（重试退避的
/// sleep 一并被取消，Reasonix `SendWithRetry` 语义）。body 复用同一 `Value`，
/// 重试字节与首试逐字节一致（不破坏 DeepSeek 前缀缓存）。
#[allow(unused_assignments)] // 保留最近一次错误供最终返回；重试循环覆盖是预期行为
pub(crate) async fn send_with_retry(
    client: &reqwest::Client,
    completions_url: &str,
    api_key: &str,
    body: &Value,
) -> Result<reqwest::Response> {
    let mut attempt: u32 = 0;
    let mut last_error: Option<Error> = None;
    let mut retry_after: Option<Duration> = None;

    loop {
        attempt += 1;
        if attempt > 1 {
            let delay = backoff_delay(attempt - 1, retry_after);
            tracing::debug!(attempt, ?delay, "provider request failed, retrying");
            thread_sleep(delay).await;
        }
        retry_after = None;

        let result = client
            .post(completions_url)
            .bearer_auth(api_key)
            .json(body)
            .send()
            .await;

        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(resp);
                }
                retry_after = parse_retry_after(resp.headers());
                let text = read_error_body(resp).await;
                let err = map_api_error(status.as_u16(), &text, api_key);
                if !retryable_status(status.as_u16()) {
                    return Err(err);
                }
                last_error = Some(err);
            }
            Err(transport_err) => {
                last_error = Some(Error::Provider(sanitize_transport_error(
                    &transport_err.to_string(),
                    api_key,
                )));
            }
        }

        if attempt > MAX_RETRIES {
            return Err(last_error
                .unwrap_or_else(|| Error::Provider("provider request failed".to_string())));
        }
    }
}

// ── 请求转换 ──────────────────────────────────────────────────

/// 修复悬空工具调用对（Reasonix `SanitizeToolPairing` / `NormalizeMessages` 语义，
/// https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）。
///
/// OpenAI 兼容 API（含 DeepSeek）要求每个 assistant `tool_calls` 都必须有对应
/// 的 `role=tool` 结果消息，且不允许存在无主（孤儿）的 tool 结果。中断/恢复
/// 的历史可能携带未配对消息，DeepSeek 对这类请求直接 400。
///
/// 修复动作（只作用于发送前的临时副本，不触碰调用方持有的历史）：
/// 1. 未被任何 ToolResult 响应的 tool_call → 紧跟其 assistant 消息补一条占位
///    tool 消息（轮次保持完整，Reasonix `interruptedToolResult`）。
/// 2. 没有对应 tool_call 的孤儿 ToolResult 块 → 丢弃；整条消息只剩孤儿结果
///    时丢弃该消息。
///
/// 健康历史（每个 tool_call 都有结果、无孤儿）返回 `None`，调用方走零拷贝
/// 快路径，保持前缀缓存字节稳定（PRD §4 原则 4）。
fn sanitize_tool_pairing(messages: &[Message]) -> Option<Vec<Message>> {
    let declared: HashSet<&str> = messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|block| block.tool_id())
        .collect();
    let answered: HashSet<&str> = messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|block| block.tool_use_id())
        .collect();

    let mut needs_fix = false;
    for id in &declared {
        if !answered.contains(id) {
            needs_fix = true;
            break;
        }
    }
    if !needs_fix {
        for m in messages {
            if m.content.iter().any(|block| {
                matches!(block, ContentBlock::ToolResult { tool_use_id, .. }
                    if !declared.contains(tool_use_id.as_str()))
            }) {
                needs_fix = true;
                break;
            }
        }
    }
    if !needs_fix {
        return None;
    }

    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + declared.len());
    for m in messages {
        if m.role == Role::User && m.content.iter().any(|block| block.is_tool_result()) {
            let filtered: Vec<ContentBlock> = m
                .content
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        declared.contains(tool_use_id.as_str())
                    }
                    _ => true,
                })
                .cloned()
                .collect();
            if filtered.is_empty() {
                // 整条消息只剩孤儿结果 → 丢弃
                continue;
            }
            out.push(if filtered.len() == m.content.len() {
                m.clone()
            } else {
                Message {
                    role: m.role,
                    content: filtered,
                }
            });
        } else {
            out.push(m.clone());
        }

        // 未配对 tool_call → 紧跟 assistant 消息补占位结果（tool 消息必须紧随
        // assistant tool_calls 才能通过 API 配对校验）。
        if m.role == Role::Assistant {
            for block in &m.content {
                if let ContentBlock::ToolUse { id, .. } = block {
                    if !answered.contains(id.as_str()) {
                        out.push(Message {
                            role: Role::User,
                            content: vec![ContentBlock::ToolResult {
                                tool_use_id: id.clone(),
                                content: INTERRUPTED_TOOL_RESULT_PLACEHOLDER.to_string(),
                                is_error: true,
                            }],
                        });
                    }
                }
            }
        }
    }
    Some(out)
}

/// 键恒发（Reasonix `openai.go:688-735`，https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）：
/// - thinking 模式（DeepSeek）对 assistant `tool_calls` 轮恒发 `reasoning_content`
///   键：空串可接受，缺键 DeepSeek 400 "must be passed back"；thinking 关闭时
///   保持历史字节原样（该轮没有 reasoning 可回传，多发的键反而改变前缀）。
/// - `role=tool` 消息恒发 `name` 键：空值可接受，严格兼容口（MiMo 等）缺键
///   400。name 优先取历史 assistant ToolUse 中的真实函数名，找不到才为空串。
fn apply_key_emission(
    messages: &mut [Value],
    deepseek_thinking: bool,
    request: &CompletionRequest,
) {
    let tool_names: HashMap<&str, &str> = request
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, .. } => Some((id.as_str(), name.as_str())),
            _ => None,
        })
        .collect();

    for m in messages.iter_mut() {
        let role = m.get("role").and_then(|r| r.as_str()).map(str::to_string);
        let is_assistant_tool_turn = deepseek_thinking
            && role.as_deref() == Some("assistant")
            && m.get("tool_calls").is_some();
        if is_assistant_tool_turn && m.get("reasoning_content").is_none() {
            m["reasoning_content"] = Value::String(String::new());
        }
        if role.as_deref() == Some("tool") && m.get("name").is_none() {
            let name = m
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .and_then(|id| tool_names.get(id))
                .copied()
                .unwrap_or("");
            m["name"] = Value::String(name.to_string());
        }
    }
}

fn reasoning_text(msg: &Message) -> Option<String> {
    let reasoning = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!reasoning.is_empty()).then_some(reasoning)
}

pub fn messages_to_openai(msg: &Message, echo_reasoning: bool) -> Vec<Value> {
    // OpenAI 要求每个 tool_call 都有对应 role=tool 回复。一条内部 User 消息
    // 可以承载多个 ToolResult，因而必须展开为多条 API 消息，不能只取第一条。
    let tool_results: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => Some(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            })),
            _ => None,
        })
        .collect();
    if !tool_results.is_empty() {
        return tool_results;
    }

    // 若含 ToolUse 块 -> assistant + tool_calls
    let tool_uses: Vec<&ContentBlock> = msg.content.iter().filter(|b| b.is_tool_use()).collect();
    if !tool_uses.is_empty() {
        let tool_calls: Vec<Value> = tool_uses
            .iter()
            .map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string(),
                    }
                }),
                _ => json!({}),
            })
            .collect();
        let mut assistant = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls,
        });
        if echo_reasoning {
            if let Some(reasoning) = reasoning_text(msg) {
                assistant["reasoning_content"] = json!(reasoning);
            }
        }
        return vec![assistant];
    }

    // 多模态 user 消息必须使用 content parts；纯文本继续保留字符串形态，以兼容
    // 实现不完整的 OpenAI-compatible 网关。
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    if msg.role == Role::User
        && msg.content.iter().any(|block| {
            matches!(block, ContentBlock::Image { .. })
                || matches!(
                    block,
                    ContentBlock::File { source }
                        if source.media_type.starts_with("image/")
                )
        })
    {
        let content = msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    Some(json!({ "type": "text", "text": text }))
                }
                ContentBlock::Image { source } => Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", source.media_type, source.data),
                    }
                })),
                ContentBlock::File { source }
                    if source.media_type.starts_with("image/") && source.data.is_some() =>
                {
                    Some(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!(
                                "data:{};base64,{}",
                                source.media_type,
                                source.data.as_deref().unwrap_or_default()
                            ),
                        }
                    }))
                }
                ContentBlock::File { source } if source.kind == "text" => Some(json!({
                    "type": "text",
                    "text": attached_file_text(source),
                })),
                ContentBlock::File { source } => Some(json!({
                    "type": "text",
                    "text": format!(
                        "[Attached file: {} ({}) — this Chat Completions route cannot read the binary body]",
                        source.name, source.media_type
                    ),
                })),
                _ => None,
            })
            .collect::<Vec<_>>();
        return vec![json!({ "role": role, "content": content })];
    }
    let mut message = json!({ "role": role, "content": text_content_with_files(msg) });
    if echo_reasoning && role == "assistant" {
        if let Some(reasoning) = reasoning_text(msg) {
            message["reasoning_content"] = json!(reasoning);
        }
    }
    vec![message]
}

fn attached_file_text(source: &agent_contract::FileSource) -> String {
    format!(
        "\n\n--- Attached file: {} ({}) ---\n{}\n--- End attached file: {} ---",
        source.name,
        source.media_type,
        source.text.as_deref().unwrap_or_default(),
        source.name,
    )
}

fn text_content_with_files(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            ContentBlock::File { source } if source.kind == "text" => {
                Some(attached_file_text(source))
            }
            ContentBlock::File { source } => Some(format!(
                "\n\n[Attached file: {} ({}) — this Chat Completions route cannot read the binary body]",
                source.name, source.media_type
            )),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// 将单条内部消息转换为一个 OpenAI 消息。
///
/// 保留这个便捷入口以兼容只含文本、ToolUse 或单个 ToolResult 的调用方；构建完整
/// 请求时必须使用 [`messages_to_openai`]，以免丢失同一消息中的后续 ToolResult。
pub fn message_to_openai(msg: &Message) -> Value {
    messages_to_openai(msg, false)
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({ "role": "user", "content": "" }))
}

fn tool_to_openai(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

// ── 响应解析 ──────────────────────────────────────────────────

fn parse_openai_response(v: &Value) -> Result<CompletionResponse> {
    let choice = v
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| Error::Provider("response missing choices".into()))?;

    let mut content = Vec::new();
    if let Some(msg) = choice.get("message") {
        if let Some(text) = msg.get("content").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .pointer("/function/name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = tc
                    .pointer("/function/arguments")
                    .and_then(|t| t.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(Value::Null);
                content.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input: args,
                });
            }
        }
    }

    let stop_reason = choice
        .get("finish_reason")
        .and_then(|t| t.as_str())
        .map(parse_finish_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = v
        .get("usage")
        .map(|u| {
            let (cache_read_tokens, cache_write_tokens) = parse_cache_tokens(u);
            Usage {
                input_tokens: u.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
                output_tokens: u
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0) as u32,
                cache_read_tokens,
                cache_write_tokens,
            }
        })
        .unwrap_or_default();

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

/// 从 usage 对象解析缓存 token 数。DeepSeek 字段优先（真实 DeepSeek 响应同时
/// 回传两套字段，优先 DeepSeek 字段避免双计数或错值，见
/// https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-cache-baseline.md）：`prompt_cache_hit_tokens` →
/// cache_read_tokens，缺失时回落 OpenAI 官方风格
/// `prompt_tokens_details.cached_tokens`；`prompt_cache_miss_tokens` →
/// cache_write_tokens。字段都缺时回退 Some(0)，与"服务端未返回 usage"（整体
/// 无 usage 键）区分开（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §8）。
fn parse_cache_tokens(usage: &Value) -> (Option<u32>, Option<u32>) {
    let cache_read = usage
        .get("prompt_cache_hit_tokens")
        .and_then(|tokens| tokens.as_u64())
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(|tokens| tokens.as_u64())
        })
        .unwrap_or(0) as u32;
    let cache_write = usage
        .get("prompt_cache_miss_tokens")
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or(0) as u32;
    (Some(cache_read), Some(cache_write))
}

fn parse_finish_reason(s: &str) -> StopReason {
    match s {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
    }
}

#[derive(Default)]
struct OpenAiErrorDetails {
    message: Option<String>,
    error_type: Option<String>,
    code: Option<String>,
    param: Option<String>,
}

/// 将 OpenAI 风格的错误负载压缩为可展示文本，避免把原始 JSON 交给调用方。
///
/// Responses provider 复用同一套错误归一化与密钥脱敏，不另写一份。
pub(crate) fn map_api_error(status: u16, body: &str, api_key: &str) -> Error {
    let message = normalize_api_error(body, api_key);
    match status {
        401 | 403 => Error::AuthFailed("authentication failed (check api_key)".into()),
        429 => Error::RateLimited { retry_after: 0 },
        404 => Error::ModelNotFound(format!("model not found (status {status}): {message}")),
        _ => Error::ApiError { status, message },
    }
}

pub(crate) fn normalize_api_error(body: &str, api_key: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let details = parsed
        .as_ref()
        .map(|value| extract_openai_error_details(value, api_key))
        .unwrap_or_default();

    let message = details
        .message
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| {
            if parsed.is_some() {
                "request failed".into()
            } else {
                let fallback = sanitize_error_value(body, api_key, MAX_ERROR_MESSAGE_CHARS);
                if fallback.is_empty() {
                    "request failed".into()
                } else {
                    fallback
                }
            }
        });

    let mut metadata = Vec::new();
    if let Some(error_type) = details.error_type.filter(|value| !value.is_empty()) {
        metadata.push(format!("type={error_type}"));
    }
    if let Some(code) = details.code.filter(|value| !value.is_empty()) {
        metadata.push(format!("code={code}"));
    }
    if let Some(param) = details.param.filter(|value| !value.is_empty()) {
        metadata.push(format!("param={param}"));
    }

    if metadata.is_empty() {
        message
    } else {
        format!("{message} ({})", metadata.join(", "))
    }
}

fn extract_openai_error_details(value: &Value, api_key: &str) -> OpenAiErrorDetails {
    let error = value.get("error").unwrap_or(value);
    let field = |name| error.get(name).or_else(|| value.get(name));
    let sanitize_message = |value: &Value| {
        json_scalar_to_string(value)
            .map(|value| sanitize_error_value(&value, api_key, MAX_ERROR_MESSAGE_CHARS))
    };
    let sanitize_metadata = |value: &Value| {
        json_scalar_to_string(value)
            .map(|value| sanitize_error_value(&value, api_key, MAX_ERROR_METADATA_CHARS))
    };

    OpenAiErrorDetails {
        message: error
            .as_str()
            .map(|value| sanitize_error_value(value, api_key, MAX_ERROR_MESSAGE_CHARS))
            .or_else(|| field("message").and_then(sanitize_message)),
        error_type: field("type").and_then(sanitize_metadata),
        code: field("code").and_then(sanitize_metadata),
        param: field("param").and_then(sanitize_metadata),
    }
}

fn json_scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(_) | Value::Bool(_) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

pub(crate) fn sanitize_transport_error(message: &str, api_key: &str) -> String {
    sanitize_error_value(message, api_key, MAX_ERROR_MESSAGE_CHARS)
}

fn sanitize_error_value(value: &str, api_key: &str, max_chars: usize) -> String {
    let value = if api_key.is_empty() {
        value.to_string()
    } else {
        value.replace(api_key, "***")
    };
    truncate_chars(&redact_common_secrets(&value), max_chars)
}

fn redact_common_secrets(value: &str) -> String {
    const ASSIGNMENT_PREFIXES: [&str; 5] = [
        "api_key=",
        "api-key=",
        "apikey=",
        "access_token=",
        "authorization=",
    ];

    let mut remaining = value;
    let mut redacted = String::with_capacity(value.len());

    while !remaining.is_empty() {
        if starts_with_ascii_ignore_case(remaining, "bearer ") {
            let prefix_len = "bearer ".len();
            let token_len = secret_token_len(&remaining[prefix_len..]);
            if token_len > 0 {
                redacted.push_str(&remaining[..prefix_len]);
                redacted.push_str("***");
                remaining = &remaining[prefix_len + token_len..];
                continue;
            }
        }

        if starts_with_ascii_ignore_case(remaining, "sk-") {
            let token_len = secret_token_len(remaining);
            if token_len > "sk-".len() {
                redacted.push_str("***");
                remaining = &remaining[token_len..];
                continue;
            }
        }

        if let Some(prefix) = ASSIGNMENT_PREFIXES
            .iter()
            .find(|prefix| starts_with_ascii_ignore_case(remaining, prefix))
        {
            let token_len = secret_token_len(&remaining[prefix.len()..]);
            if token_len > 0 {
                redacted.push_str(&remaining[..prefix.len()]);
                redacted.push_str("***");
                remaining = &remaining[prefix.len() + token_len..];
                continue;
            }
        }

        let ch = remaining.chars().next().expect("remaining is not empty");
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

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

// ── SSE 流解析 ────────────────────────────────────────────────

/// SSE 字节流的中断原因。`IdleTimeout` 是流空闲 watchdog 主动掐断（可恢复，
/// 调用方可用同一冻结请求重放）；`Transport` 是底层连接瞬时错误（与流死区分）。
pub(crate) enum SseChunkError {
    Transport(reqwest::Error),
    IdleTimeout,
}

fn parse_openai_sse(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, SseChunkError>>,
) -> impl futures::Stream<Item = StreamEvent> {
    use futures::StreamExt;
    let mut buffer = String::new();
    let mut tool_args: std::collections::HashMap<u32, (String, String)> =
        std::collections::HashMap::new();
    let mut stopped = false;

    byte_stream
        .map(move |chunk| {
            let chunk = match chunk {
                Ok(c) => c,
                Err(SseChunkError::IdleTimeout) => {
                    // 流空闲 watchdog 触发：连接已关闭，流以可恢复错误终止。
                    // 与连接断开（Transport）区分：agent 层据此用同一冻结
                    // 请求重放（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）。
                    if !stopped {
                        stopped = true;
                        return vec![StreamEvent::Stop {
                            reason: StopReason::Other(STREAM_IDLE_TIMEOUT_REASON.to_string()),
                        }];
                    }
                    return Vec::new();
                }
                // 底层传输错误：保持既有行为（静默终止，不补 Stop），与
                // watchdog 掐断的显式标记区分。
                Err(SseChunkError::Transport(error)) => {
                    tracing::debug!(error = %error, "provider stream transport error");
                    return Vec::new();
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            let mut events = Vec::new();
            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..pos + 1).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    // 少数兼容接口不会发送带 finish_reason 的最后一个 choice。若仍有
                    // 未完成工具调用，在 EOF 前将它们和 Stop 一并补齐。
                    if !stopped {
                        let reason = if tool_args.is_empty() {
                            StopReason::EndTurn
                        } else {
                            StopReason::ToolUse
                        };
                        events.extend(finish_openai_turn(&mut tool_args, reason));
                        stopped = true;
                    }
                    continue;
                }
                let parsed = parse_one_openai(data, &mut tool_args);
                if parsed
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Stop { .. }))
                {
                    stopped = true;
                }
                events.extend(parsed);
            }
            events
        })
        .flat_map(futures::stream::iter)
}

/// 给 SSE 字节流加流空闲 watchdog（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E，
/// 对齐 Reasonix `readStream` 的 idle watchdog）。
///
/// 机制（无 tokio 依赖，futures 默认 features 不含 channel）：
/// - 一个 watchdog 线程以 `idle_timeout` 为周期等待"活动"信号（std mpsc
///   `recv_timeout`）；每个字节 chunk 到达时主循环发送一次信号重置计时。
/// - watchdog 超时（半开 TCP：代理切换、服务端静默，不发 RST）→ 置 stalled
///   标志并从共享 waker 槽唤醒主循环 → 主循环终止流并产出
///   `SseChunkError::IdleTimeout` 一次，由 [`parse_openai_sse`] 转成可恢复的
///   `Stop(Other("stream_idle_timeout"))`；消费方停止拉取后底层 reqwest body
///   被 drop，连接随之关闭。
/// - 正常结束（EOF）或流被 drop → watchdog 线程在信号通道断开后自行退出。
fn parse_openai_sse_with_idle_timeout(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>
        + 'static,
    idle_timeout: Duration,
) -> impl futures::Stream<Item = StreamEvent> {
    parse_openai_sse(watch_sse_idle(byte_stream, idle_timeout))
}

/// 给 SSE 字节流加空闲 watchdog：超过 `idle_timeout` 无新字节则以
/// [`SseChunkError::IdleTimeout`] 终止，供 Chat 与 Responses 两条线路复用。
pub(crate) fn watch_sse_idle(
    byte_stream: impl futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>
        + 'static,
    idle_timeout: Duration,
) -> impl futures::Stream<Item = std::result::Result<bytes::Bytes, SseChunkError>> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    let stalled = Arc::new(AtomicBool::new(false));
    // 共享 waker 槽：主循环每次 poll 更新，watchdog 超时后借此唤醒挂起的读。
    let waker_slot: Arc<Mutex<Option<std::task::Waker>>> = Arc::new(Mutex::new(None));
    let (activity_tx, activity_rx) = std::sync::mpsc::channel::<()>();
    let watchdog_stalled = stalled.clone();
    let watchdog_slot = waker_slot.clone();
    std::thread::spawn(move || loop {
        match activity_rx.recv_timeout(idle_timeout) {
            Ok(()) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                watchdog_stalled.store(true, Ordering::SeqCst);
                if let Some(waker) = watchdog_slot.lock().unwrap().take() {
                    waker.wake();
                }
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    });

    let mut inner = Box::pin(byte_stream);
    let mut terminated = false;
    let watched = futures::stream::poll_fn(move |cx| {
        use std::task::Poll;
        if terminated {
            return Poll::Ready(None);
        }
        *waker_slot.lock().unwrap() = Some(cx.waker().clone());
        if stalled.load(Ordering::SeqCst) {
            terminated = true;
            return Poll::Ready(Some(Err(SseChunkError::IdleTimeout)));
        }
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                let _ = activity_tx.send(());
                Poll::Ready(Some(item.map_err(SseChunkError::Transport)))
            }
            Poll::Ready(None) => {
                terminated = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    });
    watched
}

fn parse_one_openai(
    data: &str,
    tool_args: &mut std::collections::HashMap<u32, (String, String)>,
) -> Vec<StreamEvent> {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };
    let mut events = Vec::new();

    if let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) {
        if let Some(delta) = choice.get("delta") {
            // DeepSeek/Qwen compatible routes commonly use `reasoning_content`; OpenRouter and
            // a few gateways normalize the same public stream as `reasoning`. Take the first
            // populated field so a gateway that aliases both cannot duplicate the UI output.
            if let Some(reasoning) = ["reasoning_content", "reasoning"]
                .iter()
                .find_map(|key| delta.get(*key).and_then(Value::as_str))
                .filter(|text| !text.is_empty())
            {
                events.push(StreamEvent::ReasoningDelta {
                    text: reasoning.to_string(),
                });
            }
            if let Some(content) = delta.get("content").and_then(|text| text.as_str()) {
                if !content.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: content.to_string(),
                    });
                }
            }

            if let Some(tool_calls) = delta.get("tool_calls").and_then(|calls| calls.as_array()) {
                for tool_call in tool_calls {
                    let index = tool_call
                        .get("index")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as u32;
                    let entry = tool_args
                        .entry(index)
                        .or_insert_with(|| (String::new(), String::new()));
                    if let Some(id) = tool_call.get("id").and_then(|value| value.as_str()) {
                        entry.0 = id.to_string();
                    }
                    if let Some(name) = tool_call
                        .pointer("/function/name")
                        .and_then(|value| value.as_str())
                    {
                        if !name.is_empty() {
                            events.push(StreamEvent::ToolUseStart {
                                id: entry.0.clone(),
                                name: name.to_string(),
                            });
                        }
                    }
                    if let Some(arguments) = tool_call
                        .pointer("/function/arguments")
                        .and_then(|value| value.as_str())
                    {
                        entry.1.push_str(arguments);
                        events.push(StreamEvent::ToolUseDelta {
                            id: entry.0.clone(),
                            input_json: arguments.to_string(),
                        });
                    }
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(|value| value.as_str()) {
            events.extend(finish_openai_turn(tool_args, parse_finish_reason(reason)));
        }
    }

    if let Some(usage) = value.get("usage") {
        // DeepSeek 在 usage 帧中回传自动前缀缓存的命中/未命中 token 数
        // （https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §3 A6，P0-B）。字段缺失时回退
        // Some(0)，与"服务端未返回 usage 帧"（整体无 usage 键，无 Usage
        // 事件）区分开：前者是已返回帧中的零命中，后者是服务端不支持。
        let (cache_read_tokens, cache_write_tokens) = parse_cache_tokens(usage);
        events.push(StreamEvent::Usage(Usage {
            input_tokens: usage
                .get("prompt_tokens")
                .and_then(|tokens| tokens.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: usage
                .get("completion_tokens")
                .and_then(|tokens| tokens.as_u64())
                .unwrap_or(0) as u32,
            cache_read_tokens,
            cache_write_tokens,
        }));
    }

    events
}

/// 将一个结束帧归一化为“所有工具完成，再停止”。OpenAI Chat Completions 在
/// `finish_reason=tool_calls` 时可以在同一个 SSE 帧中同时表达全部工具调用；
/// 不能只 drain 第一个，否则后续请求会缺少成对的 `role=tool` 消息。
fn finish_openai_turn(
    tool_args: &mut std::collections::HashMap<u32, (String, String)>,
    reason: StopReason,
) -> Vec<StreamEvent> {
    let mut tool_calls: Vec<(u32, (String, String))> = tool_args.drain().collect();
    tool_calls.sort_by_key(|(index, _)| *index);

    let mut events: Vec<StreamEvent> = tool_calls
        .into_iter()
        .map(|(_, (id, arguments))| StreamEvent::ToolUseComplete {
            id,
            input: serde_json::from_str(&arguments).unwrap_or(Value::Null),
        })
        .collect();
    events.push(StreamEvent::Stop { reason });
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_url_accepts_root_or_v1_base_url() {
        let from_root = OpenAiProvider::new(
            "sk-test".into(),
            "test-model".into(),
            "https://api.example.com".into(),
        );
        let from_v1 = OpenAiProvider::new(
            "sk-test".into(),
            "test-model".into(),
            "https://api.example.com/v1/".into(),
        );
        let deepseek = OpenAiProvider::new(
            "sk-test".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );

        assert_eq!(
            from_root.completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            from_v1.completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            deepseek.completions_url(),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[test]
    fn completions_url_keeps_vendor_version_segments() {
        // 智谱 Coding Plan 用 /v4，火山方舟套餐用 /api/coding/v3，
        // 百炼用 /compatible-mode/v1。旧的"结尾不是 /v1 就补 /v1"会把前两个
        // 拼成 .../v4/v1/chat/completions → 404。
        for base in [
            "https://open.bigmodel.cn/api/coding/paas/v4",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ] {
            let provider = OpenAiProvider::new("k".into(), "m".into(), base.into());
            assert_eq!(
                provider.completions_url(),
                format!("{base}/chat/completions")
            );
        }
    }

    #[test]
    fn system_role_injected() {
        let p = OpenAiProvider::new("k".into(), "gpt".into(), "https://api.openai.com".into());
        let req = CompletionRequest {
            model: "gpt".into(),
            system: Some("be nice".into()),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be nice");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(body["max_tokens"], 16);
    }

    #[test]
    fn openrouter_hosted_web_tools_are_sent_as_server_tools() {
        let provider = OpenAiProvider::new(
            "key".into(),
            "anthropic/claude-sonnet-5".into(),
            "https://openrouter.ai/api/v1".into(),
        );
        let request = CompletionRequest {
            model: "anthropic/claude-sonnet-5".into(),
            system: None,
            messages: vec![Message::user_text("research this")],
            tools: vec![],
            hosted_tools: vec![
                HostedToolSpec::web_search_with_format(HostedToolFormat::OpenRouter),
                HostedToolSpec::web_fetch_with_format(HostedToolFormat::OpenRouter),
            ],
            max_tokens: 128,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["tools"][0]["type"], "openrouter:web_search");
        assert_eq!(body["tools"][0]["parameters"]["max_uses"], 5);
        assert_eq!(body["tools"][1]["type"], "openrouter:web_fetch");
        assert_eq!(body["tools"][1]["parameters"]["max_uses"], 5);
    }

    #[test]
    fn deepseek_inference_options_use_thinking_and_omit_temperature() {
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let req = CompletionRequest {
            model: "deepseek-v4-pro".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: Some(0.7),
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: Some("enabled".into()),
                reasoning_effort: Some("high".into()),
                verbosity: None,
            },
        };

        let body = p.build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn deepseek_chat_maps_local_adaptive_marker_to_native_enabled() {
        let provider = OpenAiProvider::new(
            "k".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let request = CompletionRequest {
            model: "deepseek-v4-pro".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: Some(0.7),
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: Some("adaptive".into()),
                ..Default::default()
            },
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(!body.to_string().contains("adaptive"));
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn deepseek_chat_preserves_explicit_disabled_high_and_max_modes() {
        let provider = OpenAiProvider::new(
            "k".into(),
            "deepseek-v4-pro".into(),
            "https://api.deepseek.com".into(),
        );
        let mut request = CompletionRequest {
            model: "deepseek-v4-pro".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };

        request.inference.thinking = Some("disabled".into());
        request.inference.reasoning_effort = Some("max".into());
        let disabled = provider.build_body(&request, false);
        assert_eq!(disabled["thinking"]["type"], "disabled");
        assert!(disabled.get("reasoning_effort").is_none());

        request.inference.thinking = Some("enabled".into());
        for effort in ["high", "max"] {
            request.inference.reasoning_effort = Some(effort.into());
            let body = provider.build_body(&request, false);
            assert_eq!(body["thinking"]["type"], "enabled");
            assert_eq!(body["reasoning_effort"], effort);
        }
    }

    #[test]
    fn non_deepseek_chat_keeps_provider_native_adaptive_semantics() {
        let provider = OpenAiProvider::new(
            "k".into(),
            "vendor-model".into(),
            "https://gateway.example/v1".into(),
        );
        let request = CompletionRequest {
            model: "vendor-model".into(),
            system: None,
            messages: vec![],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: Some("adaptive".into()),
                ..Default::default()
            },
        };

        let body = provider.build_body(&request, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
    }

    #[test]
    fn qwen_thinking_uses_compatible_enable_thinking_flag() {
        let p = OpenAiProvider::new(
            "k".into(),
            "qwen3-max".into(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1".into(),
        );
        let req = CompletionRequest {
            model: "qwen3-max".into(),
            system: None,
            messages: vec![],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: Some("enabled".into()),
                ..Default::default()
            },
        };

        let body = p.build_body(&req, false);
        assert_eq!(body["enable_thinking"], true);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn stream_usage_is_requested_only_for_official_openai_endpoint() {
        let req = CompletionRequest {
            model: "gpt".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        };
        let official = OpenAiProvider::new(
            "sk-test".into(),
            "gpt".into(),
            "https://api.openai.com/v1".into(),
        );
        let deepseek = OpenAiProvider::new(
            "key".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let compatible = OpenAiProvider::new(
            "key".into(),
            "model".into(),
            "https://openrouter.ai/api/v1".into(),
        );
        let deepseek_gateway = OpenAiProvider::new(
            "key".into(),
            "deepseek-v4-pro".into(),
            "https://relay.internal.example/v1".into(),
        )
        .with_stream_usage();

        let official_body = official.build_body(&req, true);
        let deepseek_body = deepseek.build_body(&req, true);
        let compatible_body = compatible.build_body(&req, true);
        let deepseek_gateway_body = deepseek_gateway.build_body(&req, true);

        assert_eq!(official_body["stream"], true);
        assert_eq!(official_body["stream_options"]["include_usage"], true);
        // DeepSeek 前缀缓存字段只出现在 usage 帧中，必须请求 include_usage
        // （https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §3 A14，P0-B 前置）。
        assert_eq!(deepseek_body["stream"], true);
        assert_eq!(deepseek_body["stream_options"]["include_usage"], true);
        assert_eq!(
            deepseek_gateway_body["stream_options"]["include_usage"],
            true
        );
        assert_eq!(compatible_body["stream"], true);
        assert!(compatible_body.get("stream_options").is_none());
    }

    #[test]
    fn usage_frame_parses_deepseek_cache_tokens() {
        let mut tool_args = std::collections::HashMap::new();
        let events = parse_one_openai(
            r#"{"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_cache_hit_tokens":123,"prompt_cache_miss_tokens":7}}"#,
            &mut tool_args,
        );
        let usage = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("usage event");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_tokens, Some(123));
        assert_eq!(usage.cache_write_tokens, Some(7));
    }

    #[test]
    fn usage_frame_defaults_missing_cache_fields_to_zero() {
        let mut tool_args = std::collections::HashMap::new();
        let events = parse_one_openai(
            r#"{"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            &mut tool_args,
        );
        let usage = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("usage event");
        assert_eq!(usage.cache_read_tokens, Some(0));
        assert_eq!(usage.cache_write_tokens, Some(0));
    }

    #[test]
    fn usage_frame_falls_back_to_openai_cached_tokens() {
        let mut tool_args = std::collections::HashMap::new();
        let events = parse_one_openai(
            r#"{"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":42}}}"#,
            &mut tool_args,
        );
        let usage = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("usage event");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_tokens, Some(42));
        assert_eq!(usage.cache_write_tokens, Some(0));
    }

    #[test]
    fn response_parses_deepseek_cache_tokens() {
        let response = parse_openai_response(&json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 88,
                "prompt_cache_miss_tokens": 12
            }
        }))
        .expect("parse response");
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.cache_read_tokens, Some(88));
        assert_eq!(response.usage.cache_write_tokens, Some(12));
    }

    #[test]
    fn response_falls_back_to_openai_cached_tokens() {
        let response = parse_openai_response(&json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 64}
            }
        }))
        .expect("parse response");
        assert_eq!(response.usage.cache_read_tokens, Some(64));
        assert_eq!(response.usage.cache_write_tokens, Some(0));
    }

    #[test]
    fn deepseek_cache_tokens_take_precedence_over_openai_details() {
        // 真实 DeepSeek 响应同时回传两套字段（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-cache-baseline.md
        // 第 19 行），DeepSeek 字段优先，避免双计数或错值。
        let response = parse_openai_response(&json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 88,
                "prompt_cache_miss_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 999}
            }
        }))
        .expect("parse response");
        assert_eq!(response.usage.cache_read_tokens, Some(88));
        assert_eq!(response.usage.cache_write_tokens, Some(12));
    }

    #[test]
    fn tool_use_becomes_tool_calls() {
        let m = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "read".into(),
                input: json!({"x": 1}),
            }],
        };
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["tool_calls"][0]["function"]["name"], "read");
    }

    #[test]
    fn tool_result_becomes_tool_role() {
        let m = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "ok".into(),
                is_error: false,
            }],
        };
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "t1");
    }

    #[test]
    fn multiple_tool_results_expand_to_multiple_tool_messages() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "first".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t2".into(),
                    content: "second".into(),
                    is_error: false,
                },
            ],
        };

        let messages = messages_to_openai(&message, false);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "t1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "t2");
    }

    #[test]
    fn user_image_is_emitted_as_data_url_content_part() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "look".into(),
                },
                ContentBlock::Image {
                    source: agent_contract::ImageSource {
                        kind: "base64".into(),
                        media_type: "image/png".into(),
                        data: "iVBORw0KGgo=".into(),
                    },
                },
            ],
        };
        let messages = messages_to_openai(&message, false);
        assert_eq!(messages[0]["content"][0]["type"], "text");
        assert_eq!(messages[0]["content"][1]["type"], "image_url");
        assert_eq!(
            messages[0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
    }

    #[test]
    fn text_attachment_is_flattened_for_chat_compatible_routes() {
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::File {
                source: agent_contract::FileSource {
                    kind: "text".into(),
                    name: "main.rs".into(),
                    media_type: "text/x-rust".into(),
                    text: Some("fn main() {}".into()),
                    data: None,
                },
            }],
        };
        let messages = messages_to_openai(&message, false);
        assert_eq!(messages[0]["role"], "user");
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("Attached file: main.rs"));
    }

    #[test]
    fn tool_calls_finish_with_all_completions_and_stop() {
        let mut tool_args = std::collections::HashMap::from([
            (0, ("call-a".to_string(), r#"{"a":1}"#.to_string())),
            (1, ("call-b".to_string(), r#"{"b":2}"#.to_string())),
        ]);

        let events = parse_one_openai(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            &mut tool_args,
        );

        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::ToolUseComplete { id: first, .. },
                StreamEvent::ToolUseComplete { id: second, .. },
                StreamEvent::Stop {
                    reason: StopReason::ToolUse
                }
            ] if first == "call-a" && second == "call-b"
        ));
        assert!(tool_args.is_empty());
    }

    #[test]
    fn parse_response_text() {
        let v = json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1}
        });
        let r = parse_openai_response(&v).unwrap();
        assert_eq!(r.text(), "hi");
        assert_eq!(r.usage.input_tokens, 2);
    }

    #[test]
    fn stream_surfaces_provider_reasoning_separately_from_answer_text() {
        let mut tool_args = std::collections::HashMap::new();
        let events = parse_one_openai(
            r#"{"choices":[{"delta":{"reasoning_content":"checking","content":"done"}}]}"#,
            &mut tool_args,
        );
        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::ReasoningDelta { text: reasoning },
                StreamEvent::TextDelta { text: answer }
            ] if reasoning == "checking" && answer == "done"
        ));
    }

    #[test]
    fn api_error_sanitizes_key() {
        let e = map_api_error(401, "Bearer sk-SECRET", "sk-SECRET");
        let msg = e.to_string();
        assert!(!msg.contains("sk-SECRET"));
        assert!(matches!(e, Error::AuthFailed(_)));
    }

    #[test]
    fn api_error_extracts_service_message_from_json() {
        let e = map_api_error(
            400,
            r#"{"error":{"message":"Invalid max_tokens value","type":"invalid_request_error","code":"invalid_value","param":"max_tokens"}}"#,
            "sk-test",
        );
        let Error::ApiError { status, message } = e else {
            panic!("expected API error");
        };
        assert_eq!(status, 400);
        assert_eq!(
            message,
            "Invalid max_tokens value (type=invalid_request_error, code=invalid_value, param=max_tokens)"
        );
    }

    #[test]
    fn api_error_redacts_secrets_and_does_not_dump_unknown_json() {
        let e = map_api_error(
            400,
            r#"{"error":{"message":"invalid authorization: Bearer custom-secret sk-SECRET","type":"invalid_request_error","param":"api_key"}}"#,
            "custom-secret",
        );
        let msg = e.to_string();
        assert!(!msg.contains("custom-secret"));
        assert!(!msg.contains("sk-SECRET"));
        assert!(msg.contains("Bearer ***"));

        let e = map_api_error(
            500,
            r#"{"error":{"unexpected":"request payload should not be exposed"}}"#,
            "",
        );
        assert_eq!(e.to_string(), "API error: 500 - request failed");
    }

    // ── P1-F 键恒发（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）────

    fn assistant_tool_call_message(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: json!({"x": 1}),
            }],
        }
    }

    fn tool_result_message(id: &str, content: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: content.into(),
                is_error: false,
            }],
        }
    }

    fn tool_round_request(model: &str, messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: model.into(),
            system: None,
            messages,
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 16,
            temperature: None,
            enable_caching: false,
            inference: agent_contract::InferenceOptions {
                thinking: Some("enabled".into()),
                ..Default::default()
            },
        }
    }

    fn long_tool_evidence() -> String {
        let mut evidence =
            "path=src/核心.rs; command=cargo test; exit=0; 证据=完整\n".repeat(4_096);
        evidence.push_str("__TOOL_EVIDENCE_TAIL_CHAT__");
        assert!(evidence.len() > 100_000);
        evidence
    }

    #[test]
    fn deepseek_thinking_emits_reasoning_content_key_on_tool_call_turns() {
        // thinking 模式对 assistant tool_calls 轮恒发 reasoning_content 键
        // （空串可接受，缺键 DeepSeek 400 "must be passed back"）。
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let req = tool_round_request(
            "deepseek-chat",
            vec![
                assistant_tool_call_message("t1", "read"),
                tool_result_message("t1", "ok"),
            ],
        );
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        let assistant = &msgs[0];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["tool_calls"][0]["function"]["name"], "read");
        assert!(
            assistant.get("reasoning_content").is_some(),
            "assistant tool_calls turn must always carry reasoning_content key"
        );
        assert_eq!(assistant["reasoning_content"], "");
    }

    #[test]
    fn tool_messages_always_carry_name_key() {
        // tool 消息恒发 name 键，值优先取历史 assistant ToolUse 的函数名。
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let req = tool_round_request(
            "deepseek-chat",
            vec![
                assistant_tool_call_message("t1", "read"),
                tool_result_message("t1", "ok"),
            ],
        );
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        let tool = &msgs[1];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "t1");
        assert_eq!(
            tool["name"], "read",
            "tool message name should be backfilled from the assistant turn"
        );
    }

    #[test]
    fn non_thinking_deepseek_keeps_history_bytes_unchanged() {
        // thinking 关闭（或非 deepseek 模型）时不给 assistant tool_calls 轮
        // 添加 reasoning_content 键——保持既有字节，避免改变前缀缓存形状。
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let mut req = tool_round_request(
            "deepseek-chat",
            vec![
                assistant_tool_call_message("t1", "read"),
                tool_result_message("t1", "ok"),
            ],
        );
        req.inference.thinking = None;
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert!(msgs[0].get("reasoning_content").is_none());
        // tool 消息的 name 键仍然恒发（对所有 provider）
        assert!(msgs[1].get("name").is_some());

        let generic =
            OpenAiProvider::new("k".into(), "gpt".into(), "https://api.openai.com".into());
        let body = generic.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert!(msgs[0].get("reasoning_content").is_none());
        assert!(msgs[1].get("name").is_some());
    }

    // ── P1-F SanitizeToolPairing（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-F）

    #[test]
    fn dangling_tool_call_gets_placeholder_result() {
        // assistant 声明 t1、t2 两个调用，但只有 t1 有结果：t2 必须补占位
        // tool 消息，否则 DeepSeek 400 "must be followed by tool messages"。
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let assistant = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "read".into(),
                    input: json!({"x": 1}),
                },
                ContentBlock::ToolUse {
                    id: "t2".into(),
                    name: "write".into(),
                    input: json!({"y": 2}),
                },
            ],
        };
        let req = tool_round_request(
            "deepseek-chat",
            vec![assistant, tool_result_message("t1", "ok")],
        );
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "assistant + placeholder + t1 result");
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 2);
        // 占位 tool 消息紧跟 assistant（配对校验要求 tool 消息紧随其后）
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "t2");
        assert!(msgs[1]["content"].as_str().unwrap().contains("no result"));
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "t1");
    }

    #[test]
    fn orphan_tool_result_is_dropped() {
        // t2 的结果没有对应 tool_call → 孤儿，发送前丢弃。
        let p = OpenAiProvider::new(
            "k".into(),
            "deepseek-chat".into(),
            "https://api.deepseek.com".into(),
        );
        let orphan_results = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "ghost".into(),
                    content: "orphan".into(),
                    is_error: false,
                },
            ],
        };
        let req = tool_round_request(
            "deepseek-chat",
            vec![assistant_tool_call_message("t1", "read"), orphan_results],
        );
        let body = p.build_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "assistant + one tool result");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "t1");
    }

    #[test]
    fn healthy_history_needs_no_pairing_repair() {
        let messages = vec![
            assistant_tool_call_message("t1", "read"),
            tool_result_message("t1", "ok"),
        ];
        assert!(sanitize_tool_pairing(&messages).is_none());
    }

    #[test]
    fn custom_deepseek_gateway_preserves_paired_long_tool_evidence_and_drops_only_orphans() {
        let provider = OpenAiProvider::new(
            "k".into(),
            "deepseek-v4-pro".into(),
            "https://gateway.example/deepseek/v1".into(),
        )
        .with_stream_usage();
        let evidence = long_tool_evidence();
        let request = tool_round_request(
            "deepseek-v4-pro",
            vec![
                assistant_tool_call_message("call_read", "read_file"),
                tool_result_message("call_read", &evidence),
                tool_result_message("call_orphan", "must not reach the provider"),
            ],
        );

        let body = provider.build_body(&request, true);
        let messages = body["messages"].as_array().expect("chat messages");

        assert_eq!(messages.len(), 2, "assistant call + its paired result");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_read");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_read");
        assert_eq!(messages[1]["name"], "read_file");
        assert_eq!(messages[1]["content"].as_str(), Some(evidence.as_str()));
        assert!(messages[1]["content"]
            .as_str()
            .is_some_and(|content| content.ends_with("__TOOL_EVIDENCE_TAIL_CHAT__")));
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(!body.to_string().contains("call_orphan"));
    }

    // ── P1-E 退避计算（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）──

    #[test]
    fn backoff_delay_doubles_then_caps_and_honors_retry_after() {
        assert_eq!(backoff_delay(1, None), Duration::from_millis(500));
        assert_eq!(backoff_delay(2, None), Duration::from_millis(1000));
        assert_eq!(backoff_delay(3, None), Duration::from_millis(2000));
        // 500ms * 2^5 = 16s > 15s 封顶
        assert_eq!(backoff_delay(6, None), MAX_BACKOFF);
        // Retry-After 优先，且封顶 60s
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(120))),
            MAX_RETRY_AFTER
        );
    }

    // ── P1-E 流空闲 watchdog（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）

    #[tokio::test]
    async fn idle_stream_timeout_terminates_with_recoverable_stop() {
        use futures::StreamExt;
        let first: std::result::Result<bytes::Bytes, reqwest::Error> =
            Ok(bytes::Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            ));
        let stalled =
            futures::stream::pending::<std::result::Result<bytes::Bytes, reqwest::Error>>();
        let stream = futures::stream::once(async { first }).chain(stalled);

        let mut events = parse_openai_sse_with_idle_timeout(stream, Duration::from_millis(100));
        let mut seen = Vec::new();
        while let Some(ev) = events.next().await {
            seen.push(ev);
        }
        assert_eq!(seen.len(), 2, "delta + idle-timeout stop");
        assert!(matches!(&seen[0], StreamEvent::TextDelta { text } if text == "hi"));
        match &seen[1] {
            StreamEvent::Stop {
                reason: StopReason::Other(msg),
            } => assert!(
                msg.contains(STREAM_IDLE_TIMEOUT_REASON),
                "recoverable idle-timeout marker expected, got {msg}"
            ),
            other => panic!("expected idle-timeout stop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn active_stream_finishes_before_watchdog_fires() {
        use futures::StreamExt;
        let data = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        let stream = futures::stream::once(async {
            Ok::<_, reqwest::Error>(bytes::Bytes::from_static(data))
        });

        let mut events = parse_openai_sse_with_idle_timeout(stream, Duration::from_secs(30));
        let mut seen = Vec::new();
        while let Some(ev) = events.next().await {
            seen.push(ev);
        }
        assert!(matches!(
            seen.as_slice(),
            [
                StreamEvent::TextDelta { .. },
                StreamEvent::Stop {
                    reason: StopReason::EndTurn
                }
            ]
        ));
    }

    // ── P1-E 连接层重试（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §5 P1-E）

    /// 读取一个 HTTP 请求（header + Content-Length body），返回 body 文本。
    async fn read_http_request_body(socket: &mut tokio::net::TcpStream) -> String {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 2048];
        loop {
            let n = socket.read(&mut tmp).await.expect("read headers");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let header_end = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("header terminator")
            + 4;
        let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let lower = line.to_ascii_lowercase();
                lower
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
            })
            .unwrap_or(0);
        while buf.len() < header_end + content_length {
            let n = socket.read(&mut tmp).await.expect("read body");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }
        String::from_utf8_lossy(&buf[header_end..header_end + content_length]).to_string()
    }

    async fn write_http_response(
        socket: &mut tokio::net::TcpStream,
        status: u16,
        reason: &str,
        body: &str,
    ) {
        use tokio::io::AsyncWriteExt;
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    }

    #[tokio::test]
    async fn transient_failure_is_retried_with_identical_body() {
        use std::sync::{Arc, Mutex};

        let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let bodies_for_server = bodies.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_http_request_body(&mut socket).await;
                bodies_for_server.lock().unwrap().push(request);
                let is_first = bodies_for_server.lock().unwrap().len() == 1;
                if is_first {
                    write_http_response(
                        &mut socket,
                        500,
                        "Internal Server Error",
                        r#"{"error":{"message":"boom"}}"#,
                    )
                    .await;
                } else {
                    write_http_response(
                        &mut socket,
                        200,
                        "OK",
                        r#"{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
                    )
                    .await;
                }
            }
        });

        let p = OpenAiProvider::new("k".into(), "m".into(), format!("http://{addr}"));
        let req = tool_round_request("m", vec![Message::user_text("hello")]);
        let resp = p.complete(req).await.expect("retry should succeed");
        assert_eq!(resp.text(), "hi");

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 2, "first 5xx must be retried");
        assert_eq!(
            recorded[0], recorded[1],
            "retry must reuse identical request body bytes (prefix cache)"
        );
        server.abort();
    }

    #[tokio::test]
    async fn four_hundred_is_not_retried() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let hits = Arc::new(AtomicUsize::new(0));
        let hits_for_server = hits.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                hits_for_server.fetch_add(1, Ordering::SeqCst);
                let _ = read_http_request_body(&mut socket).await;
                write_http_response(
                    &mut socket,
                    400,
                    "Bad Request",
                    r#"{"error":{"message":"bad request"}}"#,
                )
                .await;
            }
        });

        let p = OpenAiProvider::new("k".into(), "m".into(), format!("http://{addr}"));
        let req = tool_round_request("m", vec![Message::user_text("hello")]);
        let err = p.complete(req).await.expect_err("400 must fail");
        match &err {
            Error::ApiError { status: 400, .. } => {}
            other => panic!("expected ApiError 400, got {other}"),
        }
        assert_eq!(hits.load(Ordering::SeqCst), 1, "4xx must not be retried");
        server.abort();
    }

    #[test]
    fn kimi_chat_dialect_omits_temperature_and_echoes_reasoning_content() {
        use crate::dialect::{dialect_for, DialectPort};
        let provider = OpenAiProvider::new(
            "k".into(),
            "k3-256k".into(),
            "https://api.kimi.com/coding/v1".into(),
        )
        .with_dialect(
            dialect_for("kimi_coding", "k3-256k", DialectPort::OpenAiChat)
                .expect("kimi chat dialect"),
        );
        let mut request = tool_round_request("k3-256k", vec![Message::user_text("hi")]);
        request.temperature = Some(0.1);
        request.inference.thinking = Some("enabled".into());
        request.inference.reasoning_effort = Some("high".into());
        let body = provider.build_body(&request, false);
        assert!(
            body.get("temperature").is_none(),
            "kimi must never send temperature"
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");

        let assistant = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "reasoning".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
            ],
        };
        let messages = messages_to_openai(&assistant, true);
        assert_eq!(messages[0]["reasoning_content"], "reasoning");
    }

    #[test]
    fn ark_chat_dialect_passthrough_and_forces_stream_usage() {
        use crate::dialect::{dialect_for, DialectPort};
        let provider = OpenAiProvider::new(
            "k".into(),
            "ark-code-latest".into(),
            "https://ark.cn-beijing.volces.com/api/coding/v3".into(),
        )
        .with_dialect(
            dialect_for(
                "ark_coding_openai",
                "ark-code-latest",
                DialectPort::OpenAiChat,
            )
            .expect("ark chat dialect"),
        );
        let mut request = tool_round_request("ark-code-latest", vec![Message::user_text("hi")]);
        request.temperature = Some(0.3);
        request.inference.thinking = Some("adaptive".into());
        request.inference.reasoning_effort = Some("high".into());
        let body = provider.build_body(&request, true);
        assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }
}
