//! DeepSeek 真实 API 前缀缓存探针（https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-prefix-cache.md §6 真实 API 层级）。
//!
//! 验证 P0-A 请求结构（稳定 system + append-only 历史）在真实 DeepSeek API 上的
//! 缓存命中率曲线，并验证 P0-B 的 usage 解析链路（`stream_options.include_usage`
//! → `prompt_cache_hit_tokens` → `cache_read_tokens`）。
//!
//! 用法（key 从环境变量读取，不落盘）：
//! ```text
//! DEEPSEEK_API_KEY=sk-... cargo test -p agent-llm --test deepseek_cache_probe -- --ignored --nocapture
//! ```
//!
//! 预期：第 1 轮冷启动全 miss（hit=0）；第 2 轮起前缀命中，命中率随轮次增长
//! 趋近 90%+（对照 https://github.com/foritin/r-code/blob/main/docs/archive/deepseek-cache-baseline.md 的 80.5% 第二轮基线）。

use std::sync::Arc;
use std::time::Duration;

use agent_contract::{
    CompletionRequest, ContentBlock, LlmProvider, Message, Role, StopReason, StreamEvent, Usage,
};
use agent_llm::deepseek::DeepSeekProvider;
use agent_llm::{create_provider, ProviderConfig};
use futures::StreamExt;

/// 模拟 P0-A 落地后的稳定 system（不含时间戳等动态内容，字节跨轮不变）。
/// 长度约 250 tokens，明显超过 DeepSeek 缓存块粒度（~64 tokens）。
const STABLE_SYSTEM: &str = "You are a senior software architect assistant embedded in an IDE. \
You help with code architecture, refactoring, debugging, performance optimization, and best \
practices. You always ground answers in concrete code examples. When asked about caching you \
explain byte-level prefix caching in detail, including cache block granularity, billing, and how \
clients keep prefixes stable. When asked about concurrency you discuss locks, atomics, and async \
models. When asked about databases you compare SQLite and Postgres tradeoffs. You keep answers \
structured with headings and bullet points. You never invent APIs that do not exist. You admit \
uncertainty when you are not sure. You prefer showing a minimal reproducible example over abstract \
advice. You respond in the language the user uses.";

const ROUNDS: usize = 14;
const PROTOCOL_PROBE_ROUNDS: usize = 3;
const PROTOCOL_PROBE_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "需要真实 DeepSeek API key（DEEPSEEK_API_KEY 环境变量）"]
async fn deepseek_prefix_cache_hit_curve() {
    let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") else {
        eprintln!("[probe] DEEPSEEK_API_KEY 未设置，跳过（真实 API 探针）");
        return;
    };
    let provider = DeepSeekProvider::new(api_key, "deepseek-chat".into());

    // P0-A 请求结构：稳定 system + append-only 历史（每轮追加 assistant 回复 + 新 user 消息）。
    let mut messages = vec![Message::user_text(
        "请用一段话解释什么是字节级前缀缓存，以及它如何影响 API 成本。",
    )];
    let mut curve: Vec<(u32, u32, f32)> = Vec::new();

    for round in 0..ROUNDS {
        let request = Arc::new(CompletionRequest {
            model: "deepseek-chat".into(),
            system: Some(STABLE_SYSTEM.to_string()),
            messages: messages.clone(),
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 256,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        });

        let mut stream = provider
            .stream(request)
            .await
            .expect("DeepSeek 流式请求必须成功");
        let mut reply = String::new();
        let mut usage = Usage::default();
        let mut stop = StopReason::EndTurn;
        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::TextDelta { text } => reply.push_str(&text),
                StreamEvent::Usage(u) => usage = u,
                StreamEvent::Stop { reason } => stop = reason,
                _ => {}
            }
        }

        let hit = usage.cache_read_tokens.unwrap_or(0);
        let miss = usage.cache_write_tokens.unwrap_or(0);
        let rate = if hit + miss > 0 {
            hit as f32 * 100.0 / (hit + miss) as f32
        } else {
            0.0
        };
        curve.push((hit, miss, rate));
        eprintln!(
            "[probe] round {round}: prompt={} hit={hit} miss={miss} rate={rate:.1}% stop={stop:?}",
            usage.input_tokens
        );

        // append-only：追加 assistant 回复 + 下一轮 user 消息（与 agent 循环一致）。
        if !reply.is_empty() {
            messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: reply }],
            });
        }
        messages.push(Message::user_text(format!(
            "继续：这是第 {} 轮追问，请用一段话简短回答。",
            round + 1
        )));
    }

    // 尾 3 轮平均命中率（对齐守卫测试 tail_avg 语义）。
    let tail: Vec<f32> = curve[curve.len().saturating_sub(3)..]
        .iter()
        .map(|(_, _, rate)| *rate)
        .collect();
    let tail_avg = tail.iter().sum::<f32>() / tail.len() as f32;
    eprintln!("[probe] tail_avg(3) = {tail_avg:.1}% （对照 baseline 第二轮 80.5%、守卫阈值 90%）");
    // 探针不硬断言（真实网络/限流可能影响单次运行），只输出曲线供人工对照。
    assert!(
        tail_avg > 0.0,
        "真实 API 未观测到任何缓存命中——前缀可能不稳定"
    );
}

fn live_api_key() -> Option<String> {
    match std::env::var("DEEPSEEK_API_KEY") {
        Ok(api_key) if !api_key.trim().is_empty() => Some(api_key),
        _ => {
            eprintln!("[live-probe] DEEPSEEK_API_KEY 未设置，跳过真实 API 探针");
            None
        }
    }
}

fn protocol_probe_request(model: &str) -> Arc<CompletionRequest> {
    Arc::new(CompletionRequest {
        model: model.to_string(),
        system: Some(STABLE_SYSTEM.to_string()),
        messages: vec![Message::user_text(
            "This is a transport and prefix-cache probe. Reply with exactly: cache-probe-ok",
        )],
        tools: vec![],
        hosted_tools: vec![],
        max_tokens: 64,
        temperature: None,
        enable_caching: true,
        inference: Default::default(),
    })
}

fn merge_usage(aggregate: &mut Usage, update: Usage) {
    if update.input_tokens > 0 {
        aggregate.input_tokens = update.input_tokens;
    }
    if update.output_tokens > 0 {
        aggregate.output_tokens = update.output_tokens;
    }
    if update.cache_read_tokens.is_some() {
        aggregate.cache_read_tokens = update.cache_read_tokens;
    }
    if update.cache_write_tokens.is_some() {
        aggregate.cache_write_tokens = update.cache_write_tokens;
    }
}

async fn run_protocol_probe(label: &str, provider: &dyn LlmProvider, model: &str) {
    let mut saw_cache_metrics = false;
    let mut saw_cache_hit = false;

    for round in 1..=PROTOCOL_PROBE_ROUNDS {
        let request = protocol_probe_request(model);
        let (reply, usage) = tokio::time::timeout(PROTOCOL_PROBE_TIMEOUT, async {
            let mut stream = provider.stream(request).await?;
            let mut reply = String::new();
            let mut usage = Usage::default();
            while let Some(event) = stream.next().await {
                match event {
                    StreamEvent::TextDelta { text } => reply.push_str(&text),
                    StreamEvent::Usage(update) => merge_usage(&mut usage, update),
                    _ => {}
                }
            }
            agent_error::Result::Ok((reply, usage))
        })
        .await
        .unwrap_or_else(|_| panic!("{label} 请求在 120 秒内未完成"))
        .unwrap_or_else(|error| panic!("{label} 请求失败: {error}"));

        assert!(!reply.trim().is_empty(), "{label} 返回了空响应");
        saw_cache_metrics |=
            usage.cache_read_tokens.is_some() && usage.cache_write_tokens.is_some();
        saw_cache_hit |= usage.cache_read_tokens.unwrap_or(0) > 0;
        eprintln!(
            "[live-probe:{label}] round={round} input={} output={} cache_read={:?} cache_write={:?}",
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens
        );
    }

    assert!(
        saw_cache_metrics,
        "{label} 未返回可解析的 cache read/write usage"
    );
    assert!(saw_cache_hit, "{label} 连续三次稳定前缀请求均未命中缓存");
}

#[tokio::test]
#[ignore = "需要真实 DeepSeek API key（DEEPSEEK_API_KEY 环境变量）"]
async fn deepseek_responses_protocol_and_cache_live_probe() {
    let Some(api_key) = live_api_key() else {
        return;
    };
    let model = "deepseek-v4-flash";
    let base_url = std::env::var("DEEPSEEK_RESPONSES_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    let provider = create_provider(ProviderConfig::DeepSeekResponses {
        api_key,
        model: model.to_string(),
        base_url,
    })
    .expect("必须能构造 DeepSeek Responses provider");

    run_protocol_probe("responses", provider.as_ref(), model).await;
}

#[tokio::test]
#[ignore = "需要真实 DeepSeek API key（DEEPSEEK_API_KEY 环境变量）"]
async fn deepseek_anthropic_protocol_and_cache_live_probe() {
    let Some(api_key) = live_api_key() else {
        return;
    };
    let model = "deepseek-v4-pro";
    let base_url = std::env::var("DEEPSEEK_ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/anthropic".to_string());
    let provider = create_provider(ProviderConfig::DeepSeekAnthropic {
        api_key,
        model: model.to_string(),
        base_url: Some(base_url),
    })
    .expect("必须能构造 DeepSeek Anthropic provider");

    run_protocol_probe("anthropic", provider.as_ref(), model).await;
}
