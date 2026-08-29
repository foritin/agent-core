use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_contract::{CompletionRequest, LlmProvider, Message, StreamEvent, Usage};
use agent_llm::{create_provider, ProviderConfig};
use futures::StreamExt;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::timeout;

const STABLE_PREFIX: &str =
    "stable-prefix-marker: keep system, tools, and earlier messages byte-stable";
const LOOPBACK_PHASE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
struct CapturedRequest {
    path: String,
    body: String,
}

#[derive(Clone)]
struct MockResponse {
    status: &'static str,
    content_type: &'static str,
    body: String,
}

impl MockResponse {
    fn sse(body: &str) -> Self {
        Self {
            status: "200 OK",
            content_type: "text/event-stream",
            body: body.to_string(),
        }
    }

    fn bad_request(body: &str) -> Self {
        Self {
            status: "400 Bad Request",
            content_type: "application/json",
            body: body.to_string(),
        }
    }
}

async fn spawn_mock_server(
    responses: Vec<MockResponse>,
) -> (
    String,
    Arc<Mutex<Vec<CapturedRequest>>>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = timeout(LOOPBACK_PHASE_TIMEOUT, listener.accept())
                .await
                .expect("timed out waiting for the loopback client to connect")
                .expect("failed to accept the loopback client");
            let request = timeout(LOOPBACK_PHASE_TIMEOUT, read_http_request(&mut socket))
                .await
                .expect("timed out reading the loopback HTTP request");
            requests_for_server.lock().unwrap().push(request);
            write_http_response(&mut socket, &response).await;
        }
    });
    (origin, requests, server)
}

async fn join_mock_server(mut server: JoinHandle<()>) {
    match timeout(LOOPBACK_PHASE_TIMEOUT, &mut server).await {
        Ok(result) => result.expect("loopback server task failed"),
        Err(_) => {
            server.abort();
            let _ = server.await;
            panic!("timed out waiting for the loopback server task to finish");
        }
    }
}

async fn read_http_request(socket: &mut TcpStream) -> CapturedRequest {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    let (header_end, content_length) = loop {
        let read = socket.read(&mut chunk).await.unwrap();
        assert!(
            read > 0,
            "connection closed before the request was complete"
        );
        bytes.extend_from_slice(&chunk[..read]);
        let Some(header_end) = find_bytes(&bytes, b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if bytes.len() >= header_end + 4 + content_length {
            break (header_end, content_length);
        }
    };

    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let path = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .to_string();
    let body_start = header_end + 4;
    let body = String::from_utf8(bytes[body_start..body_start + content_length].to_vec()).unwrap();
    CapturedRequest { path, body }
}

async fn write_http_response(socket: &mut TcpStream, response: &MockResponse) {
    let headers = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.content_type,
        response.body.len()
    );
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(response.body.as_bytes()).await.unwrap();
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn request(model: &str, append_tail: bool) -> Arc<CompletionRequest> {
    let mut messages = vec![Message::user_text(format!(
        "{STABLE_PREFIX}. Explain the cache result briefly."
    ))];
    if append_tail {
        messages.push(Message::user_text(
            "new tail that must not alter the earlier prefix",
        ));
    }
    Arc::new(CompletionRequest {
        model: model.to_string(),
        system: Some(STABLE_PREFIX.to_string()),
        messages,
        tools: vec![],
        hosted_tools: vec![],
        max_tokens: 128,
        temperature: None,
        enable_caching: true,
        inference: Default::default(),
    })
}

async fn collect_usage(provider: &dyn LlmProvider, request: Arc<CompletionRequest>) -> Vec<Usage> {
    let mut stream = provider.stream(request).await.unwrap();
    let mut usage = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Usage(value) = event {
            usage.push(value);
        }
    }
    usage
}

fn assert_stable_common_prefix(requests: &[CapturedRequest]) {
    assert_eq!(requests.len(), 2);
    assert_ne!(requests[0].body, requests[1].body);
    let marker_end = requests[0].body.find(STABLE_PREFIX).unwrap() + STABLE_PREFIX.len();
    let common = requests[0]
        .body
        .bytes()
        .zip(requests[1].body.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    assert!(
        common >= marker_end,
        "stable prefix diverged before the shared system/message bytes: {common} < {marker_end}"
    );
}

fn body_json(request: &CapturedRequest) -> Value {
    serde_json::from_str(&request.body).unwrap()
}

#[tokio::test]
async fn custom_deepseek_chat_gateway_requests_and_parses_cache_usage() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,",
        "\"completion_tokens\":2,\"prompt_cache_hit_tokens\":80,",
        "\"prompt_cache_miss_tokens\":20}}\n\n",
        "data: [DONE]\n\n"
    );
    let (origin, requests, server) =
        spawn_mock_server(vec![MockResponse::sse(sse), MockResponse::sse(sse)]).await;
    let provider = create_provider(ProviderConfig::DeepSeek {
        api_key: "test-key".into(),
        model: "deepseek-v4-pro".into(),
        base_url: Some(format!("{origin}/gateway/v1")),
    })
    .unwrap();

    for append_tail in [false, true] {
        let usage = collect_usage(provider.as_ref(), request("deepseek-v4-pro", append_tail)).await;
        let usage = usage.last().expect("chat usage event");
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(20));
    }
    join_mock_server(server).await;

    let requests = requests.lock().unwrap();
    assert!(requests
        .iter()
        .all(|request| request.path == "/gateway/v1/chat/completions"));
    assert!(requests.iter().all(|request| {
        body_json(request)["stream_options"]["include_usage"] == Value::Bool(true)
    }));
    assert_stable_common_prefix(&requests);
    assert_eq!(provider.capabilities().max_context_tokens, 1_000_000);
    assert!(provider.capabilities().supports_prompt_caching);
}

#[tokio::test]
async fn deepseek_responses_gateway_preserves_cache_semantics() {
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",",
        "\"sequence_number\":1,\"delta\":\"ok\"}\n\n",
        "data: {\"type\":\"response.completed\",\"sequence_number\":2,",
        "\"response\":{\"status\":\"completed\",\"output\":[],",
        "\"usage\":{\"input_tokens\":100,\"output_tokens\":2,",
        "\"input_tokens_details\":{\"cached_tokens\":80}}}}\n\n"
    );
    let (origin, requests, server) =
        spawn_mock_server(vec![MockResponse::sse(sse), MockResponse::sse(sse)]).await;
    let provider = create_provider(ProviderConfig::DeepSeekResponses {
        api_key: "test-key".into(),
        model: "deepseek-v4-flash".into(),
        base_url: format!("{origin}/gateway/v1"),
    })
    .unwrap();

    for append_tail in [false, true] {
        let usage =
            collect_usage(provider.as_ref(), request("deepseek-v4-flash", append_tail)).await;
        let usage = usage.last().expect("Responses usage event");
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(20));
    }
    join_mock_server(server).await;

    let requests = requests.lock().unwrap();
    assert!(requests
        .iter()
        .all(|request| request.path == "/gateway/v1/responses"));
    assert!(requests.iter().all(|request| {
        let body = body_json(request);
        body["store"] == Value::Bool(false) && body.get("previous_response_id").is_none()
    }));
    assert_stable_common_prefix(&requests);
    assert_eq!(provider.name(), "deepseek_responses");
    assert_eq!(provider.capabilities().max_context_tokens, 1_000_000);
    assert!(provider.capabilities().supports_prompt_caching);
}

#[tokio::test]
async fn deepseek_anthropic_gateway_uses_automatic_cache_without_markers() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{",
        "\"input_tokens\":20,\"output_tokens\":0,",
        "\"cache_read_input_tokens\":80,",
        "\"cache_creation_input_tokens\":0}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,",
        "\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{},",
        "\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",",
        "\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (origin, requests, server) =
        spawn_mock_server(vec![MockResponse::sse(sse), MockResponse::sse(sse)]).await;
    let provider = create_provider(ProviderConfig::DeepSeekAnthropic {
        api_key: "test-key".into(),
        model: "deepseek-v4-pro".into(),
        base_url: Some(format!("{origin}/gateway/anthropic")),
    })
    .unwrap();

    for append_tail in [false, true] {
        let usage = collect_usage(provider.as_ref(), request("deepseek-v4-pro", append_tail)).await;
        let cached = usage
            .iter()
            .find(|usage| usage.cache_read_tokens.is_some())
            .expect("Anthropic message_start cache usage");
        assert_eq!(cached.cache_read_tokens, Some(80));
        assert_eq!(cached.cache_write_tokens, Some(20));
    }
    join_mock_server(server).await;

    let requests = requests.lock().unwrap();
    assert!(requests
        .iter()
        .all(|request| request.path == "/gateway/anthropic/v1/messages"));
    assert!(requests
        .iter()
        .all(|request| !request.body.contains("cache_control")));
    assert_stable_common_prefix(&requests);
    assert_eq!(provider.name(), "deepseek_anthropic");
    assert_eq!(provider.capabilities().max_context_tokens, 1_000_000);
    assert!(provider.capabilities().supports_prompt_caching);
}

#[tokio::test]
async fn old_custom_chat_gateway_retries_without_stream_options() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (origin, requests, server) = spawn_mock_server(vec![
        MockResponse::bad_request(r#"{"error":{"message":"unknown field stream_options"}}"#),
        MockResponse::sse(sse),
    ])
    .await;
    let provider = create_provider(ProviderConfig::DeepSeek {
        api_key: "test-key".into(),
        model: "deepseek-v4-pro".into(),
        base_url: Some(format!("{origin}/legacy/v1")),
    })
    .unwrap();

    let mut stream = provider
        .stream(request("deepseek-v4-pro", false))
        .await
        .expect("gateway fallback should succeed");
    while stream.next().await.is_some() {}
    join_mock_server(server).await;

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].path, "/legacy/v1/chat/completions");
    assert_eq!(requests[1].path, "/legacy/v1/chat/completions");
    assert!(body_json(&requests[0]).get("stream_options").is_some());
    assert!(body_json(&requests[1]).get("stream_options").is_none());
}
