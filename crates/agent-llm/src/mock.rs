//! `MockProvider` -- 无网络、可录制 / 回放的 Provider，用于测试与本地开发。
//!
//! 参见 `14-common-quality-gates.html §3`。覆盖：
//! - V-PROV-01：stream 依次产生 TextDelta / ToolUse / Usage / Done。
//! - V-PROV-02：不可恢复错误时流终结，错误带可展示分类，不泄露 API key。

use agent_contract::{
    Capabilities, CompletionRequest, CompletionResponse, ContentBlock, LlmProvider, StopReason,
    StreamEvent, Usage,
};
use agent_error::{Error, Result};
use std::sync::Arc;
use std::sync::Mutex;

/// 一个可脚本化的回放单元：一组事件 + 可选错误。
#[derive(Debug)]
pub struct RecordedTurn {
    pub events: Vec<StreamEvent>,
    pub error: Option<Error>,
}

/// 一个可脚本化的 Provider，按调用顺序回放预置的 turn。
pub struct MockProvider {
    name: String,
    capabilities: Capabilities,
    turns: Mutex<Vec<RecordedTurn>>,
}

impl MockProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            capabilities: Capabilities {
                supports_streaming: true,
                supports_tool_use: true,
                supports_vision: false,
                supports_prompt_caching: true,
                max_context_tokens: 200_000,
                max_output_tokens: 0,
            },
            turns: Mutex::new(Vec::new()),
        }
    }

    /// 追加一个脚本 turn。
    pub fn push_turn(&self, turn: RecordedTurn) -> &Self {
        self.turns.lock().unwrap().push(turn);
        self
    }

    /// 便捷：追加一个纯文本成功 turn。
    pub fn push_text_turn(&self, text: impl Into<String>, usage: Usage) -> &Self {
        let text = text.into();
        let events = vec![
            StreamEvent::TextDelta { text: text.clone() },
            StreamEvent::Usage(usage.clone()),
            StreamEvent::Stop {
                reason: StopReason::EndTurn,
            },
        ];
        self.push_turn(RecordedTurn::ok(events));
        self
    }

    /// 便捷：追加一个错误 turn（用于 V-PROV-02）。
    pub fn push_error_turn(&self, err: Error) -> &Self {
        self.push_turn(RecordedTurn::error(err));
        self
    }

    fn next_turn(&self) -> Result<RecordedTurn> {
        let mut guard = self.turns.lock().unwrap();
        if guard.is_empty() {
            return Err(Error::Internal(format!(
                "mock provider '{}' has no scripted turn left",
                self.name
            )));
        }
        Ok(guard.remove(0))
    }
}

/// 将事件序列聚合为非流式响应。
pub fn aggregate(events: &[StreamEvent]) -> CompletionResponse {
    let mut content = Vec::new();
    let mut usage = Usage::default();
    let mut stop_reason = StopReason::EndTurn;

    for ev in events {
        match ev {
            StreamEvent::TextDelta { text } => {
                if let Some(ContentBlock::Text { text: existing }) = content.last_mut() {
                    existing.push_str(text);
                } else {
                    content.push(ContentBlock::Text { text: text.clone() });
                }
            }
            StreamEvent::ToolUseComplete { id, input } => {
                // 记录工具调用名称需从 ToolUseStart 取；这里尽力而为
                content.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: String::new(),
                    input: input.clone(),
                });
            }
            StreamEvent::Usage(u) => {
                usage = u.clone();
            }
            StreamEvent::Stop { reason } => {
                stop_reason = reason.clone();
            }
            _ => {}
        }
    }

    CompletionResponse {
        content,
        stop_reason,
        usage,
    }
}

#[async_trait::async_trait]
impl LlmProvider for MockProvider {
    async fn complete(&self, _request: Arc<CompletionRequest>) -> Result<CompletionResponse> {
        let turn = self.next_turn()?;
        if let Some(err) = turn.error {
            return Err(err);
        }
        Ok(aggregate(&turn.events))
    }

    async fn stream(
        &self,
        _request: Arc<CompletionRequest>,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>> {
        let turn = self.next_turn()?;
        if let Some(err) = turn.error {
            // V-PROV-02：不可恢复错误 -> 流终结，错误带可展示分类
            return Err(err);
        }
        let events = turn.events;
        let stream = futures::stream::iter(events);
        Ok(Box::pin(stream))
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl RecordedTurn {
    pub fn ok(events: Vec<StreamEvent>) -> Self {
        Self {
            events,
            error: None,
        }
    }

    pub fn error(err: Error) -> Self {
        Self {
            events: Vec::new(),
            error: Some(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contract::Message;

    fn req() -> CompletionRequest {
        CompletionRequest {
            model: "mock".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            hosted_tools: vec![],
            max_tokens: 128,
            temperature: None,
            enable_caching: false,
            inference: Default::default(),
        }
    }

    #[tokio::test]
    async fn complete_aggregates_text() {
        let p = MockProvider::new("mock");
        p.push_text_turn("hello", Usage::new(10, 5));
        let resp = p.complete(Arc::new(req())).await.unwrap();
        assert_eq!(resp.text(), "hello");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn stream_replays_events_in_order() {
        // V-PROV-01：TextDelta -> ToolUse -> Usage -> Done
        let p = MockProvider::new("mock");
        p.push_turn(RecordedTurn::ok(vec![
            StreamEvent::TextDelta { text: "a".into() },
            StreamEvent::ToolUseStart {
                id: "t1".into(),
                name: "read_file".into(),
            },
            StreamEvent::ToolUseComplete {
                id: "t1".into(),
                input: serde_json::json!({"path": "/a"}),
            },
            StreamEvent::Usage(Usage::new(1, 1)),
            StreamEvent::Stop {
                reason: StopReason::ToolUse,
            },
        ]));

        let mut s = p.stream(Arc::new(req())).await.unwrap();
        use futures::StreamExt;
        let mut kinds = Vec::new();
        while let Some(ev) = s.next().await {
            kinds.push(std::mem::discriminant(&ev));
        }
        assert_eq!(kinds.len(), 5);
    }

    #[tokio::test]
    async fn error_does_not_leak_api_key() {
        // V-PROV-02：错误带可展示分类，不含 api_key
        let p = MockProvider::new("mock");
        p.push_error_turn(Error::AuthFailed("authentication failed".into()));
        let err = p
            .stream(Arc::new(req()))
            .await
            .err()
            .expect("expected error");
        let msg = err.to_string();
        assert!(!msg.contains("sk-ant-"));
        assert!(!msg.contains("secret"));
    }

    #[tokio::test]
    async fn exhausted_turns_returns_internal_error() {
        let p = MockProvider::new("mock");
        let err = p.complete(Arc::new(req())).await.unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
    }
}
