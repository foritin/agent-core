//! `agent-llm` -- LLM Provider 实现。
//!
//! 参见 `01-llm-provider.html`。提供 Anthropic Messages / OpenAI Chat Completions /
//! OpenAI Responses / DeepSeek 四个真实 Provider，以及用于无网络测试的
//! `MockProvider`。通过 [`create_provider`] 工厂按 [`ProviderConfig`] 构造
//! `Box<dyn LlmProvider>`。
//!
//! 三种线路协议不能靠厂商名字区分，同一家的不同 base_url 往往是不同协议
//! （火山方舟 `/api/coding` 是 Anthropic 口、`/api/coding/v3` 是 Chat、
//! `/api/v3` 同时有 Chat 和 Responses）。调用方需显式选择。

pub mod anthropic;
pub mod deepseek;
pub mod dialect;
pub mod mock;
pub mod openai;
pub mod responses;
pub mod url;

pub use anthropic::AnthropicProvider;
pub use deepseek::DeepSeekProvider;
pub use dialect::{dialect_for, DialectPort, WireDialect};
pub use mock::{MockProvider, RecordedTurn};
pub use openai::OpenAiProvider;
pub use responses::{ReasoningMode, ResponsesProvider};

use agent_contract::{ContentBlock, LlmProvider, Message};
use agent_error::{other, Result};

/// 断言请求消息中不存在未物化的 `ContentBlock::Attachment`（fail closed）。
///
/// 引用块必须在进入适配器前由运行时从 BlobStore 物化为 `Image`/`File` 块。
/// 把引用降级为占位文本后继续请求会静默丢图——这是验收失败，必须返回类型化
/// 错误（docs/archive/implementation/multimodal-attachments-and-deepseek-plan-anchoring-implementation.md §6.3）。各协议适配器在最终序列化前调用。
pub fn assert_no_unresolved_attachments(messages: &[Message]) -> Result<()> {
    for message in messages {
        for block in &message.content {
            if let ContentBlock::Attachment { source } = block {
                return Err(other(format!(
                    "ATTACHMENT_NOT_MATERIALIZED: attachment '{}' ({}) reached the provider \
                     adapter unresolved; the runtime must materialize it from the blob store \
                     before dispatch",
                    source.name, source.attachment_id
                )));
            }
        }
    }
    Ok(())
}

/// Provider 配置（工厂入参）。
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Anthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
    /// Kimi For Coding 的 Anthropic Messages 入口。协议形状相同，但上下文窗口不同。
    KimiCodingAnthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
    /// 火山方舟 Coding / Agent Plan 的 Anthropic Messages 口。
    ArkAnthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
        /// `ark_coding` 或 `ark_agent`，决定上下文窗口。
        kind: String,
    },
    /// 火山方舟 Coding Plan 的 OpenAI Chat 口。
    ArkChat {
        api_key: String,
        model: String,
        base_url: String,
        /// 稳定厂商身份（当前为 `ark_coding_openai`）。
        kind: String,
    },
    /// Kimi For Coding 的 OpenAI Chat 口。
    KimiChat {
        api_key: String,
        model: String,
        base_url: String,
    },
    OpenAi {
        api_key: String,
        model: String,
        base_url: String,
    },
    /// OpenAI Responses API。`base_url` 必须已包含版本段（如 `/v1`、`/api/v3`）。
    Responses {
        api_key: String,
        model: String,
        base_url: String,
        /// 推理内容策略。默认 [`ReasoningMode::Drop`]（兼容性最好）；
        /// 只有 OpenAI 官方与 xAI 支持 [`ReasoningMode::EncryptedReplay`]。
        reasoning: ReasoningMode,
    },
    /// DeepSeek 的 Responses 兼容口。协议仍是 Responses，但保留厂商身份以启用
    /// 自动前缀缓存 usage 语义与 1M 上下文能力声明。
    DeepSeekResponses {
        api_key: String,
        model: String,
        base_url: String,
    },
    /// DeepSeek 的 Anthropic Messages 兼容口。它使用自动前缀缓存，不需要
    /// Anthropic `cache_control` 标记。
    DeepSeekAnthropic {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
    /// Ark Coding Plan / Agent Plan 的 Responses 口。
    ///
    /// `kind` 为稳定的厂商身份（`ark_coding` / `ark_coding_openai` / `ark_agent`），
    /// 决定上下文窗口（Coding 256K，Agent 1M）。reasoning 以 Ark 的明文
    /// summary 形式保存并回传（`ReasoningMode::SummaryReplay`）。
    ArkResponses {
        api_key: String,
        model: String,
        base_url: String,
        kind: String,
    },
    DeepSeek {
        api_key: String,
        model: String,
        base_url: Option<String>,
    },
}

/// 按配置构造 Provider 实例。
pub fn create_provider(config: ProviderConfig) -> Result<Box<dyn LlmProvider>> {
    match config {
        ProviderConfig::Anthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(AnthropicProvider::new(api_key, model, base_url)?)),
        ProviderConfig::KimiCodingAnthropic {
            api_key,
            model,
            base_url,
        } => {
            let dialect = dialect_for("kimi_coding", &model, DialectPort::AnthropicMessages)
                .ok_or(other(
                    "kimi_coding anthropic dialect not supported for this model",
                ))?;
            Ok(Box::new(
                AnthropicProvider::new_kimi_coding(api_key, model, base_url)?.with_dialect(dialect),
            ))
        }
        ProviderConfig::ArkAnthropic {
            api_key,
            model,
            base_url,
            kind,
        } => {
            let dialect =
                dialect_for(&kind, &model, DialectPort::AnthropicMessages).ok_or(other(
                    format!("{kind} anthropic dialect not supported for this model"),
                ))?;
            Ok(Box::new(
                AnthropicProvider::new(api_key, model, base_url)?.with_dialect(dialect),
            ))
        }
        ProviderConfig::ArkChat {
            api_key,
            model,
            base_url,
            kind,
        } => {
            let dialect = dialect_for(&kind, &model, DialectPort::OpenAiChat).ok_or(other(
                format!("{kind} openai-chat dialect not supported for this model"),
            ))?;
            Ok(Box::new(
                OpenAiProvider::new(api_key, model, base_url).with_dialect(dialect),
            ))
        }
        ProviderConfig::KimiChat {
            api_key,
            model,
            base_url,
        } => {
            let dialect = dialect_for("kimi_coding", &model, DialectPort::OpenAiChat).ok_or(
                other("kimi_coding openai-chat dialect not supported for this model"),
            )?;
            Ok(Box::new(
                OpenAiProvider::new(api_key, model, base_url).with_dialect(dialect),
            ))
        }
        ProviderConfig::OpenAi {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(OpenAiProvider::new(api_key, model, base_url))),
        ProviderConfig::Responses {
            api_key,
            model,
            base_url,
            reasoning,
        } => Ok(Box::new(
            ResponsesProvider::new(api_key, model, base_url).with_reasoning(reasoning),
        )),
        ProviderConfig::DeepSeekResponses {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(ResponsesProvider::new_deepseek(
            api_key, model, base_url,
        ))),
        ProviderConfig::DeepSeekAnthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(AnthropicProvider::new_deepseek(
            api_key, model, base_url,
        )?)),
        ProviderConfig::ArkResponses {
            api_key,
            model,
            base_url,
            kind,
        } => Ok(Box::new(ResponsesProvider::new_ark(
            api_key, model, base_url, &kind,
        ))),
        ProviderConfig::DeepSeek {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(DeepSeekProvider::new_with_base_url(
            api_key, model, base_url,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contract::{AttachmentKind, AttachmentPurpose, AttachmentRefV1};

    #[test]
    fn unresolved_attachment_fails_closed_before_dispatch() {
        let message = Message {
            role: agent_contract::Role::User,
            content: vec![
                agent_contract::ContentBlock::Text { text: "hi".into() },
                agent_contract::ContentBlock::Attachment {
                    source: AttachmentRefV1 {
                        version: 1,
                        attachment_id: "att-1".into(),
                        name: "shot.png".into(),
                        media_type: "image/png".into(),
                        kind: AttachmentKind::Image,
                        byte_len: 8,
                        width: Some(8),
                        height: Some(8),
                        purpose: AttachmentPurpose::NativeInput,
                    },
                },
            ],
        };
        let error = assert_no_unresolved_attachments(&[message]).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("ATTACHMENT_NOT_MATERIALIZED"), "got: {text}");
        // 错误详情不得包含附件元数据以外的任何正文。
        assert!(text.contains("att-1"));
    }

    #[test]
    fn materialized_messages_pass_the_gate() {
        let message = Message::user_text("plain");
        assert!(assert_no_unresolved_attachments(&[message]).is_ok());
    }

    #[test]
    fn factory_builds_each_protocol() {
        let anthropic = create_provider(ProviderConfig::Anthropic {
            api_key: "k".into(),
            model: "claude-sonnet-5".into(),
            base_url: None,
        })
        .unwrap();
        assert_eq!(anthropic.name(), "anthropic");

        let kimi = create_provider(ProviderConfig::KimiCodingAnthropic {
            api_key: "k".into(),
            model: "k3-256k".into(),
            base_url: Some("https://api.kimi.com/coding/".into()),
        })
        .unwrap();
        assert_eq!(kimi.name(), "kimi_coding");
        assert_eq!(kimi.capabilities().max_context_tokens, 262_144);

        let chat = create_provider(ProviderConfig::OpenAi {
            api_key: "k".into(),
            model: "gpt-5.5".into(),
            base_url: "https://api.openai.com/v1".into(),
        })
        .unwrap();
        assert_eq!(chat.name(), "openai");

        let responses = create_provider(ProviderConfig::Responses {
            api_key: "k".into(),
            model: "gpt-5.6-sol".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: ReasoningMode::EncryptedReplay,
        })
        .unwrap();
        assert_eq!(responses.name(), "openai_responses");

        let deepseek_responses = create_provider(ProviderConfig::DeepSeekResponses {
            api_key: "k".into(),
            model: "deepseek-v4-flash".into(),
            base_url: "https://api.deepseek.com".into(),
        })
        .unwrap();
        assert_eq!(deepseek_responses.name(), "deepseek_responses");
        assert!(deepseek_responses.capabilities().supports_prompt_caching);

        let deepseek_anthropic = create_provider(ProviderConfig::DeepSeekAnthropic {
            api_key: "k".into(),
            model: "deepseek-v4-pro".into(),
            base_url: Some("https://api.deepseek.com/anthropic".into()),
        })
        .unwrap();
        assert_eq!(deepseek_anthropic.name(), "deepseek_anthropic");
        assert_eq!(
            deepseek_anthropic.capabilities().max_context_tokens,
            1_000_000
        );

        let ark_agent_responses = create_provider(ProviderConfig::ArkResponses {
            api_key: "k".into(),
            model: "glm-5.3".into(),
            base_url: "https://ark.cn-beijing.volces.com/api/plan/v3".into(),
            kind: "ark_agent".into(),
        })
        .unwrap();
        assert_eq!(ark_agent_responses.name(), "ark_responses");
        assert_eq!(
            ark_agent_responses.capabilities().max_context_tokens,
            1_048_576
        );
        assert!(!ark_agent_responses.capabilities().supports_vision);

        let ark_coding_responses = create_provider(ProviderConfig::ArkResponses {
            api_key: "k".into(),
            model: "doubao-seed-2.1-pro".into(),
            base_url: "https://ark.cn-beijing.volces.com/api/coding/v3".into(),
            kind: "ark_coding".into(),
        })
        .unwrap();
        assert_eq!(
            ark_coding_responses.capabilities().max_context_tokens,
            256_000
        );
        assert!(ark_coding_responses.capabilities().supports_vision);
    }

    /// F-corr-05：dialect 不支持的 provider kind 走可读错误而非
    /// `unreachable!` panic（用户配置该类服务时不再使宿主崩溃）。
    #[test]
    fn unsupported_provider_kind_returns_error_instead_of_panicking() {
        let result = create_provider(ProviderConfig::ArkAnthropic {
            api_key: "k".into(),
            model: "m".into(),
            base_url: Some("https://example.invalid".into()),
            kind: "not_a_real_kind".into(),
        });
        let error = match result {
            Ok(_) => panic!("unsupported kind must not construct a provider"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("dialect not supported"),
            "dialect error expected, got: {error}"
        );
    }
}
