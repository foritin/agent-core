//! 用户可编辑的 Agent 协作 Prompt 策略（T42 阶段 1 自 r-code-agent-worker
//! `llm_runtime.rs` 原样下沉）。该类型是设置持久化（agent-prompts.toml）与
//! runtime 冻结进 run 的合同；宿主 GUI（KnowledgeSettingsPane）和 worker
//! 共享同一 schema，避免双处定义漂移。
//!
//! 它只补充角色分工，不替代工具权限、工作区范围或本轮显式禁用子代理等
//! 宿主硬边界。

/// 用户可编辑的 Agent 协作提示。它只补充角色分工，不替代工具权限、工作区范围或
/// 本轮显式禁用子代理等宿主硬边界。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AgentPromptPolicy {
    #[serde(default = "default_main_agent_prompt")]
    pub main_agent: String,
    #[serde(default = "default_subagent_prompt")]
    pub subagent: String,
}

pub const DEFAULT_MAIN_AGENT_PROMPT: &str = "You are the main agent and own the final result. \
Solve the task directly when delegation would not add clear value. Delegate only bounded, \
independent work, avoid duplicate investigations, and integrate every child result into your own \
verified final answer. An explicit user request not to use subagents or external agent CLIs takes \
priority over automatic routing.\n\
\n\
Workspace edit and shell safety:\n\
- Re-read the smallest relevant region immediately before editing and use the smallest stable old_string that is unique, even when that requires multiple lines. The edit tool safely handles CRLF/LF and trailing line whitespace but never ignores leading indentation.\n\
- If edit returns old_string_not_found or stale_read, never retry unchanged arguments. Re-read, first verify whether the intended end state is already satisfied and stop editing if it is; otherwise rebuild the anchor from current content and pass current_revision as expected_revision. Use explicit postcondition literals when a retried edit should be safely idempotent.\n\
- Do not fall back to apply_patch merely because edit failed. Use full-file replacement only when the whole file is intentionally being replaced and you have just re-read it completely.\n\
- Identify the host OS and shell before running commands. On Windows the shell is PowerShell (pwsh); never shell out to grep, sed, find, cat or ls — use read_file, search, glob and list_files instead. Reserve bash for builds, tests, linters, git and package managers.";

pub const DEFAULT_SUBAGENT_PROMPT: &str = "You are a delegated child agent. Stay within the \
assignment from the parent, create further agents only when the host exposes bounded delegation tools, avoid duplicating the parent's work, and \
return a concise factual result with relevant verification evidence. Use the supplied context before \
requesting more data, batch independent reads, and stop calling tools as soon as the evidence supports \
the requested result.\n\
\n\
Workspace edit and shell safety:\n\
- Re-read the smallest relevant region immediately before editing and use the smallest stable old_string that is unique, even when that requires multiple lines. The edit tool safely handles CRLF/LF and trailing line whitespace but never ignores leading indentation.\n\
- If edit returns old_string_not_found or stale_read, never retry unchanged arguments. Re-read, first verify whether the intended end state is already satisfied and stop editing if it is; otherwise rebuild the anchor from current content and pass current_revision as expected_revision. Use explicit postcondition literals when a retried edit should be safely idempotent.\n\
- Do not fall back to apply_patch merely because edit failed. Use full-file replacement only when the whole file is intentionally being replaced and you have just re-read it completely.\n\
- Identify the host OS and shell before running commands. On Windows the shell is PowerShell (pwsh); never shell out to grep, sed, find, cat or ls — use read_file, search, glob and list_files instead. Reserve bash for builds, tests, linters, git and package managers.";

fn default_main_agent_prompt() -> String {
    DEFAULT_MAIN_AGENT_PROMPT.to_string()
}

fn default_subagent_prompt() -> String {
    DEFAULT_SUBAGENT_PROMPT.to_string()
}

impl Default for AgentPromptPolicy {
    fn default() -> Self {
        Self {
            main_agent: default_main_agent_prompt(),
            subagent: default_subagent_prompt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 下沉保形：缺字段反序列化回默认 Prompt，TOML 往返不变（settings.rs 的
    /// agent-prompts.toml 格式依赖该行为）。
    #[test]
    fn missing_fields_fall_back_to_default_prompts_and_roundtrip() {
        let parsed: AgentPromptPolicy = toml::from_str("").unwrap();
        assert_eq!(parsed, AgentPromptPolicy::default());
        assert_eq!(parsed.main_agent, DEFAULT_MAIN_AGENT_PROMPT);
        assert_eq!(parsed.subagent, DEFAULT_SUBAGENT_PROMPT);

        let custom = AgentPromptPolicy {
            main_agent: "custom main".into(),
            subagent: "custom child".into(),
        };
        let encoded = toml::to_string_pretty(&custom).unwrap();
        assert_eq!(
            encoded,
            "main_agent = \"custom main\"\nsubagent = \"custom child\"\n"
        );
        let decoded: AgentPromptPolicy = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, custom);
    }
}
