//! `agent-config` -- 配置管理。
//!
//! 参见 `08-config-management.html`。TOML 配置 + 环境变量覆盖 + 默认值 + 验证。
//! 敏感值（api_key）绝不出现在 Debug 输出中（V-CFG-02）。

use agent_error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub mod agent_prompts;

pub use agent_prompts::{AgentPromptPolicy, DEFAULT_MAIN_AGENT_PROMPT, DEFAULT_SUBAGENT_PROMPT};

/// 顶层配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_provider")]
    pub default_provider: String,

    #[serde(default = "default_log_level")]
    pub log_level: String,

    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    #[serde(default)]
    pub mcp_servers: HashMap<String, ServerSpec>,

    #[serde(default = "default_storage")]
    pub storage: StorageConfig,

    #[serde(default = "default_compaction")]
    pub compaction: CompactionConfig,

    /// 主 Agent、跨引擎委派与结果复核策略。
    #[serde(default)]
    pub orchestration: OrchestrationConfig,

    /// Plan 入口建议客户偏好（docs/archive/implementation/plan-mode-dual-track-gate.md §15.1）。
    /// 普通客户只操作一个布尔偏好；Provider 资格、实验档位和证据版本由宿主
    /// 控制，不在本 schema 中。
    #[serde(default)]
    pub planning: PlanningConfig,

    /// 诊断开关（默认全关，不改变任何产品行为）。
    #[serde(default)]
    pub diagnostics: DiagnosticsConfig,

    /// 图片理解引擎（本机 OCR / 视觉模型，二选一，默认 OCR）。
    /// 旧配置缺失该段时反序列化为默认值，不产生迁移错误。
    #[serde(default)]
    pub image_understanding: ImageUnderstandingConfig,

    #[serde(default)]
    pub tauri: Option<TauriConfig>,

    /// 配置 schema 版本（FX-12）。缺失（旧文件）= 1，向后兼容；
    /// 宿主加载后对「未来版本」显式拒绝，配合 serde default 防止旧字段
    /// 被静默吞掉而无人察觉。
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// 配置 schema 当前版本。缺失字段的旧配置反序列化到该值。
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    CONFIG_SCHEMA_VERSION
}

/// 图片理解引擎选择（docs/archive/implementation/settings-ux-and-image-understanding.md D2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageUnderstandingEngine {
    /// 本机系统 OCR：png/jpeg 提取文字注入上下文（离线、免费）。
    #[default]
    Ocr,
    /// 视觉模型：由配置的多模态模型理解整张图片并生成描述文本。
    Model,
}

/// 图片理解配置。`engine == Model` 时 `model_provider` / `model` 必填，
/// 发送路径会做权威校验并返回可读错误。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ImageUnderstandingConfig {
    #[serde(default)]
    pub engine: ImageUnderstandingEngine,
    /// engine == Model 时必填：`config.providers` 的 key。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    /// engine == Model 时必填：该 provider 下的模型 id。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Plan 入口建议的客户偏好。默认关闭：首次默认开启必须等发布硬门全部通过并
/// 经过分批 rollout（docs §15.1），在此之前 evidence 解析也恒为关闭。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PlanningConfig {
    /// DeepSeek 识别到复杂任务时先询问是否制定计划。只影响以后创建的建议；
    /// 关闭不取消已 pending/accepted 的决定，也不禁用手动 Plan。
    #[serde(default)]
    pub suggest_complex_tasks: bool,
    /// DeepSeek Plan 锚定（docs/archive/implementation/multimodal-attachments-and-deepseek-plan-anchoring-implementation.md §8.1）：用户实际进入
    /// DeepSeek Plan 后，是否启用最小 Plan 轨迹（5→8 只读目录 + 最小注入）与
    /// 批准后的完整执行恢复。
    ///
    /// 与 `suggest_complex_tasks` 互不替代：可以关闭自动建议但手动进入 Plan 时
    /// 仍启用锚定，也可以保留建议但让 Plan 使用 baseline。默认 false，旧配置
    /// 缺字段时为 false。`R_CODE_PLANNING_EMERGENCY_OFF=1` 同时关闭两者（但不
    /// 关闭 Plan 的只读安全硬门）。开关值在 Plan 创建时冻结；运行中切换设置只
    /// 影响之后新建的 Plan。
    #[serde(default)]
    pub deepseek_plan_anchoring: bool,
}

/// 诊断开关段。开启的项只增加观测输出（旁路文件 / 计数），不影响请求形状、
/// 执行判定与 canonical 会话数据。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticsConfig {
    /// 会话请求信封审计（docs/archive/implementation/request-audit-and-anchoring.md 阶段 A）。默认关闭。
    /// 开启后每轮派发前向 sessions/request-audit/{storage_id}.jsonl 追加
    /// RequestHeader（含目录清单与输出预算）并做重建自检（log-only）。
    #[serde(default)]
    pub request_audit: bool,
}

/// 单个 LLM Provider 配置。api_key 在 Debug 中脱敏。
#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Stable catalog/vendor identity. Display names and gateway URLs remain freely editable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// 线路协议（wire protocol）：`anthropic_messages` / `openai_chat` /
    /// `openai_responses`。决定请求体形状、SSE 事件格式和鉴权头。
    ///
    /// **这是用户显式选择的值，不是推导出来的。** 同一个 base_url 常常同时支持多种
    /// 协议（火山方舟 `/api/coding` 是 Anthropic 口、`/api/coding/v3` 是 OpenAI 口），
    /// 计费和能力都不同，只能由用户决定走哪个。
    ///
    /// `None` = 尚未选择（升级前保存的旧配置）。此时由调用方决定回退策略——
    /// R-Code 的做法见 `commands.rs::build_provider_config`：按目录推断，但绝不
    /// 自动选中 Responses。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    /// Whether provider-exposed reasoning summaries/content are shown in the conversation UI.
    ///
    /// This is a presentation preference only: it never enables or disables model reasoning.
    /// Legacy configs default to visible so upgrading does not silently hide useful output.
    #[serde(default = "default_show_reasoning", skip_serializing_if = "is_true")]
    pub show_reasoning: bool,
}

fn default_show_reasoning() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("provider_kind", &self.provider_kind)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("protocol", &self.protocol)
            .field("show_reasoning", &self.show_reasoning)
            .finish()
    }
}

/// 存储路径配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageConfig {
    pub base_dir: PathBuf,
    pub sessions_dir: PathBuf,
    pub skills_dir: PathBuf,
    pub memories_dir: PathBuf,
}

/// 压缩配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionConfig {
    #[serde(default = "default_compaction_strategy")]
    pub strategy: String,

    #[serde(default = "default_max_context")]
    pub max_context_tokens: u32,

    #[serde(default = "default_trigger")]
    pub trigger_threshold: f64,
}

/// Tauri 外壳配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriConfig {
    #[serde(default = "default_theme")]
    pub theme: String,

    #[serde(default = "default_font_size")]
    pub font_size: u32,
}

/// 长任务循环护栏预算。宿主侧硬上限与停止信号阈值；缺字段时回落到内置默认值，
/// 旧配置文件因此保持兼容。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunBudgetConfig {
    /// 单次 run 最多允许的工具轮数（模型回合产出工具调用即 +1）。
    #[serde(default = "default_max_tool_rounds")]
    pub max_tool_rounds: u32,
    /// 单次 run 的墙钟上限（秒）。
    #[serde(default = "default_max_run_seconds")]
    pub max_run_seconds: u64,
    /// 单次 run 累计思考量上限（字符）。
    #[serde(default = "default_reasoning_budget_chars")]
    pub reasoning_budget_chars: u64,
    /// 同一错误指纹连续失败多少次后停止。
    #[serde(default = "default_same_error_limit")]
    pub same_error_limit: u8,
    /// 连续多少轮没有可观察进展后停止。
    #[serde(default = "default_no_progress_rounds")]
    pub no_progress_rounds: u32,
    /// 相邻两轮工具请求完全一致时视为 replay。
    #[serde(default = "default_true")]
    pub replay_detection: bool,
    /// 单次 run 允许成功修改的不同文件数上限。
    #[serde(default = "default_diff_file_limit")]
    pub diff_file_limit: u32,
    /// 单次 run 允许的累计变更字节上限（old+new 内容长度之和）。
    #[serde(default = "default_diff_byte_limit")]
    pub diff_byte_limit: u64,
    /// 测试/构建命令连续失败多少次后停止。
    #[serde(default = "default_test_fail_limit")]
    pub test_fail_limit: u8,
    /// 测试全绿后是否创建 git 绿灯 checkpoint。
    #[serde(default = "default_true")]
    pub checkpoint_enabled: bool,
}

impl Default for RunBudgetConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: default_max_tool_rounds(),
            max_run_seconds: default_max_run_seconds(),
            reasoning_budget_chars: default_reasoning_budget_chars(),
            same_error_limit: default_same_error_limit(),
            no_progress_rounds: default_no_progress_rounds(),
            replay_detection: true,
            diff_file_limit: default_diff_file_limit(),
            diff_byte_limit: default_diff_byte_limit(),
            test_fail_limit: default_test_fail_limit(),
            checkpoint_enabled: true,
        }
    }
}

impl RunBudgetConfig {
    /// 产品层硬边界校验。范围与运行时 clamp 保持一致，但这里直接拒绝而非收紧。
    pub fn validate(&self) -> Result<()> {
        if !(4..=200).contains(&self.max_tool_rounds) {
            return Err(Error::Config(
                "orchestration.run_budget.max_tool_rounds must be between 4 and 200".to_string(),
            ));
        }
        if !(300..=86_400).contains(&self.max_run_seconds) {
            return Err(Error::Config(
                "orchestration.run_budget.max_run_seconds must be between 300 and 86400"
                    .to_string(),
            ));
        }
        if !(20_000..=4_000_000).contains(&self.reasoning_budget_chars) {
            return Err(Error::Config(
                "orchestration.run_budget.reasoning_budget_chars must be between 20000 and 4000000"
                    .to_string(),
            ));
        }
        if !(1..=10).contains(&self.same_error_limit) {
            return Err(Error::Config(
                "orchestration.run_budget.same_error_limit must be between 1 and 10".to_string(),
            ));
        }
        if !(2..=200).contains(&self.no_progress_rounds) {
            return Err(Error::Config(
                "orchestration.run_budget.no_progress_rounds must be between 2 and 200".to_string(),
            ));
        }
        if !(1..=1_000).contains(&self.diff_file_limit) {
            return Err(Error::Config(
                "orchestration.run_budget.diff_file_limit must be between 1 and 1000".to_string(),
            ));
        }
        if !(65_536..=1_073_741_824).contains(&self.diff_byte_limit) {
            return Err(Error::Config(
                "orchestration.run_budget.diff_byte_limit must be between 65536 and 1073741824"
                    .to_string(),
            ));
        }
        if !(1..=10).contains(&self.test_fail_limit) {
            return Err(Error::Config(
                "orchestration.run_budget.test_fail_limit must be between 1 and 10".to_string(),
            ));
        }
        Ok(())
    }
}

fn default_max_tool_rounds() -> u32 {
    60
}

fn default_max_run_seconds() -> u64 {
    14_400
}

fn default_reasoning_budget_chars() -> u64 {
    120_000
}

fn default_same_error_limit() -> u8 {
    3
}

fn default_no_progress_rounds() -> u32 {
    24
}

fn default_diff_file_limit() -> u32 {
    60
}

fn default_diff_byte_limit() -> u64 {
    262_144
}

fn default_test_fail_limit() -> u8 {
    3
}

/// Agent 编排设置。所有自动决策都必须在运行事件中公开策略结论。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// 新会话默认由哪一个主 Agent 执行。
    #[serde(default)]
    pub default_agent_engine: MainAgentEngine,
    /// `delegate_task(agent="auto")` 的路由策略。
    #[serde(default)]
    pub delegation_router: DelegationRouterMode,
    /// 是否允许两个 Agent 引擎互相委派。关闭后只允许同引擎子智能体。
    #[serde(default = "default_true")]
    pub allow_cross_engine_delegation: bool,
    /// 主回复完成后的显式质量复核策略。
    #[serde(default)]
    pub quality_loop: QualityLoopMode,
    /// 执行质量复核的引擎。
    #[serde(default)]
    pub quality_reviewer: QualityReviewer,
    /// 最多复核/修订轮数；产品层限制为 1..=3。
    #[serde(default = "default_review_rounds")]
    pub max_review_rounds: u8,
    /// 已通过宿主连通验证的子代理候选槽。空池继续使用 delegation_router。
    #[serde(default)]
    pub subagent_pool: SubagentPoolConfig,
    /// 长任务循环护栏预算与停止信号阈值。
    #[serde(default)]
    pub run_budget: RunBudgetConfig,
    /// 首轮派发的工具目录策略（锚定实验，docs/archive/implementation/request-audit-and-anchoring.md C1）。
    /// 默认 Full 即现状（不过滤）；非默认值仅裁剪模型可见目录（呈现层），
    /// 不改变任何执行判定。
    #[serde(default)]
    pub first_round_catalog: FirstRoundCatalog,
    /// 受限目录的晋升信号。默认 Either：首轮 outcome 含任意 assistant 内容
    /// 即恢复完整目录（纯文字首答不会困死在受限目录）。
    #[serde(default)]
    pub first_round_promote_on: FirstRoundPromoteOn,
}

/// 首轮派发的工具目录策略（锚定实验）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FirstRoundCatalog {
    /// 默认：不过滤，首轮即完整目录（现状）。
    #[default]
    Full,
    /// 首轮只暴露只读探索五件套。serde 名固定为 "readonly"（无下划线，
    /// 与 docs/archive/implementation/request-audit-and-anchoring.md B2 分组名一致）。
    #[serde(rename = "readonly")]
    ReadOnly,
    /// 首轮只暴露 read_file + edit（对标 dsh Minimal 工具对的编辑变体）。
    EditorPair,
    /// 规划门：首轮起零工作工具，目录仅含 plan_ready（需配 plan_complete
    /// 晋升信号）。剥夺跨回合持续到模型声明规划完成——恶劣环境压榨首轮
    /// 思考的实验形态。
    PlanGate,
}

/// 晋升信号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FirstRoundPromoteOn {
    /// 默认：本轮 outcome 含任意 assistant 内容（Text 或 ToolUse）即晋升。
    /// 等价 dsh promoteOn: either——纯文字首答不会困死在受限目录。
    #[default]
    Either,
    /// 仅首次 ToolUse 晋升；模型一直不调工具则一直停留在受限目录
    ///（dsh 的 tool-call 模式，保留作对照）。困死是该对照组的有意行为。
    ToolCall,
    /// 仅当模型调用 plan_ready 才晋升。剥夺跨回合、跨 run 持续（会话级
    /// 粘性），直到模型自己声明规划完成。纯文本终答自然结束 run，粘性
    /// 保持到下一个 run 的首轮。
    PlanComplete,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            default_agent_engine: MainAgentEngine::Native,
            delegation_router: DelegationRouterMode::Balanced,
            allow_cross_engine_delegation: true,
            quality_loop: QualityLoopMode::Off,
            quality_reviewer: QualityReviewer::Native,
            max_review_rounds: default_review_rounds(),
            subagent_pool: SubagentPoolConfig::default(),
            run_budget: RunBudgetConfig::default(),
            first_round_catalog: FirstRoundCatalog::default(),
            first_round_promote_on: FirstRoundPromoteOn::default(),
        }
    }
}

pub const MAX_SUBAGENT_PROVIDER_SLOTS: usize = 3;
pub const MAX_SUBAGENT_SLOT_ID_CHARS: usize = 80;
pub const MAX_SUBAGENT_SOURCE_ID_CHARS: usize = 160;
pub const MAX_SUBAGENT_MODEL_CHARS: usize = 320;
pub const MAX_SUBAGENT_PROMPT_TEMPLATE_ID_CHARS: usize = 80;
pub const MAX_SUBAGENT_PROMPT_CHARS: usize = 12_000;

/// 一个候选槽引用的执行来源。槽位身份由 `slot_id` 决定，因此来源允许重复。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubagentProviderSource {
    /// R-Code 已配置的 HTTP LLM Provider profile。
    ApiProvider { provider_id: String },
    /// 本机受信任的 Codex CLI。
    CodexCli,
}

/// 一个可独立加权、可独立设定角色 Prompt 的候选槽。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentProviderSlot {
    pub slot_id: String,
    pub source: SubagentProviderSource,
    pub model: String,
    pub weight: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template_id: Option<String>,
    pub prompt: String,
}

/// 子代理候选池。空池表示尚未启用加权路由，保持旧版 delegation_router 行为。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentPoolConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<SubagentProviderSlot>,
}

fn validate_bounded_identifier(field: &str, value: &str, max_chars: usize) -> Result<()> {
    if value.is_empty()
        || value.trim() != value
        || value.contains('\0')
        || value.chars().any(char::is_control)
        || value.chars().count() > max_chars
    {
        return Err(Error::Config(format!(
            "{field} must be trimmed, non-empty, contain no control characters, and be at most {max_chars} characters"
        )));
    }
    Ok(())
}

impl SubagentPoolConfig {
    /// 只验证持久化合同；来源存在性和连通 receipt 由宿主在原子保存时复核。
    pub fn validate(&self) -> Result<()> {
        if self.slots.len() > MAX_SUBAGENT_PROVIDER_SLOTS {
            return Err(Error::Config(format!(
                "orchestration.subagent_pool supports at most {MAX_SUBAGENT_PROVIDER_SLOTS} slots"
            )));
        }
        if self.slots.is_empty() {
            return Ok(());
        }

        let mut slot_ids = HashSet::with_capacity(self.slots.len());
        let mut weight_sum = 0_u16;
        for slot in &self.slots {
            validate_bounded_identifier(
                "orchestration.subagent_pool.slot_id",
                &slot.slot_id,
                MAX_SUBAGENT_SLOT_ID_CHARS,
            )?;
            if !slot_ids.insert(slot.slot_id.as_str()) {
                return Err(Error::Config(format!(
                    "orchestration.subagent_pool contains duplicate slot_id '{}'",
                    slot.slot_id
                )));
            }
            match &slot.source {
                SubagentProviderSource::ApiProvider { provider_id } => {
                    validate_bounded_identifier(
                        "orchestration.subagent_pool.provider_id",
                        provider_id,
                        MAX_SUBAGENT_SOURCE_ID_CHARS,
                    )?;
                }
                SubagentProviderSource::CodexCli => {}
            }
            validate_bounded_identifier(
                "orchestration.subagent_pool.model",
                &slot.model,
                MAX_SUBAGENT_MODEL_CHARS,
            )?;
            if !(1..=100).contains(&slot.weight) {
                return Err(Error::Config(format!(
                    "orchestration.subagent_pool slot '{}' weight must be between 1 and 100",
                    slot.slot_id
                )));
            }
            weight_sum += u16::from(slot.weight);
            if let Some(template_id) = slot.prompt_template_id.as_deref() {
                validate_bounded_identifier(
                    "orchestration.subagent_pool.prompt_template_id",
                    template_id,
                    MAX_SUBAGENT_PROMPT_TEMPLATE_ID_CHARS,
                )?;
            }
            if slot.prompt.trim().is_empty()
                || slot.prompt.contains('\0')
                || slot.prompt.chars().count() > MAX_SUBAGENT_PROMPT_CHARS
            {
                return Err(Error::Config(format!(
                    "orchestration.subagent_pool slot '{}' prompt must be non-empty, contain no NUL, and be at most {MAX_SUBAGENT_PROMPT_CHARS} characters",
                    slot.slot_id
                )));
            }
        }
        if weight_sum != 100 {
            return Err(Error::Config(format!(
                "orchestration.subagent_pool enabled weights must total 100, got {weight_sum}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MainAgentEngine {
    #[default]
    #[serde(alias = "r_code")] // legacy name, read-only
    Native,
    Codex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DelegationRouterMode {
    /// 只有模型显式选择某个执行器时才路由；auto 回退当前 R-Code 引擎。
    Manual,
    /// 简单任务使用 R-Code，复杂任务优先 Codex；Codex 不可用时安全回退。
    #[default]
    Balanced,
    /// 除非明确标注复杂，否则优先 R-Code。
    RCodeFirst,
    /// 除非明确标注简单，否则优先 Codex。
    CodexFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityLoopMode {
    #[default]
    Off,
    /// 仅工具型/工作区任务完成后复核。
    Auto,
    /// 每一轮主回复都复核。
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityReviewer {
    /// 与主执行器交叉复核；不可用时回退内置引擎。
    Auto,
    #[default]
    #[serde(alias = "r_code")] // legacy name, read-only
    Native,
    Codex,
}

/// MCP server 规格（config 自有版本，不依赖 agent-mcp）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerSpec {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

// ── 默认值函数 ────────────────────────────────────────────────

fn default_provider() -> String {
    "anthropic".into()
}
fn default_log_level() -> String {
    "info".into()
}
fn default_compaction_strategy() -> String {
    "auto".into()
}
fn default_max_context() -> u32 {
    180_000
}
fn default_trigger() -> f64 {
    0.8
}
fn default_theme() -> String {
    "system".into()
}
fn default_font_size() -> u32 {
    13
}
fn default_true() -> bool {
    true
}
fn default_review_rounds() -> u8 {
    1
}

fn default_storage() -> StorageConfig {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let base = home.join(".hermes");
    StorageConfig {
        base_dir: base.join("data"),
        sessions_dir: base.join("sessions"),
        skills_dir: base.join("skills"),
        memories_dir: base.join("memories"),
    }
}

fn default_compaction() -> CompactionConfig {
    CompactionConfig {
        strategy: default_compaction_strategy(),
        max_context_tokens: default_max_context(),
        trigger_threshold: default_trigger(),
    }
}

impl Default for Config {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let base = home.join(".hermes");
        Self {
            default_provider: default_provider(),
            log_level: default_log_level(),
            providers: HashMap::new(),
            mcp_servers: HashMap::new(),
            storage: StorageConfig {
                base_dir: base.join("data"),
                sessions_dir: base.join("sessions"),
                skills_dir: base.join("skills"),
                memories_dir: base.join("memories"),
            },
            compaction: CompactionConfig {
                strategy: default_compaction_strategy(),
                max_context_tokens: default_max_context(),
                trigger_threshold: default_trigger(),
            },
            orchestration: OrchestrationConfig::default(),
            planning: PlanningConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            image_understanding: ImageUnderstandingConfig::default(),
            tauri: None,
            schema_version: default_schema_version(),
        }
    }
}

impl Config {
    /// 加载配置（优先级：默认值 < 文件 < 环境变量）。
    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if path.exists() {
            Self::load_from(&path)
        } else {
            let mut config = Self::default();
            Self::apply_env(&mut config);
            config.validate()?;
            Ok(config)
        }
    }

    /// 从指定路径加载。
    pub fn load_from(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).map_err(|_| Error::ConfigNotFound(path.to_path_buf()))?;
        let mut config: Config = toml::from_str(&content).map_err(Error::Toml)?;
        Self::apply_env(&mut config);
        config.validate()?;
        Ok(config)
    }

    /// 环境变量覆盖（ANTHROPIC_API_KEY 等）。
    fn apply_env(config: &mut Config) {
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            config
                .providers
                .entry("anthropic".into())
                .and_modify(|p| p.api_key = key.clone());
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config
                .providers
                .entry("openai".into())
                .and_modify(|p| p.api_key = key.clone());
        }
    }

    /// 校验。
    pub fn validate(&self) -> Result<()> {
        if !(1..=3).contains(&self.orchestration.max_review_rounds) {
            return Err(Error::Config(
                "orchestration.max_review_rounds must be between 1 and 3".to_string(),
            ));
        }
        self.orchestration.run_budget.validate()?;
        self.orchestration.subagent_pool.validate()?;
        if self.image_understanding.engine == ImageUnderstandingEngine::Model {
            let provider = self
                .image_understanding
                .model_provider
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    Error::Config(
                        "image_understanding.engine = \"model\" requires model_provider"
                            .to_string(),
                    )
                })?;
            if !self.providers.contains_key(provider) {
                return Err(Error::Config(format!(
                    "image_understanding.model_provider '{provider}' is not configured"
                )));
            }
            if self
                .image_understanding
                .model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                return Err(Error::Config(
                    "image_understanding.engine = \"model\" requires model".to_string(),
                ));
            }
        }
        if !self.providers.contains_key(&self.default_provider) {
            return Err(Error::Config(format!(
                "default provider '{}' not configured",
                self.default_provider
            )));
        }
        for (name, provider) in &self.providers {
            if provider.api_key.is_empty() {
                return Err(Error::Config(format!(
                    "provider '{}' has empty api_key",
                    name
                )));
            }
        }
        // 确保存储目录可写（不存在则创建）
        if !self.storage.base_dir.exists() {
            std::fs::create_dir_all(&self.storage.base_dir)?;
        }
        Ok(())
    }

    fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hermes/config.toml")
    }

    /// 获取指定 provider 配置。
    pub fn provider(&self, name: &str) -> Result<&ProviderConfig> {
        self.providers
            .get(name)
            .ok_or_else(|| Error::ProviderNotFound(name.to_string()))
    }

    /// 获取默认 provider 配置。
    pub fn default_provider_config(&self) -> Result<&ProviderConfig> {
        self.provider(&self.default_provider)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    fn image_ready_config() -> Config {
        let mut config = Config::default();
        config.providers.insert(
            "openai".to_string(),
            ProviderConfig {
                base_url: "https://api.openai.com/v1".to_string(),
                api_key: "sk-test".to_string(),
                model: "gpt-5.5".to_string(),
                provider_kind: Some("openai".to_string()),
                max_tokens: None,
                temperature: None,
                protocol: None,
                show_reasoning: true,
            },
        );
        config.default_provider = "openai".to_string();
        config
    }

    #[test]
    fn image_understanding_defaults_to_ocr_and_migrates_legacy_configs() {
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(
            config.image_understanding.engine,
            ImageUnderstandingEngine::Ocr
        );
        assert_eq!(config.image_understanding.model_provider, None);
        assert_eq!(
            config.image_understanding,
            ImageUnderstandingConfig::default()
        );
        // 旧配置（无该段）可通过 validate。
        let config = image_ready_config();
        assert!(config.validate().is_ok());
        // snake_case 往返。
        let stored: Config = toml::from_str(
            "[image_understanding]
engine = \"model\"
model_provider = \"openai\"
model = \"gpt-5.5\"
",
        )
        .unwrap();
        assert_eq!(
            stored.image_understanding.engine,
            ImageUnderstandingEngine::Model
        );
        assert_eq!(
            stored.image_understanding.model_provider.as_deref(),
            Some("openai")
        );
        assert_eq!(stored.image_understanding.model.as_deref(), Some("gpt-5.5"));
    }

    #[test]
    fn image_understanding_model_engine_requires_provider_and_model() {
        let mut config = image_ready_config();
        config.image_understanding.engine = ImageUnderstandingEngine::Model;
        // 缺 provider / model 都必须被拒绝。
        let missing_all = config.validate().unwrap_err().to_string();
        assert!(missing_all.contains("model_provider"), "{missing_all}");
        config.image_understanding.model_provider = Some("nonexistent".to_string());
        let missing_provider = config.validate().unwrap_err().to_string();
        assert!(
            missing_provider.contains("not configured"),
            "{missing_provider}"
        );
        config.image_understanding.model_provider = Some("openai".to_string());
        config.image_understanding.model = Some("  ".to_string());
        let missing_model = config.validate().unwrap_err().to_string();
        assert!(missing_model.contains("requires model"), "{missing_model}");
        config.image_understanding.model = Some("gpt-5.5".to_string());
        assert!(config.validate().is_ok());
    }

    fn subagent_slot(slot_id: &str, provider_id: &str, weight: u8) -> SubagentProviderSlot {
        SubagentProviderSlot {
            slot_id: slot_id.to_string(),
            source: SubagentProviderSource::ApiProvider {
                provider_id: provider_id.to_string(),
            },
            model: "test-model".to_string(),
            weight,
            prompt_template_id: Some("implementation".to_string()),
            prompt: "Implement the delegated feature and report verification evidence.".to_string(),
        }
    }

    #[test]
    fn first_round_catalog_and_promote_on_default_to_current_behavior() {
        // C1：旧配置（无两个新键）读回 Full/Either 即现状行为；显式实验值
        // snake_case 往返不报错。
        let legacy: OrchestrationConfig = toml::from_str("").unwrap();
        assert_eq!(legacy.first_round_catalog, FirstRoundCatalog::Full);
        assert_eq!(legacy.first_round_promote_on, FirstRoundPromoteOn::Either);
        let anchored: OrchestrationConfig = toml::from_str(
            r#"
first_round_catalog = "readonly"
first_round_promote_on = "tool_call"
"#,
        )
        .unwrap();
        assert_eq!(anchored.first_round_catalog, FirstRoundCatalog::ReadOnly);
        assert_eq!(
            anchored.first_round_promote_on,
            FirstRoundPromoteOn::ToolCall
        );
        assert_eq!(
            toml::to_string(&anchored).unwrap(),
            toml::to_string(&OrchestrationConfig {
                first_round_catalog: FirstRoundCatalog::ReadOnly,
                first_round_promote_on: FirstRoundPromoteOn::ToolCall,
                ..OrchestrationConfig::default()
            })
            .unwrap()
        );
        // editor_pair 变体可解析。
        let editor: OrchestrationConfig =
            toml::from_str("first_round_catalog = \"editor_pair\"").unwrap();
        assert_eq!(editor.first_round_catalog, FirstRoundCatalog::EditorPair);
        // 规划门档位与 plan_complete 晋升信号 snake_case 往返。
        let plan_gate: OrchestrationConfig = toml::from_str(
            r#"
first_round_catalog = "plan_gate"
first_round_promote_on = "plan_complete"
"#,
        )
        .unwrap();
        assert_eq!(plan_gate.first_round_catalog, FirstRoundCatalog::PlanGate);
        assert_eq!(
            plan_gate.first_round_promote_on,
            FirstRoundPromoteOn::PlanComplete
        );
        assert_eq!(
            toml::to_string(&plan_gate).unwrap(),
            toml::to_string(&OrchestrationConfig {
                first_round_catalog: FirstRoundCatalog::PlanGate,
                first_round_promote_on: FirstRoundPromoteOn::PlanComplete,
                ..OrchestrationConfig::default()
            })
            .unwrap()
        );
        // 非法值报配置错误（不静默回退默认）。
        assert!(
            toml::from_str::<OrchestrationConfig>("first_round_catalog = \"minimal\"").is_err()
        );
        assert!(
            toml::from_str::<OrchestrationConfig>("first_round_catalog = \"planGate\"").is_err()
        );
        assert!(
            toml::from_str::<OrchestrationConfig>("first_round_promote_on = \"plan_ready\"")
                .is_err()
        );
    }

    #[test]
    fn diagnostics_defaults_to_disabled_and_parses_opt_in() {
        // A3.2：旧配置（无 [diagnostics] 段）读回全默认；显式开启可 Round-trip。
        let legacy: Config = toml::from_str("default_provider = \"x\"").unwrap();
        assert!(!legacy.diagnostics.request_audit);
        let enabled: Config = toml::from_str(
            r#"
default_provider = "x"
[diagnostics]
request_audit = true
"#,
        )
        .unwrap();
        assert!(enabled.diagnostics.request_audit);
        let roundtrip = toml::to_string(&enabled).unwrap();
        assert!(roundtrip.contains("request_audit = true"));
    }

    #[test]
    fn legacy_orchestration_defaults_to_an_empty_subagent_pool() {
        // legacy name, read-only：default_agent_engine = "r_code" 是旧配置键值，
        // 读取别名必须继续映射到 Native。
        let orchestration: OrchestrationConfig = toml::from_str(
            r#"
default_agent_engine = "r_code"
delegation_router = "balanced"
allow_cross_engine_delegation = true
quality_loop = "off"
quality_reviewer = "r_code"
max_review_rounds = 1
"#,
        )
        .unwrap();
        assert!(orchestration.subagent_pool.slots.is_empty());
        assert_eq!(
            orchestration.delegation_router,
            DelegationRouterMode::Balanced
        );
        assert_eq!(orchestration.default_agent_engine, MainAgentEngine::Native);
        assert_eq!(orchestration.quality_reviewer, QualityReviewer::Native);
    }

    #[test]
    fn repeated_provider_sources_roundtrip_as_distinct_slots() {
        let pool = SubagentPoolConfig {
            slots: vec![
                subagent_slot("implementer", "openai", 60),
                subagent_slot("reviewer", "openai", 40),
            ],
        };
        pool.validate().unwrap();

        let encoded = toml::to_string(&pool).unwrap();
        let decoded: SubagentPoolConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, pool);
        assert!(matches!(
            decoded.slots[0].source,
            SubagentProviderSource::ApiProvider { ref provider_id } if provider_id == "openai"
        ));
        assert!(matches!(
            decoded.slots[1].source,
            SubagentProviderSource::ApiProvider { ref provider_id } if provider_id == "openai"
        ));
    }

    #[test]
    fn one_and_three_slot_pools_accept_api_and_codex_sources() {
        let mut codex = subagent_slot("codex", "unused", 100);
        codex.source = SubagentProviderSource::CodexCli;
        let single = SubagentPoolConfig { slots: vec![codex] };
        single.validate().unwrap();

        let api = subagent_slot("api", "openai", 34);
        let mut codex = subagent_slot("codex", "unused", 33);
        codex.source = SubagentProviderSource::CodexCli;
        let second_api = subagent_slot("second-api", "anthropic", 33);
        let pool = SubagentPoolConfig {
            slots: vec![api, codex, second_api],
        };
        pool.validate().unwrap();

        let encoded = toml::to_string(&pool).unwrap();
        let decoded: SubagentPoolConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, pool);
    }

    #[test]
    fn subagent_pool_rejects_duplicate_slot_ids_and_a_fourth_slot() {
        let duplicate = SubagentPoolConfig {
            slots: vec![
                subagent_slot("same", "openai", 50),
                subagent_slot("same", "anthropic", 50),
            ],
        };
        assert!(duplicate
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate slot_id 'same'"));

        let four = SubagentPoolConfig {
            slots: vec![
                subagent_slot("one", "openai", 25),
                subagent_slot("two", "openai", 25),
                subagent_slot("three", "openai", 25),
                subagent_slot("four", "openai", 25),
            ],
        };
        assert!(four
            .validate()
            .unwrap_err()
            .to_string()
            .contains("supports at most 3 slots"));
    }

    #[test]
    fn subagent_pool_rejects_zero_and_non_hundred_weight_totals() {
        let zero = SubagentPoolConfig {
            slots: vec![subagent_slot("zero", "openai", 0)],
        };
        assert!(zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("weight must be between 1 and 100"));

        let under = SubagentPoolConfig {
            slots: vec![subagent_slot("weighted", "openai", 99)],
        };
        assert!(under
            .validate()
            .unwrap_err()
            .to_string()
            .contains("weights must total 100, got 99"));

        let over = SubagentPoolConfig {
            slots: vec![
                subagent_slot("first", "openai", 50),
                subagent_slot("second", "openai", 51),
            ],
        };
        assert!(over
            .validate()
            .unwrap_err()
            .to_string()
            .contains("weights must total 100, got 101"));
    }

    #[test]
    fn subagent_pool_rejects_invalid_prompt_and_identifiers() {
        let max_slot_id = "s".repeat(MAX_SUBAGENT_SLOT_ID_CHARS);
        let max_provider_id = "p".repeat(MAX_SUBAGENT_SOURCE_ID_CHARS);
        let mut boundary = subagent_slot(&max_slot_id, &max_provider_id, 100);
        boundary.model = "m".repeat(MAX_SUBAGENT_MODEL_CHARS);
        boundary.prompt_template_id = Some("t".repeat(MAX_SUBAGENT_PROMPT_TEMPLATE_ID_CHARS));
        boundary.prompt = "x".repeat(MAX_SUBAGENT_PROMPT_CHARS);
        SubagentPoolConfig {
            slots: vec![boundary],
        }
        .validate()
        .unwrap();

        let mut empty_prompt = subagent_slot("prompt", "openai", 100);
        empty_prompt.prompt = "  \n".to_string();
        let error = SubagentPoolConfig {
            slots: vec![empty_prompt],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("prompt must be non-empty"));

        let mut nul_prompt = subagent_slot("prompt", "openai", 100);
        nul_prompt.prompt = "before\0after".to_string();
        let error = SubagentPoolConfig {
            slots: vec![nul_prompt],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("contain no NUL"));

        let mut oversized_prompt = subagent_slot("prompt", "openai", 100);
        oversized_prompt.prompt = "x".repeat(MAX_SUBAGENT_PROMPT_CHARS + 1);
        let error = SubagentPoolConfig {
            slots: vec![oversized_prompt],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("at most 12000 characters"));

        let mut padded_id = subagent_slot(" padded ", "openai", 100);
        padded_id.prompt_template_id = None;
        let error = SubagentPoolConfig {
            slots: vec![padded_id],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("subagent_pool.slot_id"));

        let mut invalid_provider = subagent_slot("provider", "open\nai", 100);
        invalid_provider.prompt_template_id = None;
        let error = SubagentPoolConfig {
            slots: vec![invalid_provider],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("subagent_pool.provider_id"));

        let mut invalid_model = subagent_slot("model", "openai", 100);
        invalid_model.model = "m".repeat(MAX_SUBAGENT_MODEL_CHARS + 1);
        let error = SubagentPoolConfig {
            slots: vec![invalid_model],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("subagent_pool.model"));

        let mut invalid_template = subagent_slot("template", "openai", 100);
        invalid_template.prompt_template_id = Some("bad\0template".to_string());
        let error = SubagentPoolConfig {
            slots: vec![invalid_template],
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("subagent_pool.prompt_template_id"));
    }

    // 环境变量是进程级全局状态，多个测试并行操作会竞态；用锁串行化。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write_temp_config(content: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        (dir, path)
    }

    #[test]
    fn toml_roundtrip() {
        let toml_str = r#"
default_provider = "anthropic"
log_level = "debug"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "sk-ant-xxx"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.log_level, "debug");
        assert_eq!(
            config.providers.get("anthropic").unwrap().model,
            "claude-sonnet-4"
        );
    }

    #[test]
    fn quality_reviewer_defaults_to_native_without_overwriting_an_explicit_choice() {
        assert_eq!(
            OrchestrationConfig::default().quality_reviewer,
            QualityReviewer::Native
        );

        let legacy: OrchestrationConfig = toml::from_str(r#"quality_loop = "auto""#).unwrap();
        assert_eq!(legacy.quality_reviewer, QualityReviewer::Native);

        let explicit: OrchestrationConfig = toml::from_str(r#"quality_reviewer = "auto""#).unwrap();
        assert_eq!(explicit.quality_reviewer, QualityReviewer::Auto);
    }

    #[test]
    fn v_cfg_01_env_overrides_file() {
        // V-CFG-01：默认值 < 文件 < 环境变量
        let _guard = ENV_LOCK.lock().unwrap();
        let toml_str = r#"
default_provider = "anthropic"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "from_file"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);

        std::env::set_var("ANTHROPIC_API_KEY", "from_env");
        let config = Config::load_from(&path).unwrap();
        std::env::remove_var("ANTHROPIC_API_KEY");

        assert_eq!(
            config.providers.get("anthropic").unwrap().api_key,
            "from_env"
        );
    }

    #[test]
    fn v_cfg_01_file_overrides_default_when_no_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let toml_str = r#"
default_provider = "anthropic"

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "file_value"
model = "claude-sonnet-4"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        std::env::remove_var("ANTHROPIC_API_KEY");
        let config = Config::load_from(&path).unwrap();
        assert_eq!(
            config.providers.get("anthropic").unwrap().api_key,
            "file_value"
        );
    }

    #[test]
    fn v_cfg_02_debug_does_not_leak_api_key() {
        // V-CFG-02：Debug 输出不含 api_key
        let config = ProviderConfig {
            base_url: "https://api.anthropic.com".into(),
            api_key: "sk-secret-12345".into(),
            model: "claude".into(),
            max_tokens: None,
            temperature: None,
            protocol: None,
            provider_kind: None,
            show_reasoning: true,
        };
        let dbg = format!("{:?}", config);
        assert!(!dbg.contains("sk-secret-12345"), "api_key leaked: {dbg}");
        assert!(dbg.contains("***"));
    }

    #[test]
    fn provider_kind_roundtrips_while_legacy_configs_default_to_none() {
        let with_identity: ProviderConfig = toml::from_str(
            r#"
base_url = "https://relay.example/v1"
api_key = "secret"
model = "deepseek-v4-flash"
provider_kind = "deepseek"
"#,
        )
        .unwrap();
        assert_eq!(with_identity.provider_kind.as_deref(), Some("deepseek"));
        let persisted = toml::to_string(&with_identity).unwrap();
        assert!(persisted.contains("provider_kind = \"deepseek\""));

        let legacy: ProviderConfig = toml::from_str(
            r#"
base_url = "https://api.deepseek.com"
api_key = "secret"
model = "deepseek-v4-pro"
"#,
        )
        .unwrap();
        assert_eq!(legacy.provider_kind, None);
        assert!(
            legacy.show_reasoning,
            "legacy provider configs should show reasoning by default"
        );
        assert!(
            !toml::to_string(&legacy).unwrap().contains("show_reasoning"),
            "the default value should not add config noise"
        );
    }

    #[test]
    fn provider_reasoning_visibility_false_roundtrips() {
        let hidden: ProviderConfig = toml::from_str(
            r#"
base_url = "https://api.example.com"
api_key = "secret"
model = "reasoning-model"
show_reasoning = false
"#,
        )
        .unwrap();
        assert!(!hidden.show_reasoning);

        let persisted = toml::to_string(&hidden).unwrap();
        assert!(persisted.contains("show_reasoning = false"));
        let reloaded: ProviderConfig = toml::from_str(&persisted).unwrap();
        assert!(!reloaded.show_reasoning);
    }

    #[test]
    fn validate_rejects_missing_default_provider() {
        let mut config = Config::default();
        config.default_provider = "nonexistent".into();
        config.providers.insert(
            "other".into(),
            ProviderConfig {
                base_url: "x".into(),
                api_key: "k".into(),
                model: "m".into(),
                max_tokens: None,
                temperature: None,
                protocol: None,
                provider_kind: None,
                show_reasoning: true,
            },
        );
        let r = config.validate();
        assert!(r.is_err());
    }

    #[test]
    fn validate_rejects_empty_api_key() {
        let mut config = Config::default();
        config.default_provider = "anthropic".into();
        config.providers.insert(
            "anthropic".into(),
            ProviderConfig {
                base_url: "x".into(),
                api_key: "".into(),
                model: "m".into(),
                max_tokens: None,
                temperature: None,
                protocol: None,
                provider_kind: None,
                show_reasoning: true,
            },
        );
        let r = config.validate();
        assert!(r.is_err());
    }

    #[test]
    fn unknown_field_ignored() {
        let toml_str = r#"
default_provider = "anthropic"
future_unknown_field = 123

[providers.anthropic]
base_url = "x"
api_key = "k"
model = "m"

[storage]
base_dir = "/tmp/h"
sessions_dir = "/tmp/h/s"
skills_dir = "/tmp/h/k"
memories_dir = "/tmp/h/m"
"#;
        let (_d, path) = write_temp_config(toml_str);
        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.providers.len(), 1);
    }

    #[test]
    fn default_values_applied() {
        let config = Config::default();
        assert_eq!(config.default_provider, "anthropic");
        assert_eq!(config.log_level, "info");
        assert_eq!(config.compaction.max_context_tokens, 180_000);
        assert!((config.compaction.trigger_threshold - 0.8).abs() < 1e-9);
        assert_eq!(config.orchestration.quality_loop, QualityLoopMode::Off);
    }
}
