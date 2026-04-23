// SPDX-License-Identifier: GPL-3.0-only
//
// Core schema types used across the parser, inventory, analysis, and export
// layers. Any type that is serialized to an export artifact (JSON / CSV /
// Markdown) must live here, must derive `Serialize`, and must be stable
// across patch releases.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Numpy-style dtype of a tensor as it lives on disk. The set is intentionally
/// narrow: Grok-1 shards only contain `float32` and `int8`. New dtypes are
/// added only when a supported checkpoint actually requires them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    F32,
    I8,
}

impl TensorDType {
    pub fn itemsize(self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::I8 => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TensorDType::F32 => "f32",
            TensorDType::I8 => "int8",
        }
    }
}

/// Row-major tensor shape. Dimensions are stored left-to-right exactly as
/// they appear in the on-disk shape tuple.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct TensorShape(pub Vec<u64>);

impl TensorShape {
    pub fn new(dims: Vec<u64>) -> Self {
        Self(dims)
    }

    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> u64 {
        self.0
            .iter()
            .copied()
            .fold(1u64, |acc, d| acc.saturating_mul(d))
    }

    pub fn is_empty_tuple(&self) -> bool {
        self.0.is_empty()
    }

    pub fn render(&self) -> String {
        match self.0.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.0[0]),
            _ => format!(
                "({})",
                self.0
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// Parser-level tag: how a tensor appears inside a pickle shard. Orthogonal
/// to semantic classification (`TensorKind`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    /// A bare `numpy.ndarray` reduced into the pickle stream.
    Tensor,
    /// The int8 weight half of a `QuantizedWeight8bit` dataclass.
    QuantWeight,
    /// The f32 scales half of a `QuantizedWeight8bit` dataclass.
    QuantScales,
}

impl TensorRole {
    pub fn label(self) -> &'static str {
        match self {
            TensorRole::Tensor => "tensor",
            TensorRole::QuantWeight => "quant.weight",
            TensorRole::QuantScales => "quant.scales",
        }
    }
}

/// Semantic classification inferred from shape, dtype, and position.
/// This is a *heuristic* assignment. It is stable for well-formed Grok-1
/// checkpoints and falls back to `Unknown` elsewhere. See
/// `docs/tensor-schema.md` for the rules.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum TensorKind {
    /// Token embedding table, shape `(vocab_size, d_model)`.
    TokenEmbedding,
    /// Final pre-head RMSNorm (or equivalent), shape `(d_model,)`.
    FinalNorm,
    /// Per-block RMSNorm, shape `(d_model,)`. Position within the block is
    /// tracked by `TensorInfo::block_slot`.
    BlockNorm,
    /// MoE router / gate table, shape `(d_model, n_experts)`.
    Router,
    /// One of the MoE expert feed-forward projections, quantized. The
    /// specific projection (`up` / `gate` / `down`) cannot always be
    /// distinguished by shape alone; when it cannot, the projection is
    /// reported as `Unresolved`.
    MoeExpertProjection { projection: MoeProjection },
    /// Companion f32 scales tensor for an MoE expert projection (where it
    /// lives outside the `QuantizedWeight8bit` envelope).
    MoeScales,
    /// Attention projection stored directly as f32 (not quantized). Exact
    /// role (q / k / v / out) is not determined at cartography time.
    AttnProjF32,
    /// Weight tensor that does not fit any of the above signatures.
    Unknown { reason: String },
}

impl TensorKind {
    pub fn short_label(&self) -> String {
        match self {
            TensorKind::TokenEmbedding => "token_embedding".into(),
            TensorKind::FinalNorm => "final_norm".into(),
            TensorKind::BlockNorm => "block_norm".into(),
            TensorKind::Router => "router".into(),
            TensorKind::MoeExpertProjection { projection } => {
                format!("moe_expert.{}", projection.label())
            }
            TensorKind::MoeScales => "moe_scales".into(),
            TensorKind::AttnProjF32 => "attn_proj_f32".into(),
            TensorKind::Unknown { .. } => "unknown".into(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeProjection {
    Up,
    Gate,
    Down,
    /// Gate/up cannot be told apart by shape alone on Grok-1; both have the
    /// same `(n_experts, d_model, d_ff)` signature. The inventory layer
    /// emits `Unresolved` for those and leaves disambiguation to a later
    /// analysis pass that inspects ordering within a block.
    Unresolved,
}

impl MoeProjection {
    pub fn label(self) -> &'static str {
        match self {
            MoeProjection::Up => "up",
            MoeProjection::Gate => "gate",
            MoeProjection::Down => "down",
            MoeProjection::Unresolved => "unresolved",
        }
    }
}

/// A single tensor as it appears in a shard, plus all metadata needed for
/// downstream analyzers. One shard may produce one or two `TensorInfo`
/// records (two for `QuantizedWeight8bit`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Absolute path of the shard file on disk.
    pub shard_path: PathBuf,
    /// 0-based ordinal of the shard in the checkpoint's sorted shard list.
    pub shard_ordinal: u32,
    /// 0-based index within the shard (in the order tensors were found).
    pub in_shard_index: u32,
    pub role: TensorRole,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    /// Byte offset of the raw payload within the shard file.
    pub offset: u64,
    /// Payload length in bytes.
    pub nbytes: u64,
    /// Inferred semantic classification.
    pub kind: TensorKind,
    /// Inferred block (transformer layer) index, if assignable.
    pub block_index: Option<u32>,
    /// Position within the block's shard ordering, if assignable.
    /// Useful for disambiguating multiple tensors of the same kind inside
    /// one block (e.g. the two `AttnProjF32` tensors per layer).
    pub block_slot: Option<u32>,
}

/// Aggregate summary of one transformer block (or the non-block singletons:
/// the embedding and the final norm).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockSummary {
    /// `None` for the embedding and the final norm; `Some(i)` for block `i`.
    pub block_index: Option<u32>,
    pub label: String,
    /// Shard ordinals in this block.
    pub shard_range: Option<ShardRange>,
    pub tensor_count: u32,
    pub total_nbytes: u64,
    pub dtypes: Vec<TensorDType>,
    pub kinds: Vec<KindCount>,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ShardRange {
    pub start: u32,
    pub end_inclusive: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KindCount {
    pub kind_label: String,
    pub count: u32,
    pub nbytes: u64,
}

/// Top-level, serializable inventory of a checkpoint directory. The JSON
/// form of this struct is the stable export contract consumed by sibling
/// repos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInventory {
    /// Short identifier of the target family. Currently only `"grok-1"` is
    /// emitted; `"grok-2"` is reserved for a future release.
    pub model_family: String,
    /// Absolute path of the checkpoint directory that was scanned.
    pub checkpoint_path: PathBuf,
    /// Total number of shard files inspected.
    pub shard_count: u32,
    /// Hyperparameters inferred from the inventory itself.
    pub inferred: InferredHyperparams,
    pub tensors: Vec<TensorInfo>,
    pub blocks: Vec<BlockSummary>,
    pub totals: InventoryTotals,
    /// Schema version. Bump on incompatible export changes.
    pub schema_version: u32,
}

/// Top-level, serializable expert-level view of a checkpoint directory.
/// This is derived entirely from `ModelInventory` and never requires a
/// forward pass or tensor-body reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertAtlas {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub shard_count: u32,
    pub inferred: InferredHyperparams,
    pub relevant_block_count: u32,
    pub expected_experts_per_block: Option<u64>,
    pub blocks: Vec<ExpertBlock>,
    pub naming_patterns: Vec<ExpertNamingPattern>,
    pub naming_checks: Vec<ExpertNamingCheck>,
    pub anomalies: Vec<ExpertIssue>,
    pub schema_version: u32,
}

/// Expert-oriented summary of one transformer block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertBlock {
    pub block_index: u32,
    pub shard_range: Option<ShardRange>,
    pub expert_count: Option<u64>,
    pub tensors: Vec<ExpertTensorRef>,
    pub experts: Vec<ExpertSlice>,
}

/// One tensor family that participates in the block's expert layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertTensorRef {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_slot: Option<u32>,
    pub role: TensorRole,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub kind_label: String,
    pub projection: MoeProjection,
    pub expert_axis: Option<u32>,
    pub expert_count: Option<u64>,
    pub family_label: String,
    pub structural_name: String,
}

/// One expert index within a block, associated to the tensor families that
/// carry expert-stacked parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertSlice {
    pub expert_index: u32,
    pub tensors: Vec<ExpertSliceTensor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertSliceTensor {
    pub family_label: String,
    pub structural_name: String,
    pub source_shard_ordinal: u32,
    pub source_in_shard_index: u32,
    pub source_block_slot: Option<u32>,
    pub projection: MoeProjection,
    pub dtype: TensorDType,
    pub slice_shape: TensorShape,
}

/// Aggregate structural naming pattern inferred from the block slot layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertNamingPattern {
    pub family_label: String,
    pub pattern: String,
    pub projection: MoeProjection,
    pub block_slots: Vec<u32>,
    pub observed_shapes: Vec<TensorShape>,
    pub observed_blocks: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertNamingCheck {
    pub check: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertIssue {
    pub severity: ExpertIssueSeverity,
    pub category: ExpertIssueCategory,
    pub block_index: Option<u32>,
    pub tensor: Option<ExpertTensorLocator>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertTensorLocator {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_slot: Option<u32>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertIssueSeverity {
    Warning,
    Error,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertIssueCategory {
    MissingOrIrregularTensor,
    NamingConsistency,
    LayoutAnomaly,
}

/// Top-level, serializable routing-oriented view of a checkpoint directory.
/// This is derived from `ModelInventory` and remains structural only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingReport {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub shard_count: u32,
    pub inferred: InferredHyperparams,
    pub relevant_block_count: u32,
    pub expected_experts_per_router: Option<u64>,
    pub candidate_tensors: Vec<RoutingTensorRef>,
    pub blocks: Vec<RoutingBlockReport>,
    pub orientation_summaries: Vec<RoutingOrientationSummary>,
    pub likely_routing_critical_blocks: Vec<RoutingCriticalBlock>,
    pub grok_layout_notes: Vec<String>,
    pub anomalies: Vec<RoutingIssue>,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingBlockReport {
    pub block_index: Option<u32>,
    pub label: String,
    pub shard_range: Option<ShardRange>,
    pub local_expert_count: Option<u64>,
    pub primary_candidate: Option<RoutingTensorLocator>,
    pub candidates: Vec<RoutingTensorRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingTensorRef {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_index: Option<u32>,
    pub block_slot: Option<u32>,
    pub role: TensorRole,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub kind_label: String,
    pub orientation: RoutingOrientation,
    pub expert_axis: Option<u32>,
    pub linked_expert_count: Option<u64>,
    pub matches_inferred_expert_count: bool,
    pub structural_name: String,
    pub gate_metrics: RoutingGateMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingGateMetrics {
    pub total_elements: u64,
    pub total_nbytes: u64,
    pub input_width: Option<u64>,
    pub output_width: Option<u64>,
    pub expert_count: Option<u64>,
    pub logits_per_input: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingOrientationSummary {
    pub orientation: RoutingOrientation,
    pub count: u32,
    pub observed_shapes: Vec<TensorShape>,
    pub observed_blocks: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingCriticalBlock {
    pub block_index: Option<u32>,
    pub label: String,
    pub reason: String,
    pub primary_candidate: Option<RoutingTensorLocator>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingOrientation {
    DModelToExperts,
    ExpertsToDModel,
    ExpertAxisLeading,
    ExpertAxisTrailing,
    Ambiguous,
    Unknown,
}

impl RoutingOrientation {
    pub fn label(self) -> &'static str {
        match self {
            RoutingOrientation::DModelToExperts => "d_model_to_experts",
            RoutingOrientation::ExpertsToDModel => "experts_to_d_model",
            RoutingOrientation::ExpertAxisLeading => "expert_axis_leading",
            RoutingOrientation::ExpertAxisTrailing => "expert_axis_trailing",
            RoutingOrientation::Ambiguous => "ambiguous",
            RoutingOrientation::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingIssue {
    pub severity: RoutingIssueSeverity,
    pub category: RoutingIssueCategory,
    pub block_index: Option<u32>,
    pub tensor: Option<RoutingTensorLocator>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingTensorLocator {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_slot: Option<u32>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingIssueSeverity {
    Warning,
    Error,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingIssueCategory {
    MissingCandidate,
    ShapeSummary,
    ExpertCountLinkage,
    LayoutNote,
}

/// Top-level, serializable tensor-statistics profile. Unlike the inventory,
/// this document is allowed to summarize sampled tensor payload values, but
/// it remains offline analysis only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatsProfileReport {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub shard_count: u32,
    pub inferred: InferredHyperparams,
    pub sampling: StatsSamplingConfig,
    pub tensors: Vec<TensorStats>,
    pub layers: Vec<LayerStats>,
    pub norm_summary: NormSummary,
    pub variance_summary: VarianceSummary,
    pub outlier_summary: OutlierSummary,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatsSamplingConfig {
    pub max_sample_values: u64,
    pub f32_near_zero_abs: f64,
    pub i8_near_zero_abs: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorStats {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_index: Option<u32>,
    pub block_slot: Option<u32>,
    pub structural_name: String,
    pub role: TensorRole,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub kind_label: String,
    pub sampled: bool,
    pub total_values: u64,
    pub sample_values: u64,
    pub total_nbytes: u64,
    pub mean: f64,
    pub variance: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub max_abs: f64,
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub rms: f64,
    pub zero_fraction: f64,
    pub near_zero_fraction: f64,
    pub positive_fraction: f64,
    pub negative_fraction: f64,
    pub outlier_fraction: f64,
    pub peak_to_rms: f64,
    pub distribution_label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerStats {
    pub block_index: Option<u32>,
    pub label: String,
    pub tensor_count: u32,
    pub sampled_tensor_count: u32,
    pub total_nbytes: u64,
    pub mean_rms: f64,
    pub mean_variance: f64,
    pub mean_outlier_fraction: f64,
    pub routing_tensor_count: u32,
    pub compressible_candidate_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedTensorStat {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub structural_name: String,
    pub kind_label: String,
    pub block_index: Option<u32>,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormSummary {
    pub mean_rms: f64,
    pub max_rms: Option<RankedTensorStat>,
    pub max_l2: Option<RankedTensorStat>,
    pub top_rms: Vec<RankedTensorStat>,
    pub top_l2: Vec<RankedTensorStat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarianceSummary {
    pub mean_variance: f64,
    pub max_variance: Option<RankedTensorStat>,
    pub min_variance: Option<RankedTensorStat>,
    pub top_variance: Vec<RankedTensorStat>,
    pub lowest_variance: Vec<RankedTensorStat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutlierSummary {
    pub mean_outlier_fraction: f64,
    pub most_outlier_heavy: Vec<RankedTensorStat>,
    pub highest_peak_to_rms: Vec<RankedTensorStat>,
}

/// Top-level, serializable profiling report for future SAAQ experiments.
/// This does not apply SAAQ; it identifies where experiments may be
/// promising or risky.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaaqReadinessReport {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub shard_count: u32,
    pub inferred: InferredHyperparams,
    pub candidate_targets: Vec<SaaqCandidate>,
    pub routing_critical_tensors: Vec<SaaqCandidate>,
    pub risky_tensors: Vec<SaaqCandidate>,
    pub layer_readiness: Vec<SaaqLayerReadiness>,
    pub notes: Vec<String>,
    pub manifest: CandidateTensorManifest,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateTensorManifest {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub candidates: Vec<SaaqCandidate>,
    pub schema_version: u32,
}

/// Compact checkpoint snapshot manifest for downstream tooling that needs
/// inventory-level counts without the full tensor table payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointInventorySnapshot {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub shard_count: u32,
    pub inferred: InferredHyperparams,
    pub total_tensors: u64,
    pub total_nbytes: u64,
    pub blocks: Vec<CheckpointInventoryBlockSnapshot>,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointInventoryBlockSnapshot {
    pub label: String,
    pub block_index: Option<u32>,
    pub shard_range: Option<ShardRange>,
    pub tensor_count: u32,
    pub total_nbytes: u64,
    pub kind_labels: Vec<String>,
}

/// Machine-readable routing-critical tensor list for downstream
/// compression / orchestration tooling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingCriticalTensorManifest {
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub tensors: Vec<RoutingCriticalTensor>,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingCriticalTensor {
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_index: Option<u32>,
    pub block_slot: Option<u32>,
    pub structural_name: String,
    pub kind_label: String,
    pub orientation: RoutingOrientation,
    pub linked_expert_count: Option<u64>,
    pub total_nbytes: u64,
    pub criticality_reason: Option<String>,
}

/// Small machine-readable summary of the main findings from a single
/// analysis artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FindingsSummary {
    pub analysis: String,
    pub model_family: String,
    pub checkpoint_path: PathBuf,
    pub checkpoint_slug: String,
    pub headline: String,
    pub findings: Vec<FindingsSummaryItem>,
    pub schema_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FindingsSummaryItem {
    pub severity: FindingsSeverity,
    pub category: String,
    pub detail: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingsSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaaqCandidate {
    pub rank: u32,
    pub shard_ordinal: u32,
    pub in_shard_index: u32,
    pub block_index: Option<u32>,
    pub block_slot: Option<u32>,
    pub structural_name: String,
    pub kind_label: String,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub region_class: SaaqRegionClass,
    pub disposition: SaaqDisposition,
    pub readiness_score: f64,
    pub opportunity_score: f64,
    pub risk_score: f64,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaaqLayerReadiness {
    pub block_index: Option<u32>,
    pub label: String,
    pub routing_critical: bool,
    pub candidate_target_count: u32,
    pub mean_readiness_score: f64,
    pub max_risk_score: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaaqRegionClass {
    RoutingCritical,
    NormalizationSensitive,
    AlreadyCompressed,
    PotentialCompressionTarget,
    EmbeddingHeavy,
    Unknown,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaaqDisposition {
    Candidate,
    ObserveOnly,
    AvoidForNow,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InferredHyperparams {
    pub vocab_size: Option<u64>,
    pub d_model: Option<u64>,
    pub n_experts: Option<u64>,
    pub d_ff: Option<u64>,
    pub n_blocks: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InventoryTotals {
    pub tensors: u64,
    pub quant_tensors: u64,
    pub f32_tensors: u64,
    pub i8_tensors: u64,
    pub total_nbytes: u64,
    pub total_elements: u64,
}
