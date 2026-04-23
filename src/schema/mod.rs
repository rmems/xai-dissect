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
