// SPDX-License-Identifier: GPL-3.0-only
//
// Inventory layer: walks a checkpoint directory, drives the parser, applies
// shape-based semantic classification, and groups tensors into blocks.
//
// This layer is intentionally conservative. It identifies what can be
// identified from `(rank, dtype, dims)` alone plus an inferred `(d_model,
// vocab_size, n_experts)` triple. Anything ambiguous is reported as
// `Unknown { reason }` or `MoeProjection::Unresolved` and deferred to a
// later analysis pass.
//
// Deeper semantic analysis (routing math, expert-level statistics,
// dequantized parameter accounting) is explicitly *not* done here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::parser::{self, RawTensor};
use crate::schema::{
    BlockSummary, Grok1CoverageCounts, Grok1CoverageManifest, Grok1UnknownSlot,
    InferredHyperparams, InventoryTotals, KindCount, ModelInventory, MoeProjection,
    QuantizedAttentionWidth, ShardRange, TensorDType, TensorInfo, TensorKind, TensorRole,
};

pub const SCHEMA_VERSION: u32 = 2;
pub const GROK1_COVERAGE_SCHEMA_VERSION: u32 = 1;

const GROK1_EXPECTED_BLOCKS: u32 = 64;
const GROK1_EXPECTED_TENSORS: u64 = 770;
const GROK1_EXPECTED_ROUTERS: u64 = 64;
const GROK1_EXPECTED_EXPERT_FAMILIES: u64 = 64 * 3;
const GROK1_D_MODEL: u64 = 6_144;
const GROK1_D_FF: u64 = 32_768;
const GROK1_N_EXPERTS: u64 = 8;

/// Configuration for enumerating and classifying a checkpoint.
#[derive(Clone, Debug)]
pub struct InventoryConfig {
    /// Filename prefix used to select shard files inside the checkpoint
    /// directory. Grok-1 uses `tensor`.
    pub prefix: String,
    /// Hard cap on the number of shards scanned (sorted by filename).
    /// `None` means scan everything.
    pub limit: Option<usize>,
    /// Target model family label written into the export header.
    pub model_family: String,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            prefix: "tensor".to_string(),
            limit: None,
            model_family: "grok-1".to_string(),
        }
    }
}

/// Build a full `ModelInventory` for the checkpoint directory at `path`.
pub fn build_inventory(path: &Path, cfg: &InventoryConfig) -> Result<ModelInventory> {
    let md = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !md.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let shard_selection = collect_shards(path, &cfg.prefix, cfg.limit)?;
    if shard_selection.shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            path.display(),
            cfg.prefix
        );
    }

    // Pass 1: parse every shard into RawTensor records.
    let mut raws_per_shard: Vec<Vec<RawTensor>> = Vec::with_capacity(shard_selection.shards.len());
    for shard in &shard_selection.shards {
        let ts = parser::dissect_shard(shard)
            .with_context(|| format!("parse shard {}", shard.display()))?;
        raws_per_shard.push(ts);
    }

    // Pass 2: infer model hyperparameters from the raw set.
    let hp = infer_hyperparams(&raws_per_shard);

    // Pass 3: classify each raw tensor into a TensorKind.
    let mut tensors: Vec<TensorInfo> = Vec::new();
    for (shard_ordinal, (shard_path, raws)) in shard_selection
        .shards
        .iter()
        .zip(raws_per_shard.iter())
        .enumerate()
    {
        for (in_shard_index, raw) in raws.iter().enumerate() {
            let kind = classify_tensor(raw, &hp);
            tensors.push(TensorInfo {
                shard_path: shard_path.clone(),
                shard_ordinal: shard_ordinal as u32,
                in_shard_index: in_shard_index as u32,
                role: raw.role,
                dtype: raw.dtype,
                shape: raw.shape.clone(),
                offset: raw.offset,
                nbytes: raw.nbytes,
                kind,
                block_index: None,
                block_slot: None,
            });
        }
    }

    // Pass 4: assign block_index / block_slot from shard ordinals, using a
    // Grok-1-shaped layout model when the shard count fits.
    let n_blocks = assign_block_indices_for_scan(
        &mut tensors,
        shard_selection.shards.len(),
        shard_selection.truncated,
    );

    // Pass 5: build block summaries and totals.
    let blocks = summarize_blocks(&tensors);
    let totals = compute_totals(&tensors);

    Ok(ModelInventory {
        model_family: cfg.model_family.clone(),
        checkpoint_path: path.to_path_buf(),
        shard_count: shard_selection.shards.len() as u32,
        inferred: InferredHyperparams { n_blocks, ..hp },
        tensors,
        blocks,
        totals,
        schema_version: SCHEMA_VERSION,
    })
}

/// Validate a complete Grok-1 inventory and emit a deterministic coverage
/// manifest. This is intentionally stricter than generic inventory building:
/// complete Grok-1 manifests must fail closed on missing blocks/tensors,
/// duplicate structural names, unknown/unexpected layouts, or incomplete
/// hyperparameter metadata.
pub fn validate_grok1_complete_manifest(inv: &ModelInventory) -> Result<Grok1CoverageManifest> {
    if inv.model_family != "grok-1" {
        bail!("Grok-1 coverage validation requires model_family 'grok-1'");
    }

    let expected = Grok1CoverageCounts {
        blocks: GROK1_EXPECTED_BLOCKS,
        tensors: GROK1_EXPECTED_TENSORS,
        routers: GROK1_EXPECTED_ROUTERS,
        expert_families: GROK1_EXPECTED_EXPERT_FAMILIES,
        unknown_tensors: 0,
    };
    let discovered = grok1_discovered_counts(inv);
    let unknown_slots = grok1_unknown_slots(inv);
    let checksum = grok1_checksum(inv, &discovered, &unknown_slots);

    let manifest = Grok1CoverageManifest {
        model_family: inv.model_family.clone(),
        schema_version: inv.schema_version,
        coverage_schema_version: GROK1_COVERAGE_SCHEMA_VERSION,
        validation: "pass".to_string(),
        checksum,
        expected: expected.clone(),
        discovered: discovered.clone(),
        unknown_slots,
    };

    let mut errors = Vec::new();
    if inv.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "schema_version {} != expected {}",
            inv.schema_version, SCHEMA_VERSION
        ));
    }
    if inv.inferred.n_blocks != Some(GROK1_EXPECTED_BLOCKS) {
        errors.push(format!(
            "inferred n_blocks {:?} != expected {}",
            inv.inferred.n_blocks, GROK1_EXPECTED_BLOCKS
        ));
    }
    if inv.inferred.d_model != Some(GROK1_D_MODEL) {
        errors.push(format!(
            "inferred d_model {:?} != expected {}",
            inv.inferred.d_model, GROK1_D_MODEL
        ));
    }
    if inv.inferred.d_ff != Some(GROK1_D_FF) {
        errors.push(format!(
            "inferred d_ff {:?} != expected {}",
            inv.inferred.d_ff, GROK1_D_FF
        ));
    }
    if inv.inferred.n_experts != Some(GROK1_N_EXPERTS) {
        errors.push(format!(
            "inferred n_experts {:?} != expected {}",
            inv.inferred.n_experts, GROK1_N_EXPERTS
        ));
    }

    compare_count("blocks", discovered.blocks, expected.blocks, &mut errors);
    compare_count("tensors", discovered.tensors, expected.tensors, &mut errors);
    compare_count("routers", discovered.routers, expected.routers, &mut errors);
    compare_count(
        "expert_families",
        discovered.expert_families,
        expected.expert_families,
        &mut errors,
    );
    compare_count(
        "unknown_tensors",
        discovered.unknown_tensors,
        expected.unknown_tensors,
        &mut errors,
    );

    validate_grok1_blocks(inv, &mut errors);
    validate_unique_source_keys(inv, &mut errors);
    validate_unique_structural_names(inv, &mut errors);
    validate_grok1_expected_slots(inv, &mut errors);

    if !errors.is_empty() {
        bail!(
            "Grok-1 complete manifest validation failed: {}",
            errors.join("; ")
        );
    }

    Ok(manifest)
}

// --- Shard enumeration -----------------------------------------------------

#[derive(Debug)]
struct ShardSelection {
    shards: Vec<PathBuf>,
    truncated: bool,
}

fn collect_shards(path: &Path, prefix: &str, limit: Option<usize>) -> Result<ShardSelection> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", path.display()))?;
        let p = entry.path();
        if p.is_file()
            && p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix))
                .unwrap_or(false)
        {
            shards.push(p);
        }
    }
    shards.sort();
    let truncated = limit.map(|n| shards.len() > n).unwrap_or(false);
    if let Some(n) = limit {
        shards.truncate(n);
    }
    Ok(ShardSelection { shards, truncated })
}

// --- Hyperparameter inference ---------------------------------------------

fn infer_hyperparams(raws_per_shard: &[Vec<RawTensor>]) -> InferredHyperparams {
    let mut hp = InferredHyperparams::default();

    // Flatten once for convenience; cheap, we only hold references.
    let all: Vec<&RawTensor> = raws_per_shard.iter().flat_map(|v| v.iter()).collect();

    // d_model + vocab_size: look for the largest 2-D f32 `tensor` (not
    // quant.*). On Grok-1 this is uniquely the embedding table.
    let mut best_embed: Option<(u64, u64, u64)> = None; // (numel, vocab, d_model)
    for t in &all {
        if !matches!(t.role, TensorRole::Tensor) {
            continue;
        }
        if t.dtype != TensorDType::F32 {
            continue;
        }
        if t.shape.rank() != 2 {
            continue;
        }
        let dims = t.shape.dims();
        let numel = t.shape.numel();
        if best_embed.map(|(n, _, _)| numel > n).unwrap_or(true) {
            best_embed = Some((numel, dims[0], dims[1]));
        }
    }
    if let Some((_, v, d)) = best_embed {
        hp.vocab_size = Some(v);
        hp.d_model = Some(d);
    }

    // n_experts + d_ff: look for the 3-D int8 `quant.weight` tensors. Grok-1
    // emits two distinct 3-D signatures per layer; both agree on `n_experts`
    // (leading dim). Prefer the one where the inner product matches
    // `(d_model, d_ff)`.
    let d_model = hp.d_model;
    for t in &all {
        if !matches!(t.role, TensorRole::QuantWeight) {
            continue;
        }
        if t.dtype != TensorDType::I8 {
            continue;
        }
        if t.shape.rank() != 3 {
            continue;
        }
        let dims = t.shape.dims();
        let (e, a, b) = (dims[0], dims[1], dims[2]);
        if hp.n_experts.is_none() {
            hp.n_experts = Some(e);
        }
        if let Some(dm) = d_model {
            if a == dm && hp.d_ff.is_none() {
                hp.d_ff = Some(b);
            } else if b == dm && hp.d_ff.is_none() {
                hp.d_ff = Some(a);
            }
        }
        if hp.n_experts.is_some() && hp.d_ff.is_some() {
            break;
        }
    }

    hp
}

// --- Classification --------------------------------------------------------

fn classify_tensor(t: &RawTensor, hp: &InferredHyperparams) -> TensorKind {
    let dims = t.shape.dims();
    let rank = t.shape.rank();

    // Paired quant tensors.
    match (t.role, t.dtype, rank) {
        (TensorRole::QuantWeight, TensorDType::I8, 3) => {
            let (e, a, b) = (dims[0], dims[1], dims[2]);
            let expected_e = hp.n_experts;
            let d_model = hp.d_model;

            let is_expert_block = expected_e.map(|n| e == n).unwrap_or(false);
            if is_expert_block {
                return match d_model {
                    Some(dm) if a == dm => TensorKind::MoeExpertProjection {
                        // (E, d_model, d_ff): the up/gate projection. Shape
                        // alone cannot tell them apart on Grok-1.
                        projection: MoeProjection::Unresolved,
                    },
                    Some(dm) if b == dm => TensorKind::MoeExpertProjection {
                        projection: MoeProjection::Down,
                    },
                    _ => TensorKind::MoeExpertProjection {
                        projection: MoeProjection::Unresolved,
                    },
                };
            }
            return TensorKind::Unknown {
                reason: format!("quant.weight rank=3 shape={:?} unmatched", dims),
            };
        }
        (TensorRole::QuantScales, TensorDType::F32, _) => {
            return TensorKind::MoeScales;
        }
        _ => {}
    }

    if t.role == TensorRole::QuantWeight && t.dtype == TensorDType::I8 && rank == 2 {
        let (a, b) = (dims[0], dims[1]);
        if let Some(dm) = hp.d_model {
            if a == dm && b == dm {
                return TensorKind::QuantizedAttentionProjection {
                    width: QuantizedAttentionWidth::ModelWidth,
                };
            }
            if a == dm && b < dm {
                return TensorKind::QuantizedAttentionProjection {
                    width: QuantizedAttentionWidth::Narrow,
                };
            }
        }
    }

    // Plain tensors.
    if t.role != TensorRole::Tensor {
        return TensorKind::Unknown {
            reason: format!("unexpected role={:?} dtype={:?}", t.role, t.dtype),
        };
    }

    // Rank-2 f32: embedding or router.
    if rank == 2 && t.dtype == TensorDType::F32 {
        let (a, b) = (dims[0], dims[1]);
        if hp.vocab_size == Some(a) && hp.d_model == Some(b) {
            return TensorKind::TokenEmbedding;
        }
        // Router: (d_model, n_experts) where n_experts is small.
        if hp.d_model == Some(a) && hp.n_experts == Some(b) {
            return TensorKind::Router;
        }
        // Larger rank-2 f32 that is not the embedding is treated as an f32
        // attention projection stored outside the quant envelope.
        return TensorKind::AttnProjF32;
    }

    // Rank-1 f32 of width d_model: per-block or final norm. Block vs final
    // is decided after block assignment (see `finalize_norms`). Start as
    // BlockNorm and let that pass promote the tail-position record to
    // FinalNorm.
    if rank == 1 && t.dtype == TensorDType::F32 {
        if hp.d_model == Some(dims[0]) {
            return TensorKind::BlockNorm;
        }
        return TensorKind::Unknown {
            reason: format!("rank-1 f32 width={} != d_model", dims[0]),
        };
    }

    // Rank-3+ f32 `tensor` (not quant.scales): treat as AttnProjF32 where
    // plausible, else Unknown.
    if rank >= 3 && t.dtype == TensorDType::F32 {
        return TensorKind::AttnProjF32;
    }

    TensorKind::Unknown {
        reason: format!(
            "unhandled rank={} dtype={:?} dims={:?}",
            rank, t.dtype, dims
        ),
    }
}

// --- Block assignment ------------------------------------------------------

/// Shard layout assumption for a well-formed Grok-1 checkpoint:
///
///   shard 0                  = token embedding         (1 shard)
///   one edge norm singleton  = final/pre-head norm     (1 shard)
///   remaining shards         = transformer blocks      (K shards / block)
///
/// For Grok-1 `ckpt-0` observed in the wild: `K = 12`, total = 770.
/// The norm singleton can appear either at the tail or immediately after
/// the embedding depending on the sorted shard order. Pick the placement
/// whose block window accounts for router-shaped tensors instead of leaving
/// them unassigned.
///
/// If the shard count does not fit this layout exactly, we leave
/// `block_index` unset and the norm singleton un-promoted; downstream consumers
/// can still use `kind`, shape, and shard_ordinal directly.
fn assign_block_indices(tensors: &mut [TensorInfo], shard_count: usize) -> Option<u32> {
    // Try to divide the interior shards into equally-sized blocks, preferring
    // a known-good K if the numbers agree.
    if shard_count < 3 {
        return None;
    }
    let interior = shard_count - 2; // drop embedding + final-norm singletons
    // Candidate block sizes we try, in priority order.
    let candidates = [12usize];
    let mut chosen: Option<(usize, usize)> = None; // (k_per_block, n_blocks)
    for &k in &candidates {
        if k > 0 && interior % k == 0 {
            chosen = Some((k, interior / k));
            break;
        }
    }

    let (k_per_block, n_blocks) = chosen?;

    let layout = choose_grok_block_layout(tensors, shard_count, k_per_block, n_blocks)?;

    for t in tensors.iter_mut() {
        let ord = t.shard_ordinal as usize;
        if ord == 0 {
            // Embedding: no block assignment.
            continue;
        }
        if ord == layout.norm_singleton_shard {
            // Final/pre-head norm singleton. Promote a BlockNorm record to FinalNorm.
            if matches!(t.kind, TensorKind::BlockNorm) {
                t.kind = TensorKind::FinalNorm;
            }
            continue;
        }
        if ord >= layout.first_block_shard && ord <= layout.last_block_shard {
            let b = (ord - layout.first_block_shard) / k_per_block;
            let slot = (ord - layout.first_block_shard) % k_per_block;
            t.block_index = Some(b as u32);
            t.block_slot = Some(slot as u32);
        }
    }

    Some(n_blocks as u32)
}

fn assign_block_indices_for_scan(
    tensors: &mut [TensorInfo],
    shard_count: usize,
    limit_truncated: bool,
) -> Option<u32> {
    if limit_truncated {
        return None;
    }
    assign_block_indices(tensors, shard_count)
}

#[derive(Clone, Copy, Debug)]
struct GrokBlockLayout {
    first_block_shard: usize,
    last_block_shard: usize,
    norm_singleton_shard: usize,
    assigned_routers: usize,
    terminal_router_evidence: bool,
}

fn choose_grok_block_layout(
    tensors: &[TensorInfo],
    shard_count: usize,
    k_per_block: usize,
    n_blocks: usize,
) -> Option<GrokBlockLayout> {
    let block_shards = k_per_block.checked_mul(n_blocks)?;
    let last_shard = shard_count.checked_sub(1)?;
    let tail_norm = grok_layout_candidate(tensors, 1, last_shard, block_shards);
    let leading_norm = grok_layout_candidate(tensors, 2, 1, block_shards);

    match (tail_norm, leading_norm) {
        (Some(tail), Some(leading)) if leading.assigned_routers > tail.assigned_routers => {
            Some(leading)
        }
        (Some(tail), _) => Some(tail),
        (None, Some(leading)) if leading.terminal_router_evidence => Some(leading),
        (None, None) => None,
        (None, Some(_)) => None,
    }
}

fn grok_layout_candidate(
    tensors: &[TensorInfo],
    first_block_shard: usize,
    norm_singleton_shard: usize,
    block_shards: usize,
) -> Option<GrokBlockLayout> {
    let last_block_shard = first_block_shard
        .checked_add(block_shards)?
        .checked_sub(1)?;
    let norm_is_plausible = tensors.iter().any(|tensor| {
        tensor.shard_ordinal as usize == norm_singleton_shard
            && matches!(tensor.kind, TensorKind::BlockNorm)
    }) && !tensors.iter().any(|tensor| {
        tensor.shard_ordinal as usize == norm_singleton_shard
            && matches!(
                tensor.kind,
                TensorKind::Router | TensorKind::MoeExpertProjection { .. }
            )
    });
    if !norm_is_plausible {
        return None;
    }

    let assigned_routers = tensors
        .iter()
        .filter(|tensor| {
            matches!(tensor.kind, TensorKind::Router)
                && (first_block_shard..=last_block_shard).contains(&(tensor.shard_ordinal as usize))
        })
        .count();
    let terminal_router_evidence = tensors
        .iter()
        .filter(|tensor| {
            matches!(tensor.kind, TensorKind::Router)
                && tensor.shard_ordinal as usize == last_block_shard
        })
        .count()
        > 0;

    Some(GrokBlockLayout {
        first_block_shard,
        last_block_shard,
        norm_singleton_shard,
        assigned_routers,
        terminal_router_evidence,
    })
}

// --- Block summaries -------------------------------------------------------

fn summarize_blocks(tensors: &[TensorInfo]) -> Vec<BlockSummary> {
    use std::collections::BTreeMap;

    // Bucket tensors: None => embedding + final norm singletons, Some(i) => block i.
    let mut by_block: BTreeMap<Option<u32>, Vec<&TensorInfo>> = BTreeMap::new();
    for t in tensors {
        by_block.entry(t.block_index).or_default().push(t);
    }

    // We want the output order: embedding singleton first, then block 0..N,
    // then final-norm singleton. To keep a single summary type, we emit
    // block-assigned entries under `Some(i)` and a synthetic singleton under
    // `None` whose label distinguishes its members.
    let mut out: Vec<BlockSummary> = Vec::new();

    // Split the `None` bucket by kind so embedding and final-norm get their
    // own summary rows.
    if let Some(singletons) = by_block.remove(&None) {
        let embed: Vec<&&TensorInfo> = singletons
            .iter()
            .filter(|t| matches!(t.kind, TensorKind::TokenEmbedding))
            .collect();
        let finals: Vec<&&TensorInfo> = singletons
            .iter()
            .filter(|t| matches!(t.kind, TensorKind::FinalNorm))
            .collect();
        let other: Vec<&&TensorInfo> = singletons
            .iter()
            .filter(|t| !matches!(t.kind, TensorKind::TokenEmbedding | TensorKind::FinalNorm))
            .collect();

        if !embed.is_empty() {
            out.push(build_summary(
                None,
                "embedding",
                embed.iter().map(|t| **t).collect(),
            ));
        }
        for (i, b) in by_block {
            out.push(build_summary(i, &format!("block_{:03}", i.unwrap_or(0)), b));
        }
        if !finals.is_empty() {
            out.push(build_summary(
                None,
                "final_norm",
                finals.iter().map(|t| **t).collect(),
            ));
        }
        if !other.is_empty() {
            out.push(build_summary(
                None,
                "unassigned",
                other.iter().map(|t| **t).collect(),
            ));
        }
    } else {
        for (i, b) in by_block {
            out.push(build_summary(i, &format!("block_{:03}", i.unwrap_or(0)), b));
        }
    }

    out
}

fn build_summary(block_index: Option<u32>, label: &str, members: Vec<&TensorInfo>) -> BlockSummary {
    use std::collections::BTreeMap;

    let shard_range = if members.is_empty() {
        None
    } else {
        let mut lo = u32::MAX;
        let mut hi = 0u32;
        for t in &members {
            lo = lo.min(t.shard_ordinal);
            hi = hi.max(t.shard_ordinal);
        }
        Some(ShardRange {
            start: lo,
            end_inclusive: hi,
        })
    };

    let tensor_count = members.len() as u32;
    let total_nbytes = members.iter().map(|t| t.nbytes).sum();

    let mut dtypes: Vec<TensorDType> = Vec::new();
    for t in &members {
        if !dtypes.contains(&t.dtype) {
            dtypes.push(t.dtype);
        }
    }

    let mut by_kind: BTreeMap<String, (u32, u64)> = BTreeMap::new();
    for t in &members {
        let k = t.kind.short_label();
        let e = by_kind.entry(k).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.nbytes;
    }
    let kinds: Vec<KindCount> = by_kind
        .into_iter()
        .map(|(k, (c, n))| KindCount {
            kind_label: k,
            count: c,
            nbytes: n,
        })
        .collect();

    BlockSummary {
        block_index,
        label: label.to_string(),
        shard_range,
        tensor_count,
        total_nbytes,
        dtypes,
        kinds,
    }
}

// --- Totals ----------------------------------------------------------------

fn compute_totals(tensors: &[TensorInfo]) -> InventoryTotals {
    let mut out = InventoryTotals {
        tensors: tensors.len() as u64,
        ..Default::default()
    };
    for t in tensors {
        out.total_nbytes += t.nbytes;
        out.total_elements += t.shape.numel();
        match t.dtype {
            TensorDType::F32 => out.f32_tensors += 1,
            TensorDType::I8 => out.i8_tensors += 1,
        }
        match t.role {
            TensorRole::QuantWeight | TensorRole::QuantScales => out.quant_tensors += 1,
            TensorRole::Tensor => {}
        }
    }
    out
}

fn grok1_discovered_counts(inv: &ModelInventory) -> Grok1CoverageCounts {
    let block_indices = inv
        .tensors
        .iter()
        .filter_map(|tensor| tensor.block_index)
        .collect::<BTreeSet<_>>();

    Grok1CoverageCounts {
        blocks: block_indices.len() as u32,
        tensors: inv.tensors.len() as u64,
        routers: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::Router))
            .count() as u64,
        expert_families: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::MoeExpertProjection { .. }))
            .count() as u64,
        unknown_tensors: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::Unknown { .. }))
            .count() as u64,
    }
}

fn grok1_unknown_slots(inv: &ModelInventory) -> Vec<Grok1UnknownSlot> {
    let mut out = inv
        .tensors
        .iter()
        .filter_map(|tensor| match &tensor.kind {
            TensorKind::Unknown { reason } => Some(Grok1UnknownSlot {
                structural_name: grok1_structural_name(tensor),
                block_index: tensor.block_index,
                block_slot: tensor.block_slot,
                shape: tensor.shape.clone(),
                reason: reason.clone(),
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.structural_name.cmp(&b.structural_name));
    out
}

fn compare_count<T>(label: &str, actual: T, expected: T, errors: &mut Vec<String>)
where
    T: Copy + Eq + std::fmt::Display,
{
    if actual != expected {
        errors.push(format!("{label} {actual} != expected {expected}"));
    }
}

fn validate_grok1_blocks(inv: &ModelInventory, errors: &mut Vec<String>) {
    let mut by_block: BTreeMap<u32, Vec<&TensorInfo>> = BTreeMap::new();
    for tensor in &inv.tensors {
        if let Some(block_index) = tensor.block_index {
            by_block.entry(block_index).or_default().push(tensor);
        }
    }

    for block_index in 0..GROK1_EXPECTED_BLOCKS {
        match by_block.get(&block_index) {
            Some(tensors) if tensors.len() == 12 => {}
            Some(tensors) => errors.push(format!(
                "block_{block_index:03} has {} tensors, expected 12",
                tensors.len()
            )),
            None => errors.push(format!("missing block_{block_index:03}")),
        }
    }

    for block_index in by_block.keys() {
        if *block_index >= GROK1_EXPECTED_BLOCKS {
            errors.push(format!("unexpected block_{block_index:03}"));
        }
    }

    for tensor in &inv.tensors {
        if tensor.block_index.is_none()
            && !matches!(
                tensor.kind,
                TensorKind::TokenEmbedding | TensorKind::FinalNorm
            )
        {
            errors.push(format!(
                "unexpected unassigned tensor {}",
                grok1_structural_name(tensor)
            ));
        }
    }
}

fn validate_unique_structural_names(inv: &ModelInventory, errors: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for tensor in &inv.tensors {
        let name = grok1_structural_name(tensor);
        if !seen.insert(name.clone()) {
            errors.push(format!("duplicate structural name {name}"));
        }
    }
}

fn validate_unique_source_keys(inv: &ModelInventory, errors: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for tensor in &inv.tensors {
        let key = (tensor.shard_ordinal, tensor.in_shard_index);
        if !seen.insert(key) {
            errors.push(format!(
                "duplicate tensor key shard={} in_shard={}",
                tensor.shard_ordinal, tensor.in_shard_index
            ));
        }
    }
}

fn validate_grok1_expected_slots(inv: &ModelInventory, errors: &mut Vec<String>) {
    for block_index in 0..GROK1_EXPECTED_BLOCKS {
        for slot in 0..12u32 {
            let tensors = inv
                .tensors
                .iter()
                .filter(|tensor| {
                    tensor.block_index == Some(block_index) && tensor.block_slot == Some(slot)
                })
                .collect::<Vec<_>>();
            if tensors.is_empty() {
                errors.push(format!("missing block_{block_index:03}.slot_{slot:02}"));
            }
            if tensors.len() > 1 {
                errors.push(format!(
                    "duplicate block_{block_index:03}.slot_{slot:02} entries: {}",
                    tensors.len()
                ));
            }
            if let Some(tensor) = tensors.first() {
                validate_grok1_slot_tensor(tensor, errors);
            }
        }
    }
}

fn validate_grok1_slot_tensor(tensor: &TensorInfo, errors: &mut Vec<String>) {
    let Some(slot) = tensor.block_slot else {
        return;
    };
    let name = grok1_structural_name(tensor);
    match slot {
        0 | 2 => validate_tensor_signature(
            tensor,
            &name,
            TensorRole::QuantWeight,
            TensorDType::I8,
            &[GROK1_N_EXPERTS, GROK1_D_MODEL, GROK1_D_FF],
            |kind| matches!(kind, TensorKind::MoeExpertProjection { .. }),
            errors,
        ),
        1 => validate_tensor_signature(
            tensor,
            &name,
            TensorRole::QuantWeight,
            TensorDType::I8,
            &[GROK1_N_EXPERTS, GROK1_D_FF, GROK1_D_MODEL],
            |kind| matches!(kind, TensorKind::MoeExpertProjection { .. }),
            errors,
        ),
        3 | 6 => validate_tensor_signature(
            tensor,
            &name,
            TensorRole::QuantWeight,
            TensorDType::I8,
            &[GROK1_D_MODEL, 1_024],
            |kind| {
                matches!(
                    kind,
                    TensorKind::QuantizedAttentionProjection {
                        width: QuantizedAttentionWidth::Narrow
                    }
                )
            },
            errors,
        ),
        4 | 5 => validate_tensor_signature(
            tensor,
            &name,
            TensorRole::QuantWeight,
            TensorDType::I8,
            &[GROK1_D_MODEL, GROK1_D_MODEL],
            |kind| {
                matches!(
                    kind,
                    TensorKind::QuantizedAttentionProjection {
                        width: QuantizedAttentionWidth::ModelWidth
                    }
                )
            },
            errors,
        ),
        11 => validate_tensor_signature(
            tensor,
            &name,
            TensorRole::Tensor,
            TensorDType::F32,
            &[GROK1_D_MODEL, GROK1_N_EXPERTS],
            |kind| matches!(kind, TensorKind::Router),
            errors,
        ),
        7..=10 => {
            if matches!(tensor.kind, TensorKind::Unknown { .. }) {
                errors.push(format!("unexpected unknown tensor at {name}"));
            }
        }
        _ => errors.push(format!("unexpected block slot {slot} at {name}")),
    }
}

fn validate_tensor_signature(
    tensor: &TensorInfo,
    name: &str,
    role: TensorRole,
    dtype: TensorDType,
    shape: &[u64],
    kind_matches: impl FnOnce(&TensorKind) -> bool,
    errors: &mut Vec<String>,
) {
    if tensor.role != role
        || tensor.dtype != dtype
        || tensor.shape.dims() != shape
        || !kind_matches(&tensor.kind)
    {
        errors.push(format!(
            "unexpected tensor layout at {name}: role={} dtype={} shape={} kind={}",
            tensor.role.label(),
            tensor.dtype.label(),
            tensor.shape.render(),
            tensor.kind.short_label()
        ));
    }
}

fn grok1_structural_name(tensor: &TensorInfo) -> String {
    match (tensor.block_index, tensor.block_slot) {
        (Some(block), Some(slot)) => {
            format!(
                "block_{block:03}.slot_{slot:02}.{}",
                tensor.kind.short_label()
            )
        }
        (Some(block), None) => format!(
            "block_{block:03}.slot_unknown.{}",
            tensor.kind.short_label()
        ),
        (None, _) if matches!(tensor.kind, TensorKind::TokenEmbedding) => "embedding".to_string(),
        (None, _) if matches!(tensor.kind, TensorKind::FinalNorm) => "final_norm".to_string(),
        (None, Some(slot)) => format!("unassigned.slot_{slot:02}.{}", tensor.kind.short_label()),
        (None, None) => format!("unassigned.{}", tensor.kind.short_label()),
    }
}

fn grok1_checksum(
    inv: &ModelInventory,
    discovered: &Grok1CoverageCounts,
    unknown_slots: &[Grok1UnknownSlot],
) -> String {
    let mut lines = vec![
        format!("model_family={}", inv.model_family),
        format!("schema_version={}", inv.schema_version),
        format!("blocks={}", discovered.blocks),
        format!("tensors={}", discovered.tensors),
        format!("routers={}", discovered.routers),
        format!("expert_families={}", discovered.expert_families),
        format!("unknown_tensors={}", discovered.unknown_tensors),
    ];

    let mut tensor_lines = inv
        .tensors
        .iter()
        .map(|tensor| {
            format!(
                "tensor|{}|{}|{}|{}|{}|{}|{}",
                grok1_structural_name(tensor),
                tensor.shard_ordinal,
                tensor.in_shard_index,
                tensor.role.label(),
                tensor.dtype.label(),
                tensor.shape.render(),
                tensor.kind.short_label()
            )
        })
        .collect::<Vec<_>>();
    tensor_lines.sort();
    lines.extend(tensor_lines);
    for unknown in unknown_slots {
        lines.push(format!(
            "unknown|{}|{}",
            unknown.structural_name, unknown.reason
        ));
    }

    stable_fnv1a64(lines.join("\n").as_bytes())
}

fn stable_fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::parser::RawTensor;
    use crate::routing::build_routing_report;
    use crate::schema::{TensorRole, TensorShape};

    use super::*;

    #[test]
    fn shifted_grok_layout_assigns_all_sixty_four_routers_to_blocks() {
        let mut tensors = shifted_grok_ckpt0_tensors();

        let n_blocks = assign_block_indices(&mut tensors, 770);

        assert_eq!(n_blocks, Some(64));
        assert!(tensors.iter().any(|tensor| tensor.shard_ordinal == 1
            && matches!(tensor.kind, TensorKind::FinalNorm)
            && tensor.block_index.is_none()));

        let routers = tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::Router))
            .collect::<Vec<_>>();
        assert_eq!(routers.len(), 64);
        assert!(routers.iter().all(|tensor| tensor.block_index.is_some()));
        assert!(routers.iter().any(|tensor| tensor.shard_ordinal == 13
            && tensor.block_index == Some(0)
            && tensor.block_slot == Some(11)));
        assert!(routers.iter().any(|tensor| tensor.shard_ordinal == 769
            && tensor.block_index == Some(63)
            && tensor.block_slot == Some(11)));
    }

    #[test]
    fn shifted_grok_layout_routing_report_has_no_unassigned_candidate() {
        let mut tensors = shifted_grok_ckpt0_tensors();
        assign_block_indices(&mut tensors, 770);
        let inv = inventory(tensors, 770);

        let report = build_routing_report(&inv);

        assert_eq!(report.candidate_tensors.len(), 64);
        assert_eq!(report.relevant_block_count, 64);
        assert!(
            report
                .candidate_tensors
                .iter()
                .all(|tensor| tensor.block_index.is_some())
        );
        assert!(
            report
                .candidate_tensors
                .iter()
                .all(|tensor| !tensor.structural_name.starts_with("unassigned."))
        );
        assert!(
            report
                .candidate_tensors
                .iter()
                .any(|tensor| tensor.structural_name == "block_000.routing_slot_11")
        );
        assert!(
            report
                .candidate_tensors
                .iter()
                .any(|tensor| tensor.structural_name == "block_063.routing_slot_11")
        );
    }

    #[test]
    fn missing_tail_norm_does_not_force_shifted_layout() {
        let mut tensors = normal_grok_ckpt0_tensors_without_tail_norm();

        let n_blocks = assign_block_indices(&mut tensors, 770);

        assert_eq!(n_blocks, None);
        assert!(
            tensors
                .iter()
                .filter(|tensor| matches!(tensor.kind, TensorKind::Router))
                .all(|tensor| tensor.block_index.is_none() && tensor.block_slot.is_none())
        );
    }

    #[test]
    fn limited_scan_skips_block_assignment() {
        let mut tensors = shifted_grok_ckpt0_tensors();

        let n_blocks = assign_block_indices_for_scan(&mut tensors, 770, true);

        assert_eq!(n_blocks, None);
        assert!(tensors.iter().all(|tensor| tensor.block_index.is_none()));
        assert!(
            tensors
                .iter()
                .all(|tensor| !matches!(tensor.kind, TensorKind::FinalNorm))
        );
    }

    #[test]
    fn non_truncating_limited_scan_keeps_block_assignment() {
        let mut tensors = shifted_grok_ckpt0_tensors();

        let n_blocks = assign_block_indices_for_scan(&mut tensors, 770, false);

        assert_eq!(n_blocks, Some(64));
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| matches!(tensor.kind, TensorKind::Router))
                .filter(|tensor| tensor.block_index.is_some())
                .count(),
            64
        );
        assert!(
            tensors
                .iter()
                .any(|tensor| matches!(tensor.kind, TensorKind::FinalNorm))
        );
    }

    #[test]
    fn inventory_fails_on_shard_parse_error() {
        let dir = temp_dir("bad_shard");
        fs::write(dir.join("tensor00000_000"), b"not a pickle").unwrap();

        let err = build_inventory(&dir, &InventoryConfig::default()).unwrap_err();

        assert!(format!("{err:#}").contains("parse shard"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn classifies_rank2_quantized_attention_widths() {
        let hp = InferredHyperparams {
            d_model: Some(6_144),
            ..InferredHyperparams::default()
        };

        let model_width = classify_tensor(
            &raw(TensorRole::QuantWeight, TensorDType::I8, vec![6_144, 6_144]),
            &hp,
        );
        let narrow = classify_tensor(
            &raw(TensorRole::QuantWeight, TensorDType::I8, vec![6_144, 1_024]),
            &hp,
        );

        assert_eq!(
            model_width,
            TensorKind::QuantizedAttentionProjection {
                width: QuantizedAttentionWidth::ModelWidth
            }
        );
        assert_eq!(model_width.short_label(), "attn_proj_i8.model_width");
        assert_eq!(
            narrow,
            TensorKind::QuantizedAttentionProjection {
                width: QuantizedAttentionWidth::Narrow
            }
        );
        assert_eq!(narrow.short_label(), "attn_proj_i8.narrow");
    }

    #[test]
    fn validates_complete_grok1_coverage_manifest() {
        let inv = complete_grok1_inventory();

        let manifest = validate_grok1_complete_manifest(&inv).expect("valid manifest");

        assert_eq!(manifest.expected.tensors, 770);
        assert_eq!(manifest.discovered.tensors, 770);
        assert_eq!(manifest.expected.blocks, 64);
        assert_eq!(manifest.discovered.blocks, 64);
        assert_eq!(manifest.expected.routers, 64);
        assert_eq!(manifest.discovered.routers, 64);
        assert_eq!(manifest.expected.expert_families, 192);
        assert_eq!(manifest.discovered.expert_families, 192);
        assert!(manifest.unknown_slots.is_empty());
        assert_eq!(manifest.checksum, "fnv1a64:ce4d8e7cde002f74");
    }

    #[test]
    fn grok1_coverage_fails_on_missing_block() {
        let mut inv = complete_grok1_inventory();
        inv.tensors.retain(|tensor| tensor.block_index != Some(10));
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("missing block_010"));
    }

    #[test]
    fn grok1_coverage_fails_on_missing_tensor() {
        let mut inv = complete_grok1_inventory();
        inv.tensors
            .retain(|tensor| !(tensor.block_index == Some(7) && tensor.block_slot == Some(4)));
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("missing block_007.slot_04"));
    }

    #[test]
    fn grok1_coverage_fails_on_duplicate_tensor_key() {
        let mut inv = complete_grok1_inventory();
        let duplicate = inv
            .tensors
            .iter()
            .find(|tensor| tensor.block_index == Some(3) && tensor.block_slot == Some(5))
            .expect("source tensor")
            .clone();
        inv.tensors.push(duplicate);
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("duplicate structural name"));
        assert!(format!("{err:#}").contains("block_003.slot_05"));
    }

    #[test]
    fn grok1_coverage_fails_on_unexpected_key_layout() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(4) && tensor.block_slot == Some(3))
            .expect("slot tensor");
        tensor.kind = TensorKind::Unknown {
            reason: "synthetic unexpected slot layout".to_string(),
        };
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unknown_tensors 1 != expected 0"));
        assert!(format!("{err:#}").contains("unexpected tensor layout at block_004.slot_03"));
    }

    #[test]
    fn grok1_coverage_checksum_is_reproducible() {
        let inv = complete_grok1_inventory();

        let first = validate_grok1_complete_manifest(&inv)
            .expect("first manifest")
            .checksum;
        let second = validate_grok1_complete_manifest(&inv)
            .expect("second manifest")
            .checksum;

        assert_eq!(first, second);
    }

    fn shifted_grok_ckpt0_tensors() -> Vec<TensorInfo> {
        let mut tensors = Vec::new();
        tensors.push(tensor(
            0,
            TensorKind::TokenEmbedding,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![131_072, 6_144],
        ));
        tensors.push(tensor(
            1,
            TensorKind::BlockNorm,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![6_144],
        ));

        append_grok_blocks(&mut tensors, 2);

        tensors
    }

    fn complete_grok1_inventory() -> ModelInventory {
        let mut tensors = vec![
            complete_tensor(
                0,
                0,
                None,
                None,
                TensorKind::TokenEmbedding,
                TensorRole::Tensor,
                TensorDType::F32,
                vec![131_072, 6_144],
            ),
            complete_tensor(
                1,
                0,
                None,
                None,
                TensorKind::FinalNorm,
                TensorRole::Tensor,
                TensorDType::F32,
                vec![6_144],
            ),
        ];

        for block in 0..64u32 {
            for slot in 0..12u32 {
                let shard = 2 + block * 12 + slot;
                let (kind, role, dtype, shape) = match slot {
                    0 | 2 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Unresolved,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6_144, 32_768],
                    ),
                    1 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Down,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32_768, 6_144],
                    ),
                    3 | 6 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::Narrow,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 1_024],
                    ),
                    4 | 5 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::ModelWidth,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 6_144],
                    ),
                    7 | 8 => (
                        TensorKind::MoeScales,
                        TensorRole::QuantScales,
                        TensorDType::F32,
                        vec![8, 32_768],
                    ),
                    9 | 10 => (
                        TensorKind::BlockNorm,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144],
                    ),
                    11 => (
                        TensorKind::Router,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144, 8],
                    ),
                    _ => unreachable!(),
                };
                tensors.push(complete_tensor(
                    shard,
                    0,
                    Some(block),
                    Some(slot),
                    kind,
                    role,
                    dtype,
                    shape,
                ));
            }
        }

        let mut inv = ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                vocab_size: Some(131_072),
                d_model: Some(6_144),
                n_experts: Some(8),
                d_ff: Some(32_768),
                n_blocks: Some(64),
            },
            tensors,
            blocks: Vec::new(),
            totals: InventoryTotals::default(),
            schema_version: SCHEMA_VERSION,
        };
        refresh_inventory_derived_fields(&mut inv);
        inv
    }

    fn refresh_inventory_derived_fields(inv: &mut ModelInventory) {
        inv.blocks = summarize_blocks(&inv.tensors);
        inv.totals = compute_totals(&inv.tensors);
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_tensor(
        shard_ordinal: u32,
        in_shard_index: u32,
        block_index: Option<u32>,
        block_slot: Option<u32>,
        kind: TensorKind,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!("/tmp/grok-1/ckpt-0/tensor{shard_ordinal:05}_000")),
            shard_ordinal,
            in_shard_index,
            role,
            dtype,
            shape: TensorShape::new(shape.clone()),
            offset: 0,
            nbytes: dtype.itemsize() as u64 * shape.iter().product::<u64>(),
            kind,
            block_index,
            block_slot,
        }
    }

    fn normal_grok_ckpt0_tensors_without_tail_norm() -> Vec<TensorInfo> {
        let mut tensors = Vec::new();
        tensors.push(tensor(
            0,
            TensorKind::TokenEmbedding,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![131_072, 6_144],
        ));
        append_grok_blocks(&mut tensors, 1);
        tensors
    }

    fn append_grok_blocks(tensors: &mut Vec<TensorInfo>, first_block_shard: u32) {
        for block in 0..64u32 {
            let start = first_block_shard + block * 12;
            for slot in 0..12u32 {
                let shard = start + slot;
                match slot {
                    0..=3 => tensors.push(tensor(
                        shard,
                        TensorKind::BlockNorm,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144],
                    )),
                    4..=5 => tensors.push(tensor(
                        shard,
                        TensorKind::AttnProjF32,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144, 6_144],
                    )),
                    6..=7 => tensors.push(tensor(
                        shard,
                        TensorKind::MoeScales,
                        TensorRole::QuantScales,
                        TensorDType::F32,
                        vec![8, 32_768],
                    )),
                    8 | 9 => tensors.push(tensor(
                        shard,
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Unresolved,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6_144, 32_768],
                    )),
                    10 => tensors.push(tensor(
                        shard,
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Down,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32_768, 6_144],
                    )),
                    11 => tensors.push(tensor(
                        shard,
                        TensorKind::Router,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144, 8],
                    )),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn inventory(tensors: Vec<TensorInfo>, shard_count: u32) -> ModelInventory {
        let blocks = summarize_blocks(&tensors);
        let totals = compute_totals(&tensors);
        ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1/ckpt-0"),
            shard_count,
            inferred: InferredHyperparams {
                vocab_size: Some(131_072),
                d_model: Some(6_144),
                n_experts: Some(8),
                d_ff: Some(32_768),
                n_blocks: Some(64),
            },
            tensors,
            blocks,
            totals,
            schema_version: SCHEMA_VERSION,
        }
    }

    fn tensor(
        shard_ordinal: u32,
        kind: TensorKind,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!("/tmp/grok-1/ckpt-0/tensor{shard_ordinal:05}_000")),
            shard_ordinal,
            in_shard_index: 0,
            role,
            dtype,
            shape: TensorShape::new(shape),
            offset: 0,
            nbytes: 0,
            kind,
            block_index: None,
            block_slot: None,
        }
    }

    fn raw(role: TensorRole, dtype: TensorDType, shape: Vec<u64>) -> RawTensor {
        RawTensor {
            role,
            dtype,
            shape: TensorShape::new(shape.clone()),
            offset: 0,
            nbytes: dtype.itemsize() as u64 * shape.iter().product::<u64>(),
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xai_dissect_inventory_{label}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
