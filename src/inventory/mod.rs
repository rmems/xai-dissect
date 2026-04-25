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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::parser::{self, RawTensor};
use crate::schema::{
    BlockSummary, InferredHyperparams, InventoryTotals, KindCount, ModelInventory, MoeProjection,
    ShardRange, TensorDType, TensorInfo, TensorKind, TensorRole,
};

pub const SCHEMA_VERSION: u32 = 1;

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

    let shards = collect_shards(path, &cfg.prefix, cfg.limit)?;
    if shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            path.display(),
            cfg.prefix
        );
    }

    // Pass 1: parse every shard into RawTensor records.
    let mut raws_per_shard: Vec<Vec<RawTensor>> = Vec::with_capacity(shards.len());
    for shard in &shards {
        match parser::dissect_shard(shard) {
            Ok(ts) => raws_per_shard.push(ts),
            Err(e) => {
                eprintln!("warn: {}: {:#}", shard.display(), e);
                raws_per_shard.push(Vec::new());
            }
        }
    }

    // Pass 2: infer model hyperparameters from the raw set.
    let hp = infer_hyperparams(&raws_per_shard);

    // Pass 3: classify each raw tensor into a TensorKind.
    let mut tensors: Vec<TensorInfo> = Vec::new();
    for (shard_ordinal, (shard_path, raws)) in shards.iter().zip(raws_per_shard.iter()).enumerate()
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
    let n_blocks = assign_block_indices(&mut tensors, shards.len());

    // Pass 5: build block summaries and totals.
    let blocks = summarize_blocks(&tensors);
    let totals = compute_totals(&tensors);

    Ok(ModelInventory {
        model_family: cfg.model_family.clone(),
        checkpoint_path: path.to_path_buf(),
        shard_count: shards.len() as u32,
        inferred: InferredHyperparams { n_blocks, ..hp },
        tensors,
        blocks,
        totals,
        schema_version: SCHEMA_VERSION,
    })
}

// --- Shard enumeration -----------------------------------------------------

fn collect_shards(path: &Path, prefix: &str, limit: Option<usize>) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(prefix))
                    .unwrap_or(false)
        })
        .collect();
    shards.sort();
    if let Some(n) = limit {
        shards.truncate(n);
    }
    Ok(shards)
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
/// `block_index` unset and the tail norm un-promoted; downstream consumers
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

#[derive(Clone, Copy, Debug)]
struct GrokBlockLayout {
    first_block_shard: usize,
    last_block_shard: usize,
    norm_singleton_shard: usize,
    assigned_routers: usize,
    singleton_routers: usize,
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
        (Some(tail), Some(leading))
            if leading.assigned_routers > tail.assigned_routers
                && leading.singleton_routers < tail.singleton_routers =>
        {
            Some(leading)
        }
        (Some(tail), _) => Some(tail),
        (None, Some(leading)) => Some(leading),
        (None, None) => None,
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
    let singleton_routers = tensors
        .iter()
        .filter(|tensor| {
            matches!(tensor.kind, TensorKind::Router)
                && tensor.shard_ordinal as usize == norm_singleton_shard
        })
        .count();

    Some(GrokBlockLayout {
        first_block_shard,
        last_block_shard,
        norm_singleton_shard,
        assigned_routers,
        singleton_routers,
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

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

        for block in 0..64u32 {
            let start = 2 + block * 12;
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

        tensors
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
}
