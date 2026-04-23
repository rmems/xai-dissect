// SPDX-License-Identifier: GPL-3.0-only
//
// Offline tensor-payload profiling for SAAQ-readiness scouting. This layer
// may read tensor payload bytes, but it never mutates weights and never
// executes model code.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::routing::build_routing_report;
use crate::schema::{
    CandidateTensorManifest, InferredHyperparams, LayerStats, ModelInventory, NormSummary,
    OutlierSummary, RankedTensorStat, RoutingReport, SaaqCandidate, SaaqDisposition,
    SaaqLayerReadiness, SaaqReadinessReport, SaaqRegionClass, StatsProfileReport,
    StatsSamplingConfig, TensorDType, TensorInfo, TensorKind, TensorStats, VarianceSummary,
};

pub const STATS_PROFILE_SCHEMA_VERSION: u32 = 1;
pub const SAAQ_READINESS_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct StatsConfig {
    pub max_sample_values: usize,
    pub f32_near_zero_abs: f64,
    pub i8_near_zero_abs: i64,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            max_sample_values: 65_536,
            f32_near_zero_abs: 1e-3,
            i8_near_zero_abs: 1,
        }
    }
}

/// Build a tensor-statistics profile from an inventory.
pub fn build_stats_report(inv: &ModelInventory, cfg: &StatsConfig) -> Result<StatsProfileReport> {
    let mut shard_cache = ShardCache::default();
    let mut tensors = Vec::with_capacity(inv.tensors.len());

    for tensor in &inv.tensors {
        let bytes = shard_cache.tensor_bytes(tensor)?;
        tensors.push(profile_tensor(tensor, bytes, cfg)?);
    }

    let layers = summarize_layers(&tensors);
    let norm_summary = summarize_norms(&tensors);
    let variance_summary = summarize_variance(&tensors);
    let outlier_summary = summarize_outliers(&tensors);

    Ok(StatsProfileReport {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        shard_count: inv.shard_count,
        inferred: clone_hyperparams(&inv.inferred),
        sampling: StatsSamplingConfig {
            max_sample_values: cfg.max_sample_values as u64,
            f32_near_zero_abs: cfg.f32_near_zero_abs,
            i8_near_zero_abs: cfg.i8_near_zero_abs,
        },
        tensors,
        layers,
        norm_summary,
        variance_summary,
        outlier_summary,
        schema_version: STATS_PROFILE_SCHEMA_VERSION,
    })
}

/// Build a SAAQ-readiness report from an inventory. This reuses the routing
/// analysis and the tensor statistics profile to rank candidate tensors.
pub fn build_saaq_readiness_report(
    inv: &ModelInventory,
    stats: &StatsProfileReport,
) -> SaaqReadinessReport {
    let routing = build_routing_report(inv);
    let routing_locs = routing_tensor_set(&routing);
    let routing_blocks = routing_block_set(&routing);

    let mut scored = stats
        .tensors
        .iter()
        .map(|tensor| score_tensor(tensor, inv, &routing, &routing_locs, &routing_blocks))
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| {
        b.readiness_score
            .partial_cmp(&a.readiness_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (index, candidate) in scored.iter_mut().enumerate() {
        candidate.rank = (index + 1) as u32;
    }

    let mut candidate_targets = scored
        .iter()
        .filter(|candidate| candidate.disposition == SaaqDisposition::Candidate)
        .cloned()
        .collect::<Vec<_>>();
    if candidate_targets.is_empty() {
        if let Some(fallback) = scored.iter().find(|candidate| {
            !matches!(
                candidate.region_class,
                SaaqRegionClass::RoutingCritical
                    | SaaqRegionClass::NormalizationSensitive
                    | SaaqRegionClass::AlreadyCompressed
            )
        }) {
            let mut fallback = fallback.clone();
            fallback.disposition = SaaqDisposition::Candidate;
            fallback
                .reasons
                .push("promoted as the best available non-routing target in a constrained sample".to_string());
            candidate_targets.push(fallback);
        }
    }
    let routing_critical_tensors = scored
        .iter()
        .filter(|candidate| candidate.region_class == SaaqRegionClass::RoutingCritical)
        .cloned()
        .collect::<Vec<_>>();
    let risky_tensors = scored
        .iter()
        .filter(|candidate| candidate.risk_score >= 0.7)
        .take(25)
        .cloned()
        .collect::<Vec<_>>();
    let layer_readiness = summarize_layer_readiness(&scored, &routing_blocks);
    let notes = build_saaq_notes(&candidate_targets, &routing_critical_tensors, &routing);
    let manifest = CandidateTensorManifest {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        candidates: candidate_targets.clone(),
        schema_version: SAAQ_READINESS_SCHEMA_VERSION,
    };

    SaaqReadinessReport {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        shard_count: inv.shard_count,
        inferred: clone_hyperparams(&inv.inferred),
        candidate_targets,
        routing_critical_tensors,
        risky_tensors,
        layer_readiness,
        notes,
        manifest,
        schema_version: SAAQ_READINESS_SCHEMA_VERSION,
    }
}

#[derive(Default)]
struct ShardCache {
    by_path: HashMap<PathBuf, Mmap>,
}

impl ShardCache {
    fn tensor_bytes<'a>(&'a mut self, tensor: &TensorInfo) -> Result<&'a [u8]> {
        if !self.by_path.contains_key(&tensor.shard_path) {
            let file = File::open(&tensor.shard_path)
                .with_context(|| format!("open {}", tensor.shard_path.display()))?;
            let mm = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap {}", tensor.shard_path.display()))?;
            self.by_path.insert(tensor.shard_path.clone(), mm);
        }
        let mm = self
            .by_path
            .get(&tensor.shard_path)
            .expect("inserted mmap missing");
        let start = tensor.offset as usize;
        let end = start.saturating_add(tensor.nbytes as usize);
        if end > mm.len() {
            bail!(
                "tensor payload overruns shard: {} [{}..{}) > {}",
                tensor.shard_path.display(),
                start,
                end,
                mm.len()
            );
        }
        Ok(&mm[start..end])
    }
}

fn profile_tensor(tensor: &TensorInfo, bytes: &[u8], cfg: &StatsConfig) -> Result<TensorStats> {
    let values = sample_tensor_values(tensor, bytes, cfg.max_sample_values)?;
    if values.is_empty() {
        bail!(
            "tensor {} has no values after sampling",
            tensor_structural_name(tensor)
        );
    }

    let sample_values = values.len() as u64;
    let total_values = tensor.shape.numel();
    let sampled = sample_values < total_values;

    let mean = values.iter().copied().sum::<f64>() / sample_values as f64;
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / sample_values as f64;
    let stddev = variance.sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_abs = values.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let l1_norm = values.iter().copied().map(f64::abs).sum::<f64>();
    let l2_norm = values.iter().copied().map(|v| v * v).sum::<f64>().sqrt();
    let rms = (values.iter().copied().map(|v| v * v).sum::<f64>() / sample_values as f64).sqrt();

    let near_zero_abs = match tensor.dtype {
        TensorDType::F32 => cfg.f32_near_zero_abs,
        TensorDType::I8 => cfg.i8_near_zero_abs as f64,
    };
    let zero_count = values.iter().filter(|v| **v == 0.0).count() as u64;
    let near_zero_count = values.iter().filter(|v| v.abs() <= near_zero_abs).count() as u64;
    let positive_count = values.iter().filter(|v| **v > 0.0).count() as u64;
    let negative_count = values.iter().filter(|v| **v < 0.0).count() as u64;
    let outlier_count = if stddev > 0.0 {
        values
            .iter()
            .filter(|v| ((*v - mean).abs() / stddev) > 6.0)
            .count() as u64
    } else {
        0
    };
    let peak_to_rms = if rms > 0.0 { max_abs / rms } else { 0.0 };
    let distribution_label = distribution_label(
        zero_count as f64 / sample_values as f64,
        near_zero_count as f64 / sample_values as f64,
        outlier_count as f64 / sample_values as f64,
        peak_to_rms,
    );

    Ok(TensorStats {
        shard_ordinal: tensor.shard_ordinal,
        in_shard_index: tensor.in_shard_index,
        block_index: tensor.block_index,
        block_slot: tensor.block_slot,
        structural_name: tensor_structural_name(tensor),
        role: tensor.role,
        dtype: tensor.dtype,
        shape: tensor.shape.clone(),
        kind_label: tensor.kind.short_label(),
        sampled,
        total_values,
        sample_values,
        total_nbytes: tensor.nbytes,
        mean,
        variance,
        stddev,
        min,
        max,
        max_abs,
        l1_norm,
        l2_norm,
        rms,
        zero_fraction: zero_count as f64 / sample_values as f64,
        near_zero_fraction: near_zero_count as f64 / sample_values as f64,
        positive_fraction: positive_count as f64 / sample_values as f64,
        negative_fraction: negative_count as f64 / sample_values as f64,
        outlier_fraction: outlier_count as f64 / sample_values as f64,
        peak_to_rms,
        distribution_label,
    })
}

fn sample_tensor_values(
    tensor: &TensorInfo,
    bytes: &[u8],
    max_sample_values: usize,
) -> Result<Vec<f64>> {
    let total_values = tensor.shape.numel() as usize;
    if total_values == 0 {
        return Ok(Vec::new());
    }
    let itemsize = tensor.dtype.itemsize();
    if bytes.len() != total_values.saturating_mul(itemsize) {
        bail!(
            "tensor byte length mismatch for {}: got {} expected {}",
            tensor_structural_name(tensor),
            bytes.len(),
            total_values.saturating_mul(itemsize)
        );
    }

    let sample_len = total_values.min(max_sample_values.max(1));
    let mut out = Vec::with_capacity(sample_len);
    if sample_len == total_values {
        for index in 0..total_values {
            out.push(read_value(tensor.dtype, bytes, index));
        }
    } else {
        for sample_index in 0..sample_len {
            let index = sample_index.saturating_mul(total_values) / sample_len;
            out.push(read_value(tensor.dtype, bytes, index));
        }
    }
    Ok(out)
}

fn read_value(dtype: TensorDType, bytes: &[u8], index: usize) -> f64 {
    match dtype {
        TensorDType::F32 => {
            let start = index * 4;
            let mut raw = [0u8; 4];
            raw.copy_from_slice(&bytes[start..start + 4]);
            f32::from_le_bytes(raw) as f64
        }
        TensorDType::I8 => (bytes[index] as i8) as f64,
    }
}

fn distribution_label(
    zero_fraction: f64,
    near_zero_fraction: f64,
    outlier_fraction: f64,
    peak_to_rms: f64,
) -> String {
    if zero_fraction >= 0.2 {
        "zero_heavy".to_string()
    } else if near_zero_fraction >= 0.5 {
        "near_zero_heavy".to_string()
    } else if outlier_fraction >= 0.01 || peak_to_rms >= 20.0 {
        "outlier_heavy".to_string()
    } else {
        "dense_balanced".to_string()
    }
}

fn summarize_layers(tensors: &[TensorStats]) -> Vec<LayerStats> {
    let mut by_layer: BTreeMap<(Option<u32>, String), Vec<&TensorStats>> = BTreeMap::new();
    for tensor in tensors {
        let label = layer_label_for_tensor(tensor);
        by_layer
            .entry((tensor.block_index, label))
            .or_default()
            .push(tensor);
    }

    by_layer
        .into_iter()
        .map(|((block_index, label), members)| {
            let tensor_count = members.len() as u32;
            let sampled_tensor_count =
                members.iter().filter(|tensor| tensor.sampled).count() as u32;
            let total_nbytes = members
                .iter()
                .map(|tensor| tensor.total_nbytes)
                .sum::<u64>();
            let mean_rms =
                members.iter().map(|tensor| tensor.rms).sum::<f64>() / tensor_count as f64;
            let mean_variance =
                members.iter().map(|tensor| tensor.variance).sum::<f64>() / tensor_count as f64;
            let mean_outlier_fraction = members
                .iter()
                .map(|tensor| tensor.outlier_fraction)
                .sum::<f64>()
                / tensor_count as f64;
            let routing_tensor_count = members
                .iter()
                .filter(|tensor| tensor.kind_label == "router")
                .count() as u32;
            let compressible_candidate_count = members
                .iter()
                .filter(|tensor| is_potential_compression_target(tensor))
                .count() as u32;

            LayerStats {
                block_index,
                label,
                tensor_count,
                sampled_tensor_count,
                total_nbytes,
                mean_rms,
                mean_variance,
                mean_outlier_fraction,
                routing_tensor_count,
                compressible_candidate_count,
            }
        })
        .collect()
}

fn summarize_norms(tensors: &[TensorStats]) -> NormSummary {
    let mean_rms = mean_of(tensors.iter().map(|tensor| tensor.rms));
    NormSummary {
        mean_rms,
        max_rms: rank_max_by(tensors, |tensor| tensor.rms),
        max_l2: rank_max_by(tensors, |tensor| tensor.l2_norm),
        top_rms: top_by_desc(tensors, |tensor| tensor.rms, 10),
        top_l2: top_by_desc(tensors, |tensor| tensor.l2_norm, 10),
    }
}

fn summarize_variance(tensors: &[TensorStats]) -> VarianceSummary {
    VarianceSummary {
        mean_variance: mean_of(tensors.iter().map(|tensor| tensor.variance)),
        max_variance: rank_max_by(tensors, |tensor| tensor.variance),
        min_variance: rank_min_by(tensors, |tensor| tensor.variance),
        top_variance: top_by_desc(tensors, |tensor| tensor.variance, 10),
        lowest_variance: top_by_asc(tensors, |tensor| tensor.variance, 10),
    }
}

fn summarize_outliers(tensors: &[TensorStats]) -> OutlierSummary {
    OutlierSummary {
        mean_outlier_fraction: mean_of(tensors.iter().map(|tensor| tensor.outlier_fraction)),
        most_outlier_heavy: top_by_desc(tensors, |tensor| tensor.outlier_fraction, 10),
        highest_peak_to_rms: top_by_desc(tensors, |tensor| tensor.peak_to_rms, 10),
    }
}

fn rank_max_by<F>(tensors: &[TensorStats], f: F) -> Option<RankedTensorStat>
where
    F: Fn(&TensorStats) -> f64,
{
    tensors
        .iter()
        .max_by(|a, b| f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Equal))
        .map(|tensor| ranked_tensor_stat(tensor, f(tensor)))
}

fn rank_min_by<F>(tensors: &[TensorStats], f: F) -> Option<RankedTensorStat>
where
    F: Fn(&TensorStats) -> f64,
{
    tensors
        .iter()
        .min_by(|a, b| f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Equal))
        .map(|tensor| ranked_tensor_stat(tensor, f(tensor)))
}

fn top_by_desc<F>(tensors: &[TensorStats], f: F, limit: usize) -> Vec<RankedTensorStat>
where
    F: Fn(&TensorStats) -> f64,
{
    let mut items = tensors
        .iter()
        .map(|tensor| ranked_tensor_stat(tensor, f(tensor)))
        .collect::<Vec<_>>();
    items.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(limit);
    items
}

fn top_by_asc<F>(tensors: &[TensorStats], f: F, limit: usize) -> Vec<RankedTensorStat>
where
    F: Fn(&TensorStats) -> f64,
{
    let mut items = tensors
        .iter()
        .map(|tensor| ranked_tensor_stat(tensor, f(tensor)))
        .collect::<Vec<_>>();
    items.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(limit);
    items
}

fn ranked_tensor_stat(tensor: &TensorStats, value: f64) -> RankedTensorStat {
    RankedTensorStat {
        shard_ordinal: tensor.shard_ordinal,
        in_shard_index: tensor.in_shard_index,
        structural_name: tensor.structural_name.clone(),
        kind_label: tensor.kind_label.clone(),
        block_index: tensor.block_index,
        value,
    }
}

fn mean_of<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut count = 0u64;
    let mut total = 0.0;
    for value in iter {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn score_tensor(
    tensor: &TensorStats,
    inv: &ModelInventory,
    routing: &RoutingReport,
    routing_locs: &BTreeSet<(u32, u32)>,
    routing_blocks: &BTreeSet<Option<u32>>,
) -> SaaqCandidate {
    let region_class = classify_region(tensor, routing_locs, routing_blocks);
    let size_score = if inv.totals.total_nbytes > 0 {
        tensor.total_nbytes as f64 / inv.totals.total_nbytes as f64
    } else {
        0.0
    };
    let sparsity_score = (tensor.zero_fraction * 0.7 + tensor.near_zero_fraction * 0.3).min(1.0);
    let outlier_penalty = (tensor.outlier_fraction * 20.0)
        .min(1.0)
        .max((tensor.peak_to_rms / 50.0).min(1.0));
    let dtype_bonus = match tensor.dtype {
        TensorDType::F32 => 1.0,
        TensorDType::I8 => 0.05,
    };
    let opportunity_score = clamp01(
        size_score.sqrt() * 0.45
            + sparsity_score * 0.25
            + (1.0 - outlier_penalty) * 0.2
            + dtype_bonus * 0.1,
    );

    let structural_penalty = match region_class {
        SaaqRegionClass::RoutingCritical => 0.95,
        SaaqRegionClass::NormalizationSensitive => 0.85,
        SaaqRegionClass::AlreadyCompressed => 0.9,
        SaaqRegionClass::EmbeddingHeavy => 0.55,
        SaaqRegionClass::PotentialCompressionTarget => 0.2,
        SaaqRegionClass::Unknown => 0.4,
    };
    let risk_score = clamp01(structural_penalty * 0.65 + outlier_penalty * 0.35);
    let readiness_score = clamp01(opportunity_score * (1.0 - structural_penalty * 0.85));
    let disposition = match region_class {
        SaaqRegionClass::PotentialCompressionTarget if readiness_score >= 0.15 => {
            SaaqDisposition::Candidate
        }
        SaaqRegionClass::EmbeddingHeavy if readiness_score >= 0.12 => SaaqDisposition::Candidate,
        SaaqRegionClass::RoutingCritical
        | SaaqRegionClass::NormalizationSensitive
        | SaaqRegionClass::AlreadyCompressed => SaaqDisposition::AvoidForNow,
        _ => SaaqDisposition::ObserveOnly,
    };

    let mut reasons = Vec::new();
    reasons.push(format!("distribution={}", tensor.distribution_label));
    reasons.push(format!(
        "sampled_values={}/{}",
        tensor.sample_values, tensor.total_values
    ));
    reasons.push(format!("zero_fraction={:.4}", tensor.zero_fraction));
    reasons.push(format!(
        "near_zero_fraction={:.4}",
        tensor.near_zero_fraction
    ));
    reasons.push(format!("outlier_fraction={:.4}", tensor.outlier_fraction));
    reasons.push(format!("peak_to_rms={:.3}", tensor.peak_to_rms));
    if region_class == SaaqRegionClass::RoutingCritical {
        reasons.push("linked to routing structure".to_string());
    }
    if routing
        .likely_routing_critical_blocks
        .iter()
        .any(|block| block.block_index == tensor.block_index)
        && region_class != SaaqRegionClass::RoutingCritical
    {
        reasons.push("lives in a routing-critical block".to_string());
    }

    SaaqCandidate {
        rank: 0,
        shard_ordinal: tensor.shard_ordinal,
        in_shard_index: tensor.in_shard_index,
        block_index: tensor.block_index,
        block_slot: tensor.block_slot,
        structural_name: tensor.structural_name.clone(),
        kind_label: tensor.kind_label.clone(),
        dtype: tensor.dtype,
        shape: tensor.shape.clone(),
        region_class,
        disposition,
        readiness_score,
        opportunity_score,
        risk_score,
        reasons,
    }
}

fn classify_region(
    tensor: &TensorStats,
    routing_locs: &BTreeSet<(u32, u32)>,
    routing_blocks: &BTreeSet<Option<u32>>,
) -> SaaqRegionClass {
    if routing_locs.contains(&(tensor.shard_ordinal, tensor.in_shard_index))
        || tensor.kind_label == "router"
    {
        return SaaqRegionClass::RoutingCritical;
    }
    if tensor.kind_label == "block_norm" || tensor.kind_label == "final_norm" {
        return SaaqRegionClass::NormalizationSensitive;
    }
    if tensor.dtype == TensorDType::I8
        || tensor.kind_label.starts_with("moe_scales")
        || tensor.kind_label.starts_with("moe_expert.")
    {
        return SaaqRegionClass::AlreadyCompressed;
    }
    if tensor.kind_label == "token_embedding" {
        return SaaqRegionClass::EmbeddingHeavy;
    }
    if tensor.dtype == TensorDType::F32
        && (tensor.kind_label == "attn_proj_f32"
            || tensor.kind_label == "unknown"
            || routing_blocks.contains(&tensor.block_index))
    {
        return SaaqRegionClass::PotentialCompressionTarget;
    }
    SaaqRegionClass::Unknown
}

fn summarize_layer_readiness(
    candidates: &[SaaqCandidate],
    routing_blocks: &BTreeSet<Option<u32>>,
) -> Vec<SaaqLayerReadiness> {
    let mut by_layer: BTreeMap<(Option<u32>, String), Vec<&SaaqCandidate>> = BTreeMap::new();
    for candidate in candidates {
        let label = match candidate.block_index {
            Some(index) => format!("block_{index:03}"),
            None => "unassigned".to_string(),
        };
        by_layer
            .entry((candidate.block_index, label))
            .or_default()
            .push(candidate);
    }

    by_layer
        .into_iter()
        .map(|((block_index, label), members)| {
            let candidate_target_count = members
                .iter()
                .filter(|candidate| candidate.disposition == SaaqDisposition::Candidate)
                .count() as u32;
            let mean_readiness_score = members
                .iter()
                .map(|candidate| candidate.readiness_score)
                .sum::<f64>()
                / members.len() as f64;
            let max_risk_score = members
                .iter()
                .map(|candidate| candidate.risk_score)
                .fold(0.0, f64::max);

            SaaqLayerReadiness {
                block_index,
                label,
                routing_critical: routing_blocks.contains(&block_index),
                candidate_target_count,
                mean_readiness_score,
                max_risk_score,
            }
        })
        .collect()
}

fn build_saaq_notes(
    candidate_targets: &[SaaqCandidate],
    routing_critical_tensors: &[SaaqCandidate],
    routing: &RoutingReport,
) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(top) = candidate_targets.first() {
        notes.push(format!(
            "Top candidate target is `{}` with readiness {:.3}.",
            top.structural_name, top.readiness_score
        ));
    }
    if !routing_critical_tensors.is_empty() {
        notes.push(format!(
            "{} tensors are classified as routing-critical and should be handled cautiously.",
            routing_critical_tensors.len()
        ));
    }
    if !routing.grok_layout_notes.is_empty() {
        notes.push(format!(
            "Routing analysis notes: {}",
            routing.grok_layout_notes.join(" ")
        ));
    }
    notes
}

fn routing_tensor_set(routing: &RoutingReport) -> BTreeSet<(u32, u32)> {
    routing
        .candidate_tensors
        .iter()
        .map(|tensor| (tensor.shard_ordinal, tensor.in_shard_index))
        .collect()
}

fn routing_block_set(routing: &RoutingReport) -> BTreeSet<Option<u32>> {
    routing
        .likely_routing_critical_blocks
        .iter()
        .map(|block| block.block_index)
        .collect()
}

fn layer_label_for_tensor(tensor: &TensorStats) -> String {
    match tensor.block_index {
        Some(index) => format!("block_{index:03}"),
        None if tensor.kind_label == "token_embedding" => "embedding".to_string(),
        None if tensor.kind_label == "final_norm" => "final_norm".to_string(),
        None => "unassigned".to_string(),
    }
}

fn is_potential_compression_target(tensor: &TensorStats) -> bool {
    tensor.dtype == TensorDType::F32
        && tensor.kind_label != "router"
        && tensor.kind_label != "block_norm"
        && tensor.kind_label != "final_norm"
}

fn tensor_structural_name(tensor: &TensorInfo) -> String {
    let slot = tensor
        .block_slot
        .map(|slot| format!("slot_{slot:02}"))
        .unwrap_or_else(|| "slot_na".to_string());
    match tensor.block_index {
        Some(index) => format!("block_{index:03}.{slot}.{}", tensor.kind.short_label()),
        None => match tensor.kind {
            TensorKind::TokenEmbedding => "embedding.slot_00.token_embedding".to_string(),
            TensorKind::FinalNorm => "final_norm.slot_00.final_norm".to_string(),
            _ => format!(
                "unassigned.shard_{:03}.idx_{:03}.{}",
                tensor.shard_ordinal,
                tensor.in_shard_index,
                tensor.kind.short_label()
            ),
        },
    }
}

fn clone_hyperparams(hp: &InferredHyperparams) -> InferredHyperparams {
    InferredHyperparams {
        vocab_size: hp.vocab_size,
        d_model: hp.d_model,
        n_experts: hp.n_experts,
        d_ff: hp.d_ff,
        n_blocks: hp.n_blocks,
    }
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::schema::{
        BlockSummary, InferredHyperparams, InventoryTotals, KindCount, ModelInventory,
        SaaqDisposition, SaaqRegionClass, TensorDType, TensorInfo, TensorKind, TensorRole,
        TensorShape,
    };

    use super::{StatsConfig, build_saaq_readiness_report, build_stats_report};

    #[test]
    fn stats_report_profiles_tensor_values() {
        let dir = temp_dir("stats_profile");
        let shard = dir.join("tensor00000_000");
        let mut bytes = Vec::new();
        for value in [0.0f32, 1.0, -1.0, 10.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&shard, bytes).unwrap();

        let inv = inventory(vec![tensor(
            shard,
            TensorDType::F32,
            TensorRole::Tensor,
            TensorKind::AttnProjF32,
            vec![4],
            None,
            None,
        )]);
        let stats = build_stats_report(&inv, &StatsConfig::default()).unwrap();

        assert_eq!(stats.tensors.len(), 1);
        assert!((stats.tensors[0].mean - 2.5).abs() < 1e-6);
        assert!(stats.tensors[0].max_abs >= 10.0);
        assert!(stats.tensors[0].outlier_fraction >= 0.0);
    }

    #[test]
    fn saaq_readiness_prefers_f32_attention_and_avoids_router() {
        let dir = temp_dir("saaq_readiness");
        let router_shard = dir.join("tensor00001_000");
        let attn_shard = dir.join("tensor00002_000");

        let mut router_bytes = Vec::new();
        for value in [0.0f32, 1.0, 2.0, 3.0] {
            router_bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&router_shard, router_bytes).unwrap();

        let mut attn_bytes = Vec::new();
        for value in [0.0f32, 0.0, 0.01, 0.02] {
            attn_bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&attn_shard, attn_bytes).unwrap();

        let inv = inventory(vec![
            tensor(
                router_shard,
                TensorDType::F32,
                TensorRole::Tensor,
                TensorKind::Router,
                vec![2, 2],
                Some(0),
                Some(0),
            ),
            tensor(
                attn_shard,
                TensorDType::F32,
                TensorRole::Tensor,
                TensorKind::AttnProjF32,
                vec![2, 2],
                Some(0),
                Some(1),
            ),
        ]);
        let stats = build_stats_report(&inv, &StatsConfig::default()).unwrap();
        let readiness = build_saaq_readiness_report(&inv, &stats);

        assert!(
            readiness
                .candidate_targets
                .iter()
                .any(|candidate| candidate.kind_label == "attn_proj_f32")
        );
        assert!(
            readiness
                .routing_critical_tensors
                .iter()
                .any(|candidate| candidate.region_class == SaaqRegionClass::RoutingCritical)
        );
        assert!(
            readiness
                .routing_critical_tensors
                .iter()
                .all(|candidate| candidate.disposition == SaaqDisposition::AvoidForNow)
        );
    }

    fn inventory(tensors: Vec<TensorInfo>) -> ModelInventory {
        ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1"),
            shard_count: tensors.len() as u32,
            inferred: InferredHyperparams {
                vocab_size: Some(131072),
                d_model: Some(6144),
                n_experts: Some(8),
                d_ff: Some(32768),
                n_blocks: Some(1),
            },
            tensors,
            blocks: vec![BlockSummary {
                block_index: Some(0),
                label: "block_000".to_string(),
                shard_range: None,
                tensor_count: 2,
                total_nbytes: 0,
                dtypes: Vec::new(),
                kinds: vec![KindCount {
                    kind_label: "router".to_string(),
                    count: 1,
                    nbytes: 0,
                }],
            }],
            totals: InventoryTotals {
                tensors: 2,
                quant_tensors: 0,
                f32_tensors: 2,
                i8_tensors: 0,
                total_nbytes: 32,
                total_elements: 8,
            },
            schema_version: 1,
        }
    }

    fn tensor(
        shard_path: PathBuf,
        dtype: TensorDType,
        role: TensorRole,
        kind: TensorKind,
        shape: Vec<u64>,
        block_index: Option<u32>,
        block_slot: Option<u32>,
    ) -> TensorInfo {
        TensorInfo {
            shard_path,
            shard_ordinal: block_slot.unwrap_or(0),
            in_shard_index: 0,
            role,
            dtype,
            shape: TensorShape::new(shape),
            offset: 0,
            nbytes: 16,
            kind,
            block_index,
            block_slot,
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xai_dissect_{label}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
