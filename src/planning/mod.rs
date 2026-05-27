// SPDX-License-Identifier: GPL-3.0-only
//
// Static planning artifacts that bridge validated Grok-1 structure into
// downstream conversion / quantization repos without executing or mutating the
// checkpoint.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};

use crate::inventory::{GROK1_BASELINE_PROFILE, validate_grok1_complete_manifest};
use crate::schema::{
    ConversionManifest, ConversionManifestTensor, ExpertAtlas, Grok1CoverageManifest, MetricStatus,
    ModelInventory, MoeProjection, PilotBlockSelection, PilotQuantizationMode, PilotSelectionPlan,
    QuantPlan, QuantPolicy, RouteMetricStatus, RoutePreservationReport, RoutingOrientation,
    RoutingReport, SaaqCandidate, SaaqReadinessReport, SaaqRegionClass, TensorInfo, TensorKind,
};

pub const CONVERSION_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const QUANT_PLAN_SCHEMA_VERSION: u32 = 1;

const GROK1_EXPECTED_EXPERTS_PER_BLOCK: u64 = 8;
const GROK1_EXPECTED_EXPERT_FAMILIES_PER_BLOCK: u32 = 3;
const GROK1_EXPECTED_ROUTER_SHAPE: [u64; 2] = [6_144, 8];
const GROK1_KEEP_FP32_ORDER: &[&str] = &["router", "block_norm", "final_norm"];
const GROK1_PILOT_QUANTIZE_ORDER: &[&str] = &[
    "attn_proj_i8.model_width",
    "attn_proj_i8.narrow",
    "moe_expert.gate",
    "moe_expert.up",
    "moe_expert.down",
];
const GROK1_DEFER_ORDER: &[&str] = &["token_embedding"];

pub fn build_grok1_planning_artifacts(
    inv: &ModelInventory,
    atlas: &ExpertAtlas,
    routing: &RoutingReport,
    readiness: &SaaqReadinessReport,
) -> Result<(ConversionManifest, QuantPlan)> {
    let coverage = validate_grok1_clean_baseline(inv)?;
    validate_expert_atlas(atlas, &coverage)?;
    validate_routing_report(routing, &coverage)?;
    validate_readiness_groups(inv, readiness)?;

    let conversion = build_conversion_manifest(inv, atlas, routing, readiness, &coverage)?;
    let quant_plan = build_quant_plan(inv, readiness, &coverage);
    Ok((conversion, quant_plan))
}

pub fn validate_grok1_clean_baseline(inv: &ModelInventory) -> Result<Grok1CoverageManifest> {
    let coverage = validate_grok1_complete_manifest(inv)?;
    if coverage.baseline_profile != GROK1_BASELINE_PROFILE {
        bail!(
            "coverage baseline_profile {} != expected {}",
            coverage.baseline_profile,
            GROK1_BASELINE_PROFILE
        );
    }
    Ok(coverage)
}

fn validate_expert_atlas(atlas: &ExpertAtlas, coverage: &Grok1CoverageManifest) -> Result<()> {
    if atlas.relevant_block_count != coverage.discovered.blocks {
        bail!(
            "expert atlas relevant_block_count {} != baseline blocks {}",
            atlas.relevant_block_count,
            coverage.discovered.blocks
        );
    }
    if atlas.expected_experts_per_block != Some(GROK1_EXPECTED_EXPERTS_PER_BLOCK) {
        bail!(
            "expert atlas expected_experts_per_block {:?} != expected {}",
            atlas.expected_experts_per_block,
            GROK1_EXPECTED_EXPERTS_PER_BLOCK
        );
    }
    for block in &atlas.blocks {
        if block.expert_count != Some(GROK1_EXPECTED_EXPERTS_PER_BLOCK) {
            bail!(
                "expert atlas block_{:03} expert_count {:?} != expected {}",
                block.block_index,
                block.expert_count,
                GROK1_EXPECTED_EXPERTS_PER_BLOCK
            );
        }
        if block.tensors.len() != GROK1_EXPECTED_EXPERT_FAMILIES_PER_BLOCK as usize {
            bail!(
                "expert atlas block_{:03} exposes {} tensor families, expected {}",
                block.block_index,
                block.tensors.len(),
                GROK1_EXPECTED_EXPERT_FAMILIES_PER_BLOCK
            );
        }
        if block.experts.len() != GROK1_EXPECTED_EXPERTS_PER_BLOCK as usize {
            bail!(
                "expert atlas block_{:03} exposes {} expert slices, expected {}",
                block.block_index,
                block.experts.len(),
                GROK1_EXPECTED_EXPERTS_PER_BLOCK
            );
        }
    }
    Ok(())
}

fn validate_routing_report(
    routing: &RoutingReport,
    coverage: &Grok1CoverageManifest,
) -> Result<()> {
    if routing.relevant_block_count != coverage.discovered.blocks {
        bail!(
            "routing relevant_block_count {} != baseline blocks {}",
            routing.relevant_block_count,
            coverage.discovered.blocks
        );
    }
    if routing.expected_experts_per_router != Some(GROK1_EXPECTED_EXPERTS_PER_BLOCK) {
        bail!(
            "routing expected_experts_per_router {:?} != expected {}",
            routing.expected_experts_per_router,
            GROK1_EXPECTED_EXPERTS_PER_BLOCK
        );
    }
    if routing.candidate_tensors.len() as u64 != coverage.discovered.routers {
        bail!(
            "routing candidate count {} != baseline routers {}",
            routing.candidate_tensors.len(),
            coverage.discovered.routers
        );
    }
    for tensor in &routing.candidate_tensors {
        if tensor.orientation != RoutingOrientation::DModelToExperts {
            bail!(
                "routing tensor {} has orientation {}, expected d_model_to_experts",
                tensor.structural_name,
                tensor.orientation.label()
            );
        }
        if tensor.shape.dims() != GROK1_EXPECTED_ROUTER_SHAPE {
            bail!(
                "routing tensor {} has shape {}, expected (6144, 8)",
                tensor.structural_name,
                tensor.shape.render()
            );
        }
    }
    Ok(())
}

fn validate_readiness_groups(inv: &ModelInventory, readiness: &SaaqReadinessReport) -> Result<()> {
    let inventory_keys = inv
        .tensors
        .iter()
        .map(|tensor| {
            (
                (tensor.shard_ordinal, tensor.in_shard_index),
                tensor_identity_name(tensor),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut seen: BTreeMap<(u32, u32), &'static str> = BTreeMap::new();
    for (label, tensors) in [
        (
            "quantization_candidates",
            &readiness.quantization_candidates,
        ),
        (
            "routing_critical_tensors",
            &readiness.routing_critical_tensors,
        ),
        (
            "precision_sensitive_tensors",
            &readiness.precision_sensitive_tensors,
        ),
        ("deferred_tensors", &readiness.deferred_tensors),
    ] {
        for tensor in tensors {
            let key = (tensor.shard_ordinal, tensor.in_shard_index);
            if let Some(prev) = seen.insert(key, label) {
                bail!(
                    "tensor shard={} idx={} appears in both {} and {}",
                    tensor.shard_ordinal,
                    tensor.in_shard_index,
                    prev,
                    label
                );
            }
        }
    }

    let missing = inventory_keys
        .iter()
        .filter(|(key, _)| !seen.contains_key(key))
        .map(|(_, name)| name.clone())
        .collect::<Vec<_>>();
    let extra = seen
        .iter()
        .filter(|(key, _)| !inventory_keys.contains_key(key))
        .map(|((shard, index), label)| format!("shard={shard} idx={index} in {label}"))
        .collect::<Vec<_>>();
    if !missing.is_empty() || !extra.is_empty() {
        bail!(
            "readiness groups cover {} tensors but inventory has {}; missing: {}; extra: {}",
            seen.len(),
            inv.tensors.len(),
            if missing.is_empty() {
                "none".to_string()
            } else {
                missing.join(", ")
            },
            if extra.is_empty() {
                "none".to_string()
            } else {
                extra.join(", ")
            }
        );
    }

    Ok(())
}

fn build_conversion_manifest(
    inv: &ModelInventory,
    atlas: &ExpertAtlas,
    routing: &RoutingReport,
    readiness: &SaaqReadinessReport,
    coverage: &Grok1CoverageManifest,
) -> Result<ConversionManifest> {
    let groups = readiness_group_map(readiness);
    let router_shape = routing
        .candidate_tensors
        .first()
        .map(|tensor| tensor.shape.clone());
    let router_orientation = routing
        .candidate_tensors
        .first()
        .map(|tensor| tensor.orientation);

    let mut warnings = BTreeSet::new();
    let tensors = inv
        .tensors
        .iter()
        .map(|tensor| {
            let membership = groups
                .get(&(tensor.shard_ordinal, tensor.in_shard_index))
                .copied()
                .expect("validated readiness membership missing");
            let region = membership.region;
            let (quant_policy, protected_reason, entry_warnings) =
                quant_policy_for_tensor(tensor, membership.group);
            for warning in &entry_warnings {
                warnings.insert(warning.clone());
            }
            ConversionManifestTensor {
                tensor_name: tensor_identity_name(tensor),
                structural_name: structural_name(tensor),
                model_family: inv.model_family.clone(),
                block: tensor.block_index,
                slot: tensor.block_slot,
                kind: tensor.kind.short_label(),
                region,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                numel: tensor.shape.numel(),
                byte_len: tensor.nbytes,
                shard_index: tensor.shard_ordinal,
                source_shard_path: tensor.shard_path.clone(),
                source_in_shard_index: tensor.in_shard_index,
                quant_policy,
                protected_reason,
                deterministic_hash: tensor_descriptor_hash(tensor, quant_policy, region),
                warnings: entry_warnings,
            }
        })
        .collect::<Vec<_>>();

    Ok(ConversionManifest {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        baseline_profile: coverage.baseline_profile.clone(),
        required_validation: coverage.expected.clone(),
        discovered_validation: coverage.discovered.clone(),
        relevant_block_count: atlas.relevant_block_count,
        expected_experts_per_block: atlas.expected_experts_per_block,
        expert_tensor_families_per_block: Some(GROK1_EXPECTED_EXPERT_FAMILIES_PER_BLOCK),
        router_orientation,
        router_shape,
        tensors,
        warnings: warnings.into_iter().collect(),
        schema_version: CONVERSION_MANIFEST_SCHEMA_VERSION,
    })
}

fn build_quant_plan(
    inv: &ModelInventory,
    readiness: &SaaqReadinessReport,
    coverage: &Grok1CoverageManifest,
) -> QuantPlan {
    let candidate_kinds = readiness
        .quantization_candidates
        .iter()
        .map(|candidate| candidate.kind_label.as_str())
        .collect::<BTreeSet<_>>();
    let deferred_kinds = readiness
        .deferred_tensors
        .iter()
        .map(|candidate| candidate.kind_label.as_str())
        .collect::<BTreeSet<_>>();

    let keep_fp32 = GROK1_KEEP_FP32_ORDER
        .iter()
        .map(|kind| (*kind).to_string())
        .collect::<Vec<_>>();
    let pilot_quantize = GROK1_PILOT_QUANTIZE_ORDER
        .iter()
        .filter(|kind| candidate_kinds.contains(**kind))
        .map(|kind| (*kind).to_string())
        .collect::<Vec<_>>();
    let mut defer = GROK1_DEFER_ORDER
        .iter()
        .filter(|kind| deferred_kinds.contains(**kind))
        .map(|kind| (*kind).to_string())
        .collect::<Vec<_>>();
    let remaining_defer = deferred_kinds
        .into_iter()
        .filter(|kind| !GROK1_DEFER_ORDER.contains(kind))
        .map(str::to_string)
        .collect::<Vec<_>>();
    defer.extend(remaining_defer);

    let mut notes = vec![format!(
        "{} tensors partition into {} quantization candidates, {} routing-critical, {} precision-sensitive, and {} deferred entries.",
        inv.tensors.len(),
        readiness.quantization_candidates.len(),
        readiness.routing_critical_tensors.len(),
        readiness.precision_sensitive_tensors.len(),
        readiness.deferred_tensors.len()
    )];
    if let Some(top) = readiness.quantization_candidates.first() {
        notes.push(format!(
            "Top quantization candidate is `{}` with readiness {:.3}.",
            top.structural_name, top.readiness_score
        ));
    }

    QuantPlan {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        baseline: coverage.baseline_profile.clone(),
        required_validation: coverage.expected.clone(),
        discovered_validation: coverage.discovered.clone(),
        keep_fp32,
        pilot_quantize,
        defer,
        notes,
        schema_version: QUANT_PLAN_SCHEMA_VERSION,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ReadinessGroup {
    QuantizationCandidate,
    RoutingCritical,
    PrecisionSensitive,
    Deferred,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ReadinessMembership {
    group: ReadinessGroup,
    region: SaaqRegionClass,
}

fn readiness_group_map(
    readiness: &SaaqReadinessReport,
) -> BTreeMap<(u32, u32), ReadinessMembership> {
    let mut groups = BTreeMap::new();
    for candidate in &readiness.quantization_candidates {
        insert_readiness_membership(
            &mut groups,
            candidate,
            ReadinessGroup::QuantizationCandidate,
        );
    }
    for candidate in &readiness.routing_critical_tensors {
        insert_readiness_membership(&mut groups, candidate, ReadinessGroup::RoutingCritical);
    }
    for candidate in &readiness.precision_sensitive_tensors {
        insert_readiness_membership(&mut groups, candidate, ReadinessGroup::PrecisionSensitive);
    }
    for candidate in &readiness.deferred_tensors {
        insert_readiness_membership(&mut groups, candidate, ReadinessGroup::Deferred);
    }
    groups
}

fn insert_readiness_membership(
    groups: &mut BTreeMap<(u32, u32), ReadinessMembership>,
    candidate: &SaaqCandidate,
    group: ReadinessGroup,
) {
    groups.insert(
        (candidate.shard_ordinal, candidate.in_shard_index),
        ReadinessMembership {
            group,
            region: candidate.region_class,
        },
    );
}

fn quant_policy_for_tensor(
    tensor: &TensorInfo,
    group: ReadinessGroup,
) -> (QuantPolicy, Option<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let (policy, protected_reason) = match &tensor.kind {
        TensorKind::Router => (
            QuantPolicy::PassthroughF32Router,
            Some("protected router tensor; keep f32 to preserve expert selection".to_string()),
        ),
        TensorKind::BlockNorm | TensorKind::FinalNorm => (
            QuantPolicy::PassthroughF32Norm,
            Some("protected normalization tensor; keep f32 in first-pass planning".to_string()),
        ),
        TensorKind::TokenEmbedding => (QuantPolicy::CandidateSaaqEmbedding, None),
        TensorKind::MoeExpertProjection { projection } => {
            if *projection == MoeProjection::Unresolved {
                warnings.push(
                    "expert projection label remains unresolved; wrapping existing int8 tensor without projection-specific renaming"
                        .to_string(),
                );
            }
            (QuantPolicy::WrapExistingInt8Expert, None)
        }
        TensorKind::QuantizedAttentionProjection { .. } => {
            (QuantPolicy::WrapExistingInt8Unknown, None)
        }
        TensorKind::Unknown { reason } => {
            warnings.push(format!("unknown tensor classification: {reason}"));
            (QuantPolicy::UnknownPassthroughOrWarn, None)
        }
        TensorKind::MoeScales | TensorKind::AttnProjF32 => {
            warnings.push(format!(
                "{} is outside the first-pass Grok-1 conversion policy set",
                tensor.kind.short_label()
            ));
            (QuantPolicy::UnknownPassthroughOrWarn, None)
        }
    };

    if matches!(group, ReadinessGroup::Deferred)
        && !matches!(tensor.kind, TensorKind::TokenEmbedding)
    {
        warnings.push("tensor was deferred by readiness grouping".to_string());
    }

    (policy, protected_reason, warnings)
}

fn tensor_identity_name(tensor: &TensorInfo) -> String {
    let file_name = tensor
        .shard_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("tensor");
    format!("{file_name}#{}", tensor.in_shard_index)
}

fn structural_name(tensor: &TensorInfo) -> String {
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

fn tensor_descriptor_hash(
    tensor: &TensorInfo,
    policy: QuantPolicy,
    region: SaaqRegionClass,
) -> String {
    let mut parts = vec![
        format!("shard={}", tensor.shard_ordinal),
        format!("index={}", tensor.in_shard_index),
        format!("role={}", tensor.role.label()),
        format!("dtype={}", tensor.dtype.label()),
        format!("shape={}", tensor.shape.render()),
        format!("nbytes={}", tensor.nbytes),
        format!("kind={}", tensor.kind.short_label()),
        format!("policy={:?}", policy),
        format!("region={:?}", region),
    ];
    if let Some(block) = tensor.block_index {
        parts.push(format!("block={block}"));
    }
    if let Some(slot) = tensor.block_slot {
        parts.push(format!("slot={slot}"));
    }
    fnv1a64(&parts.join("|"))
}

fn fnv1a64(input: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use crate::schema::{
        CandidateTensorManifest, ExpertAtlas, ExpertBlock, InferredHyperparams, ModelInventory,
        MoeProjection, RoutingBlockReport, RoutingCriticalBlock, RoutingGateMetrics,
        RoutingOrientation, RoutingOrientationSummary, RoutingReport, RoutingTensorLocator,
        RoutingTensorRef, SaaqCandidate, SaaqDisposition, SaaqLayerReadiness, SaaqReadinessReport,
        SaaqRegionClass, ShardRange, TensorDType, TensorInfo, TensorKind, TensorRole, TensorShape,
    };

    use crate::inventory::GROK1_BASELINE_PROFILE;

    use super::{
        QUANT_PLAN_SCHEMA_VERSION, build_grok1_planning_artifacts, tensor_descriptor_hash,
    };

    #[test]
    fn planning_artifacts_use_named_grok1_baseline() {
        let (inv, atlas, routing, readiness) = complete_inputs();
        let (conversion, plan) = build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)
            .expect("planning artifacts");

        assert_eq!(conversion.baseline_profile, GROK1_BASELINE_PROFILE);
        assert_eq!(plan.baseline, GROK1_BASELINE_PROFILE);
        assert_eq!(conversion.tensors.len(), 770);
        assert_eq!(plan.schema_version, QUANT_PLAN_SCHEMA_VERSION);
        assert_eq!(plan.keep_fp32, vec!["router", "block_norm", "final_norm"]);
        assert_eq!(
            plan.pilot_quantize,
            vec![
                "attn_proj_i8.model_width",
                "attn_proj_i8.narrow",
                "moe_expert.gate",
                "moe_expert.up",
                "moe_expert.down",
            ]
        );
        assert_eq!(plan.defer, vec!["token_embedding"]);
    }

    #[test]
    fn conversion_manifest_policies_cover_all_primary_grok1_families() {
        let (inv, atlas, routing, readiness) = complete_inputs();
        let (conversion, _plan) =
            build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)
                .expect("planning artifacts");

        let mut seen = BTreeSet::new();
        for tensor in &conversion.tensors {
            seen.insert((tensor.kind.clone(), format!("{:?}", tensor.quant_policy)));
        }
        assert!(seen.contains(&(String::from("router"), String::from("PassthroughF32Router"))));
        assert!(seen.contains(&(
            String::from("block_norm"),
            String::from("PassthroughF32Norm")
        )));
        assert!(seen.contains(&(
            String::from("final_norm"),
            String::from("PassthroughF32Norm")
        )));
        assert!(seen.contains(&(
            String::from("token_embedding"),
            String::from("CandidateSaaqEmbedding")
        )));
        assert!(seen.contains(&(
            String::from("moe_expert.gate"),
            String::from("WrapExistingInt8Expert")
        )));
        assert!(seen.contains(&(
            String::from("moe_expert.down"),
            String::from("WrapExistingInt8Expert")
        )));
        assert!(seen.contains(&(
            String::from("moe_expert.up"),
            String::from("WrapExistingInt8Expert")
        )));
        assert!(seen.contains(&(
            String::from("attn_proj_i8.model_width"),
            String::from("WrapExistingInt8Unknown")
        )));
        assert!(seen.contains(&(
            String::from("attn_proj_i8.narrow"),
            String::from("WrapExistingInt8Unknown")
        )));
    }

    #[test]
    fn tensor_hash_is_deterministic() {
        let tensor = TensorInfo {
            shard_path: PathBuf::from("/tmp/grok-1/ckpt-0/tensor00013_000"),
            shard_ordinal: 13,
            in_shard_index: 0,
            role: TensorRole::Tensor,
            dtype: TensorDType::F32,
            shape: TensorShape::new(vec![6144, 8]),
            offset: 0,
            nbytes: 196_608,
            kind: TensorKind::Router,
            block_index: Some(0),
            block_slot: Some(11),
        };
        let first = tensor_descriptor_hash(
            &tensor,
            crate::schema::QuantPolicy::PassthroughF32Router,
            SaaqRegionClass::RoutingCritical,
        );
        let second = tensor_descriptor_hash(
            &tensor,
            crate::schema::QuantPolicy::PassthroughF32Router,
            SaaqRegionClass::RoutingCritical,
        );
        assert_eq!(first, second);
    }

    #[test]
    fn validate_readiness_groups_reports_extra_tensor_keys() {
        let (inv, atlas, routing, mut readiness) = complete_inputs();
        readiness.deferred_tensors.push(candidate(
            9_999,
            7,
            None,
            None,
            "unassigned.shard_9999.idx_007.unknown",
            "unknown",
            TensorDType::F32,
            vec![1],
            SaaqRegionClass::Unknown,
            SaaqDisposition::ObserveOnly,
            0.01,
        ));

        let err = build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness).unwrap_err();

        assert!(format!("{err:#}").contains("extra: shard=9999 idx=7 in deferred_tensors"));
    }

    #[test]
    fn conversion_manifest_preserves_deferred_region_class() {
        let (inv, atlas, routing, mut readiness) = complete_inputs();
        readiness.deferred_tensors[0].region_class = SaaqRegionClass::Unknown;

        let (conversion, _) = build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)
            .expect("planning artifacts");

        let embedding = conversion
            .tensors
            .iter()
            .find(|tensor| tensor.kind == "token_embedding")
            .expect("token embedding entry");
        assert_eq!(embedding.region, SaaqRegionClass::Unknown);
    }

    #[test]
    fn conversion_manifest_token_embedding_is_only_saaq_candidate() {
        let (inv, atlas, routing, readiness) = complete_inputs();
        let (conversion, _) = build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)
            .expect("planning artifacts");

        let saaq_candidates: Vec<_> = conversion
            .tensors
            .iter()
            .filter(|t| t.quant_policy == crate::schema::QuantPolicy::CandidateSaaqEmbedding)
            .collect();

        assert_eq!(
            saaq_candidates.len(),
            1,
            "token_embedding should be the only CandidateSaaqEmbedding"
        );
        assert_eq!(saaq_candidates[0].kind, "token_embedding");
    }

    #[test]
    fn conversion_manifest_warns_on_unresolved_projections() {
        let unresolved = tensor(
            999,
            0,
            Some(0),
            Some(0),
            TensorKind::MoeExpertProjection {
                projection: MoeProjection::Unresolved,
            },
            TensorRole::QuantWeight,
            TensorDType::I8,
            vec![8, 6_144, 32_768],
        );
        let (policy, _protected, warnings) = super::quant_policy_for_tensor(
            &unresolved,
            super::ReadinessGroup::QuantizationCandidate,
        );
        assert_eq!(
            policy,
            crate::schema::QuantPolicy::WrapExistingInt8Expert,
            "unresolved projection should still map to WrapExistingInt8Expert"
        );
        assert!(
            warnings.iter().any(|w| w.contains("unresolved")),
            "expected warning about unresolved projection"
        );
    }

    #[test]
    fn conversion_manifest_warns_on_unknown_tensors() {
        let unknown = tensor(
            999,
            0,
            None,
            None,
            TensorKind::Unknown {
                reason: "dense slot does not match any known signature".into(),
            },
            TensorRole::Tensor,
            TensorDType::F32,
            vec![1],
        );
        let (policy, _protected, warnings) =
            super::quant_policy_for_tensor(&unknown, super::ReadinessGroup::Deferred);
        assert_eq!(
            policy,
            crate::schema::QuantPolicy::UnknownPassthroughOrWarn,
            "unknown tensor should map to UnknownPassthroughOrWarn"
        );
        assert!(
            warnings.iter().any(|w| w.contains("unknown")),
            "expected warning about unknown tensor classification"
        );
    }

    fn complete_inputs() -> (
        ModelInventory,
        ExpertAtlas,
        RoutingReport,
        SaaqReadinessReport,
    ) {
        let inv = complete_inventory();
        let atlas = complete_expert_atlas();
        let routing = complete_routing_report();
        let readiness = complete_readiness();
        (inv, atlas, routing, readiness)
    }

    fn complete_inventory() -> ModelInventory {
        let mut tensors = vec![tensor(
            0,
            0,
            None,
            None,
            TensorKind::TokenEmbedding,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![131_072, 6_144],
        )];
        tensors.push(tensor(
            1,
            0,
            None,
            None,
            TensorKind::FinalNorm,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![6_144],
        ));
        for block in 0..64u32 {
            for slot in 0..12u32 {
                let shard = 2 + block * 12 + slot;
                let (kind, role, dtype, shape) = match slot {
                    0 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Gate,
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
                    2 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Up,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6_144, 32_768],
                    ),
                    3 | 6 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: crate::schema::QuantizedAttentionWidth::Narrow,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 1_024],
                    ),
                    4 | 5 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: crate::schema::QuantizedAttentionWidth::ModelWidth,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 6_144],
                    ),
                    7..=10 => (
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
                tensors.push(tensor(
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
        let blocks = crate::inventory::summarize_blocks(&tensors);
        let totals = compute_totals(&tensors);
        ModelInventory {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 770,
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
            schema_version: crate::inventory::SCHEMA_VERSION,
        }
    }

    fn complete_expert_atlas() -> ExpertAtlas {
        let blocks = (0..64u32)
            .map(|block| ExpertBlock {
                block_index: block,
                shard_range: Some(ShardRange {
                    start: 2 + block * 12,
                    end_inclusive: 4 + block * 12,
                }),
                expert_count: Some(8),
                tensors: vec![
                    crate::schema::ExpertTensorRef {
                        shard_ordinal: 2 + block * 12,
                        in_shard_index: 0,
                        block_slot: Some(0),
                        role: TensorRole::QuantWeight,
                        dtype: TensorDType::I8,
                        shape: TensorShape::new(vec![8, 6_144, 32_768]),
                        kind_label: "moe_expert.gate".into(),
                        projection: MoeProjection::Gate,
                        expert_axis: Some(0),
                        expert_count: Some(8),
                        family_label: "expert_slot_00".into(),
                        structural_name: format!("block_{block:03}.expert_slot_00"),
                    },
                    crate::schema::ExpertTensorRef {
                        shard_ordinal: 3 + block * 12,
                        in_shard_index: 0,
                        block_slot: Some(1),
                        role: TensorRole::QuantWeight,
                        dtype: TensorDType::I8,
                        shape: TensorShape::new(vec![8, 32_768, 6_144]),
                        kind_label: "moe_expert.down".into(),
                        projection: MoeProjection::Down,
                        expert_axis: Some(0),
                        expert_count: Some(8),
                        family_label: "expert_slot_01".into(),
                        structural_name: format!("block_{block:03}.expert_slot_01"),
                    },
                    crate::schema::ExpertTensorRef {
                        shard_ordinal: 4 + block * 12,
                        in_shard_index: 0,
                        block_slot: Some(2),
                        role: TensorRole::QuantWeight,
                        dtype: TensorDType::I8,
                        shape: TensorShape::new(vec![8, 6_144, 32_768]),
                        kind_label: "moe_expert.up".into(),
                        projection: MoeProjection::Up,
                        expert_axis: Some(0),
                        expert_count: Some(8),
                        family_label: "expert_slot_02".into(),
                        structural_name: format!("block_{block:03}.expert_slot_02"),
                    },
                ],
                experts: (0..8u32)
                    .map(|expert| crate::schema::ExpertSlice {
                        expert_index: expert,
                        tensors: vec![
                            crate::schema::ExpertSliceTensor {
                                family_label: "expert_slot_00".into(),
                                structural_name: format!(
                                    "block_{block:03}.expert_slot_00.expert_{expert:02}"
                                ),
                                source_shard_ordinal: 2 + block * 12,
                                source_in_shard_index: 0,
                                source_block_slot: Some(0),
                                projection: MoeProjection::Gate,
                                dtype: TensorDType::I8,
                                slice_shape: TensorShape::new(vec![6_144, 32_768]),
                            },
                            crate::schema::ExpertSliceTensor {
                                family_label: "expert_slot_01".into(),
                                structural_name: format!(
                                    "block_{block:03}.expert_slot_01.expert_{expert:02}"
                                ),
                                source_shard_ordinal: 3 + block * 12,
                                source_in_shard_index: 0,
                                source_block_slot: Some(1),
                                projection: MoeProjection::Down,
                                dtype: TensorDType::I8,
                                slice_shape: TensorShape::new(vec![32_768, 6_144]),
                            },
                            crate::schema::ExpertSliceTensor {
                                family_label: "expert_slot_02".into(),
                                structural_name: format!(
                                    "block_{block:03}.expert_slot_02.expert_{expert:02}"
                                ),
                                source_shard_ordinal: 4 + block * 12,
                                source_in_shard_index: 0,
                                source_block_slot: Some(2),
                                projection: MoeProjection::Up,
                                dtype: TensorDType::I8,
                                slice_shape: TensorShape::new(vec![6_144, 32_768]),
                            },
                        ],
                    })
                    .collect(),
            })
            .collect();
        ExpertAtlas {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                d_model: Some(6_144),
                n_experts: Some(8),
                d_ff: Some(32_768),
                n_blocks: Some(64),
                ..Default::default()
            },
            relevant_block_count: 64,
            expected_experts_per_block: Some(8),
            blocks,
            naming_patterns: Vec::new(),
            naming_checks: Vec::new(),
            anomalies: Vec::new(),
            schema_version: 1,
        }
    }

    fn complete_routing_report() -> RoutingReport {
        let candidates = (0..64u32)
            .map(|block| RoutingTensorRef {
                shard_ordinal: 13 + block * 12,
                in_shard_index: 0,
                block_index: Some(block),
                block_slot: Some(11),
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![6_144, 8]),
                kind_label: "router".into(),
                orientation: RoutingOrientation::DModelToExperts,
                expert_axis: Some(1),
                linked_expert_count: Some(8),
                matches_inferred_expert_count: true,
                structural_name: format!("block_{block:03}.routing_slot_11"),
                gate_metrics: RoutingGateMetrics {
                    total_elements: 49_152,
                    total_nbytes: 196_608,
                    input_width: Some(6_144),
                    output_width: Some(8),
                    expert_count: Some(8),
                    logits_per_input: Some(8),
                },
            })
            .collect::<Vec<_>>();
        RoutingReport {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                d_model: Some(6_144),
                n_experts: Some(8),
                n_blocks: Some(64),
                ..Default::default()
            },
            relevant_block_count: 64,
            expected_experts_per_router: Some(8),
            candidate_tensors: candidates.clone(),
            blocks: (0..64u32)
                .map(|block| RoutingBlockReport {
                    block_index: Some(block),
                    label: format!("block_{block:03}"),
                    shard_range: Some(ShardRange {
                        start: 2 + block * 12,
                        end_inclusive: 13 + block * 12,
                    }),
                    local_expert_count: Some(8),
                    primary_candidate: Some(RoutingTensorLocator {
                        shard_ordinal: 13 + block * 12,
                        in_shard_index: 0,
                        block_slot: Some(11),
                    }),
                    candidates: vec![candidates[block as usize].clone()],
                })
                .collect(),
            orientation_summaries: vec![RoutingOrientationSummary {
                orientation: RoutingOrientation::DModelToExperts,
                count: 64,
                observed_shapes: vec![TensorShape::new(vec![6_144, 8])],
                observed_blocks: 64,
            }],
            likely_routing_critical_blocks: (0..64u32)
                .map(|block| RoutingCriticalBlock {
                    block_index: Some(block),
                    label: format!("block_{block:03}"),
                    reason: "primary router present".into(),
                    primary_candidate: Some(RoutingTensorLocator {
                        shard_ordinal: 13 + block * 12,
                        in_shard_index: 0,
                        block_slot: Some(11),
                    }),
                })
                .collect(),
            grok_layout_notes: Vec::new(),
            anomalies: Vec::new(),
            schema_version: 1,
        }
    }

    fn complete_readiness() -> SaaqReadinessReport {
        let mut quantization_candidates = Vec::new();
        let mut routing_critical_tensors = Vec::new();
        let mut precision_sensitive_tensors = Vec::new();
        let mut deferred_tensors = vec![candidate(
            0,
            0,
            None,
            None,
            "embedding.slot_00.token_embedding",
            "token_embedding",
            TensorDType::F32,
            vec![131_072, 6_144],
            SaaqRegionClass::EmbeddingHeavy,
            SaaqDisposition::ObserveOnly,
            0.08,
        )];
        for block in 0..64u32 {
            for (slot, kind, dtype, shape, readiness_score) in [
                (
                    0,
                    "moe_expert.gate",
                    TensorDType::I8,
                    vec![8, 6_144, 32_768],
                    0.82,
                ),
                (
                    1,
                    "moe_expert.down",
                    TensorDType::I8,
                    vec![8, 32_768, 6_144],
                    0.79,
                ),
                (
                    2,
                    "moe_expert.up",
                    TensorDType::I8,
                    vec![8, 6_144, 32_768],
                    0.81,
                ),
                (
                    3,
                    "attn_proj_i8.narrow",
                    TensorDType::I8,
                    vec![6_144, 1_024],
                    0.55,
                ),
                (
                    4,
                    "attn_proj_i8.model_width",
                    TensorDType::I8,
                    vec![6_144, 6_144],
                    0.63,
                ),
                (
                    5,
                    "attn_proj_i8.model_width",
                    TensorDType::I8,
                    vec![6_144, 6_144],
                    0.61,
                ),
                (
                    6,
                    "attn_proj_i8.narrow",
                    TensorDType::I8,
                    vec![6_144, 1_024],
                    0.54,
                ),
            ] {
                quantization_candidates.push(candidate(
                    2 + block * 12 + slot,
                    0,
                    Some(block),
                    Some(slot),
                    &format!("block_{block:03}.slot_{slot:02}.{kind}"),
                    kind,
                    dtype,
                    shape,
                    SaaqRegionClass::PotentialCompressionTarget,
                    SaaqDisposition::Candidate,
                    readiness_score,
                ));
            }
            for slot in 7..=10u32 {
                precision_sensitive_tensors.push(candidate(
                    2 + block * 12 + slot,
                    0,
                    Some(block),
                    Some(slot),
                    &format!("block_{block:03}.slot_{slot:02}.block_norm"),
                    "block_norm",
                    TensorDType::F32,
                    vec![6_144],
                    SaaqRegionClass::NormalizationSensitive,
                    SaaqDisposition::AvoidForNow,
                    0.04,
                ));
            }
            routing_critical_tensors.push(candidate(
                13 + block * 12,
                0,
                Some(block),
                Some(11),
                &format!("block_{block:03}.slot_11.router"),
                "router",
                TensorDType::F32,
                vec![6_144, 8],
                SaaqRegionClass::RoutingCritical,
                SaaqDisposition::AvoidForNow,
                0.03,
            ));
        }
        precision_sensitive_tensors.push(candidate(
            1,
            0,
            None,
            None,
            "final_norm.slot_00.final_norm",
            "final_norm",
            TensorDType::F32,
            vec![6_144],
            SaaqRegionClass::NormalizationSensitive,
            SaaqDisposition::AvoidForNow,
            0.02,
        ));

        quantization_candidates.sort_by(|a, b| b.readiness_score.total_cmp(&a.readiness_score));
        for (idx, candidate) in quantization_candidates.iter_mut().enumerate() {
            candidate.rank = (idx + 1) as u32;
        }
        for (idx, candidate) in routing_critical_tensors.iter_mut().enumerate() {
            candidate.rank = (idx + 1) as u32;
        }
        for (idx, candidate) in precision_sensitive_tensors.iter_mut().enumerate() {
            candidate.rank = (idx + 1) as u32;
        }
        for (idx, candidate) in deferred_tensors.iter_mut().enumerate() {
            candidate.rank = (idx + 1) as u32;
        }

        SaaqReadinessReport {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                d_model: Some(6_144),
                n_experts: Some(8),
                d_ff: Some(32_768),
                n_blocks: Some(64),
                ..Default::default()
            },
            candidate_targets: quantization_candidates.clone(),
            quantization_candidates: quantization_candidates.clone(),
            routing_critical_tensors: routing_critical_tensors.clone(),
            precision_sensitive_tensors: precision_sensitive_tensors.clone(),
            deferred_tensors: deferred_tensors.clone(),
            risky_tensors: routing_critical_tensors.iter().take(10).cloned().collect(),
            layer_readiness: vec![SaaqLayerReadiness {
                block_index: Some(0),
                label: "block_000".into(),
                routing_critical: true,
                candidate_target_count: 7,
                mean_readiness_score: 0.46,
                max_risk_score: 0.93,
            }],
            notes: vec![
                "Top candidate target is `block_000.slot_00.moe_expert.gate` with readiness 0.820."
                    .into(),
            ],
            manifest: CandidateTensorManifest {
                model_family: "grok-1".into(),
                checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
                candidates: quantization_candidates,
                schema_version: 1,
            },
            schema_version: 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn candidate(
        shard_ordinal: u32,
        in_shard_index: u32,
        block_index: Option<u32>,
        block_slot: Option<u32>,
        structural_name: &str,
        kind_label: &str,
        dtype: TensorDType,
        shape: Vec<u64>,
        region_class: SaaqRegionClass,
        disposition: SaaqDisposition,
        readiness_score: f64,
    ) -> SaaqCandidate {
        SaaqCandidate {
            rank: 0,
            shard_ordinal,
            in_shard_index,
            block_index,
            block_slot,
            structural_name: structural_name.into(),
            kind_label: kind_label.into(),
            dtype,
            shape: TensorShape::new(shape),
            region_class,
            disposition,
            readiness_score,
            opportunity_score: readiness_score,
            risk_score: 1.0 - readiness_score,
            reasons: vec!["synthetic planning fixture".into()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tensor(
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
            shard_path: PathBuf::from(format!(
                "/tmp/grok-1-official/ckpt-0/tensor{shard_ordinal:05}_000"
            )),
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

    fn compute_totals(tensors: &[TensorInfo]) -> crate::schema::InventoryTotals {
        let mut totals = crate::schema::InventoryTotals {
            tensors: tensors.len() as u64,
            ..Default::default()
        };
        for tensor in tensors {
            totals.total_nbytes += tensor.nbytes;
            totals.total_elements += tensor.shape.numel();
            match tensor.dtype {
                TensorDType::F32 => totals.f32_tensors += 1,
                TensorDType::I8 => totals.i8_tensors += 1,
            }
            match tensor.role {
                TensorRole::QuantWeight | TensorRole::QuantScales => totals.quant_tensors += 1,
                TensorRole::Tensor => {}
            }
        }
        totals
    }
}

pub const PILOT_SELECTION_PLAN_SCHEMA_VERSION: u32 = 1;
pub const ROUTE_PRESERVATION_REPORT_SCHEMA_VERSION: u32 = 1;
const GROK1_PILOT_BLOCKS: &[(u32, &str, &str)] = &[
    (0, "block_000", "early baseline"),
    (8, "block_008", "near-zero-sensitive router"),
    (28, "block_028", "near-zero-sensitive router"),
    (60, "block_060", "high readiness/routing-critical sample"),
    (
        63,
        "block_063",
        "late-layer / high peak-to-rms router region",
    ),
];

pub fn build_grok1_pilot_selection_plan(inv: &ModelInventory) -> Result<PilotSelectionPlan> {
    let coverage = validate_grok1_clean_baseline(inv)?;
    Ok(PilotSelectionPlan {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        baseline: coverage.baseline_profile.clone(),
        required_validation: coverage.expected.clone(),
        selected_blocks: GROK1_PILOT_BLOCKS
            .iter()
            .map(|(block_index, label, rationale)| PilotBlockSelection {
                block_index: *block_index,
                label: (*label).to_string(),
                rationale: (*rationale).to_string(),
            })
            .collect(),
        modes: vec![
            PilotQuantizationMode::AttentionOnly,
            PilotQuantizationMode::ExpertOnly,
            PilotQuantizationMode::AttentionPlusExpert,
        ],
        protection_rules: vec![
            "router tensors must remain untouched in every first-pass pilot".to_string(),
            "block_norm and final_norm tensors must remain untouched in every first-pass pilot"
                .to_string(),
            "pilot artifacts must be emitted per mode and remain comparable across selected blocks"
                .to_string(),
        ],
        comparison_artifacts: vec![
            "pilot-selection-plan.json".to_string(),
            "pilot-selection-plan.md".to_string(),
            "route-preservation-report.json".to_string(),
            "route-preservation-report.md".to_string(),
        ],
        notes: vec![
            "This is a planning artifact only; xai-dissect does not mutate checkpoints or execute a quantization runtime.".to_string(),
            "Use the selected blocks and protected-family rules to drive downstream bounded pilot runs.".to_string(),
        ],
        schema_version: PILOT_SELECTION_PLAN_SCHEMA_VERSION,
    })
}

pub fn build_grok1_route_preservation_report(
    inv: &ModelInventory,
) -> Result<RoutePreservationReport> {
    let coverage = validate_grok1_clean_baseline(inv)?;
    let router_metrics = vec![
        RouteMetricStatus {
            name: "router_top1_agreement".to_string(),
            scope: "router_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: Some(">= 99.0%".to_string()),
            observed: None,
            detail: "Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference.".to_string(),
        },
        RouteMetricStatus {
            name: "router_top2_set_agreement".to_string(),
            scope: "router_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: Some(">= 99.5%".to_string()),
            observed: None,
            detail: "Threshold reserved for downstream pilot comparison artifacts; report as first-class sprint evidence when available.".to_string(),
        },
        RouteMetricStatus {
            name: "expert_load_distribution_delta".to_string(),
            scope: "router_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Capture expert-load distribution drift once bounded pilot routing traces are available.".to_string(),
        },
        RouteMetricStatus {
            name: "expert_load_js_divergence".to_string(),
            scope: "router_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report JS/KL-style divergence over expert-load distributions when downstream pilot evidence exists.".to_string(),
        },
        RouteMetricStatus {
            name: "router_logit_rank_correlation".to_string(),
            scope: "router_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report rank correlation for router logits when logits are captured by downstream pilot comparisons.".to_string(),
        },
    ];
    let block_metrics = vec![
        RouteMetricStatus {
            name: "block_output_cosine".to_string(),
            scope: "block_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: Some(">= 0.995".to_string()),
            observed: None,
            detail: "Tracked as a go/no-go threshold once bounded pilot outputs exist.".to_string(),
        },
        RouteMetricStatus {
            name: "block_output_rmse".to_string(),
            scope: "block_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report alongside cosine similarity for bounded pilot comparisons.".to_string(),
        },
        RouteMetricStatus {
            name: "residual_stream_drift".to_string(),
            scope: "block_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Summarize residual-stream drift once downstream pilot artifacts provide comparable block activations.".to_string(),
        },
    ];
    let weight_metrics = vec![
        RouteMetricStatus {
            name: "weight_reconstruction_mse".to_string(),
            scope: "weight_reconstruction".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Generic reconstruction metrics remain secondary to router-behavior preservation for Grok-1 MoE validation.".to_string(),
        },
        RouteMetricStatus {
            name: "weight_cosine_similarity".to_string(),
            scope: "weight_reconstruction".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Useful companion metric, but not sufficient by itself to clear a full quantization run.".to_string(),
        },
        RouteMetricStatus {
            name: "weight_max_absolute_error".to_string(),
            scope: "weight_reconstruction".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report max absolute reconstruction error when downstream pilot comparisons include raw tensor deltas.".to_string(),
        },
        RouteMetricStatus {
            name: "per_channel_scale_error_summary".to_string(),
            scope: "weight_reconstruction".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Summarize per-channel scale/error drift where quantization metadata is available.".to_string(),
        },
        RouteMetricStatus {
            name: "logit_kl".to_string(),
            scope: "model_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report-only placeholder for model/logit KL when downstream pilot inference captures logits.".to_string(),
        },
        RouteMetricStatus {
            name: "perplexity_delta".to_string(),
            scope: "model_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report-only placeholder for calibration-data perplexity delta once downstream pilot evaluation exists.".to_string(),
        },
        RouteMetricStatus {
            name: "generation_sanity_summary".to_string(),
            scope: "model_behavior".to_string(),
            status: MetricStatus::Unknown,
            threshold: None,
            observed: None,
            detail: "Report-only placeholder for short generation sanity checks when pilot inference is available.".to_string(),
        },
    ];
    let mut summary = Vec::new();
    summary.extend(router_metrics.iter().cloned());
    summary.extend(block_metrics.iter().cloned());
    summary.extend(weight_metrics.iter().cloned());
    Ok(RoutePreservationReport {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        baseline: coverage.baseline_profile.clone(),
        required_validation: coverage.expected.clone(),
        summary,
        router_metrics,
        block_metrics,
        weight_metrics,
        notes: vec![
            "This report defines the required route-preservation surface and thresholds for Grok-1 pilot evidence.".to_string(),
            "Statuses remain unknown until downstream pilot artifacts supply the observed values.".to_string(),
        ],
        schema_version: ROUTE_PRESERVATION_REPORT_SCHEMA_VERSION,
    })
}
