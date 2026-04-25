// SPDX-License-Identifier: GPL-3.0-only
//
// Routing-structure analysis derived from `ModelInventory`. This layer
// identifies candidate router tensors, summarizes their shape/orientation,
// links them to block-local expert structure, and reports layout anomalies
// without executing any routing logic.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    InferredHyperparams, ModelInventory, RoutingBlockReport, RoutingCriticalBlock,
    RoutingGateMetrics, RoutingIssue, RoutingIssueCategory, RoutingIssueSeverity,
    RoutingOrientation, RoutingOrientationSummary, RoutingReport, RoutingTensorLocator,
    RoutingTensorRef, ShardRange, TensorDType, TensorInfo, TensorKind, TensorRole, TensorShape,
};

pub const ROUTING_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone)]
struct Candidate<'a> {
    tensor: &'a TensorInfo,
    orientation: RoutingOrientation,
    expert_axis: Option<u32>,
    linked_expert_count: Option<u64>,
    structural_name: String,
    gate_metrics: RoutingGateMetrics,
}

/// Build a routing-oriented structural report from an inventory.
pub fn build_routing_report(inv: &ModelInventory) -> RoutingReport {
    let mut by_block: BTreeMap<Option<u32>, Vec<&TensorInfo>> = BTreeMap::new();
    for tensor in &inv.tensors {
        by_block.entry(tensor.block_index).or_default().push(tensor);
    }

    let mut blocks = Vec::new();
    let mut candidate_tensors = Vec::new();
    let mut anomalies = Vec::new();

    for (block_index, mut members) in by_block {
        members.sort_by_key(|t| {
            (
                t.block_slot.unwrap_or(u32::MAX),
                t.shard_ordinal,
                t.in_shard_index,
            )
        });

        let label = block_label(block_index);
        let local_expert_count = local_expert_count(&members);
        let candidates = routing_candidates(
            &label,
            &members,
            inv.inferred.n_experts,
            inv.inferred.d_model,
            local_expert_count,
        );
        let primary_candidate = select_primary_candidate(&candidates).map(locator_from_candidate);

        if local_expert_count.is_some() && candidates.is_empty() {
            anomalies.push(RoutingIssue {
                severity: RoutingIssueSeverity::Warning,
                category: RoutingIssueCategory::MissingCandidate,
                block_index,
                tensor: None,
                message: "block carries expert projections but no routing candidate was identified"
                    .to_string(),
            });
        }

        if candidates.len() > 1 {
            anomalies.push(RoutingIssue {
                severity: RoutingIssueSeverity::Warning,
                category: RoutingIssueCategory::ShapeSummary,
                block_index,
                tensor: None,
                message: format!(
                    "multiple routing candidates identified in {}: {}",
                    label,
                    candidates
                        .iter()
                        .map(|candidate| candidate.tensor.shape.render())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        }

        if block_index.is_none() && !candidates.is_empty() {
            for candidate in &candidates {
                anomalies.push(RoutingIssue {
                    severity: RoutingIssueSeverity::Warning,
                    category: RoutingIssueCategory::LayoutNote,
                    block_index,
                    tensor: Some(locator_from_candidate(candidate)),
                    message: format!(
                        "routing candidate `{}` is unassigned because inventory block assignment did not have enough layout evidence to map it to a canonical block",
                        candidate.structural_name
                    ),
                });
            }
        }

        if let Some(primary) = select_primary_candidate(&candidates) {
            if matches!(
                primary.orientation,
                RoutingOrientation::Ambiguous | RoutingOrientation::Unknown
            ) {
                anomalies.push(RoutingIssue {
                    severity: RoutingIssueSeverity::Warning,
                    category: RoutingIssueCategory::ShapeSummary,
                    block_index,
                    tensor: Some(locator_from_candidate(primary)),
                    message: format!(
                        "primary routing candidate has unclear orientation: {}",
                        primary.tensor.shape.render()
                    ),
                });
            }
            if let Some(local) = local_expert_count {
                if primary.linked_expert_count != Some(local) {
                    anomalies.push(RoutingIssue {
                        severity: RoutingIssueSeverity::Warning,
                        category: RoutingIssueCategory::ExpertCountLinkage,
                        block_index,
                        tensor: Some(locator_from_candidate(primary)),
                        message: format!(
                            "routing tensor expert count {:?} does not match local expert count {}",
                            primary.linked_expert_count, local
                        ),
                    });
                }
            }
        }

        let refs = candidates
            .iter()
            .map(|candidate| tensor_ref_from_candidate(candidate, inv.inferred.n_experts))
            .collect::<Vec<_>>();
        candidate_tensors.extend(refs.iter().cloned());

        if !refs.is_empty() || local_expert_count.is_some() {
            blocks.push(RoutingBlockReport {
                block_index,
                label,
                shard_range: shard_range(&members),
                local_expert_count,
                primary_candidate,
                candidates: refs,
            });
        }
    }

    let orientation_summaries = orientation_summaries(&candidate_tensors);
    let likely_routing_critical_blocks = likely_routing_critical_blocks(&blocks);
    let grok_layout_notes =
        grok_layout_notes(&blocks, inv.inferred.d_model, inv.inferred.n_experts);
    let relevant_block_count = blocks
        .iter()
        .filter(|block| block.block_index.is_some())
        .count() as u32;
    let expected_experts_per_router = mode_u64(
        candidate_tensors
            .iter()
            .filter_map(|tensor| tensor.linked_expert_count)
            .collect(),
    );

    RoutingReport {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        shard_count: inv.shard_count,
        inferred: InferredHyperparams {
            vocab_size: inv.inferred.vocab_size,
            d_model: inv.inferred.d_model,
            n_experts: inv.inferred.n_experts,
            d_ff: inv.inferred.d_ff,
            n_blocks: inv.inferred.n_blocks,
        },
        relevant_block_count,
        expected_experts_per_router,
        candidate_tensors,
        blocks,
        orientation_summaries,
        likely_routing_critical_blocks,
        grok_layout_notes,
        anomalies,
        schema_version: ROUTING_REPORT_SCHEMA_VERSION,
    }
}

fn routing_candidates<'a>(
    label: &str,
    members: &[&'a TensorInfo],
    inferred_experts: Option<u64>,
    d_model: Option<u64>,
    local_expert_count: Option<u64>,
) -> Vec<Candidate<'a>> {
    let mut candidates = Vec::new();
    let mut ordinal = 0u32;

    for tensor in members {
        let is_router_kind = matches!(tensor.kind, TensorKind::Router);
        let expert_axis = infer_expert_axis(tensor, inferred_experts, local_expert_count);
        let is_shape_candidate = tensor.role == TensorRole::Tensor
            && tensor.dtype == TensorDType::F32
            && tensor.shape.rank() >= 2
            && expert_axis.is_some();
        if !is_router_kind && !is_shape_candidate {
            continue;
        }

        let linked_expert_count = expert_axis.map(|axis| tensor.shape.dims()[axis as usize]);
        let orientation = infer_orientation(tensor, d_model, inferred_experts, local_expert_count);
        let structural_name = match tensor.block_slot {
            Some(slot) => format!("{label}.routing_slot_{slot:02}"),
            None => {
                let name = format!("{label}.routing_candidate_{ordinal:02}");
                ordinal += 1;
                name
            }
        };
        let gate_metrics = gate_metrics(tensor, orientation, linked_expert_count);

        candidates.push(Candidate {
            tensor,
            orientation,
            expert_axis,
            linked_expert_count,
            structural_name,
            gate_metrics,
        });
    }

    candidates
}

fn infer_expert_axis(
    tensor: &TensorInfo,
    inferred_experts: Option<u64>,
    local_expert_count: Option<u64>,
) -> Option<u32> {
    let mut matched_axes = Vec::new();
    let expected = [local_expert_count, inferred_experts];
    for (index, dim) in tensor.shape.dims().iter().copied().enumerate() {
        if expected.iter().flatten().any(|expected| *expected == dim) {
            matched_axes.push(index as u32);
        }
    }

    if matched_axes.len() == 1 {
        return matched_axes.first().copied();
    }
    if matches!(tensor.kind, TensorKind::Router) && tensor.shape.rank() == 2 {
        return Some(1);
    }
    None
}

fn infer_orientation(
    tensor: &TensorInfo,
    d_model: Option<u64>,
    inferred_experts: Option<u64>,
    local_expert_count: Option<u64>,
) -> RoutingOrientation {
    let dims = tensor.shape.dims();
    let expert_count = local_expert_count.or(inferred_experts);

    if dims.len() == 2 {
        let (a, b) = (dims[0], dims[1]);
        if let (Some(dm), Some(experts)) = (d_model, expert_count) {
            if a == dm && b == experts {
                return RoutingOrientation::DModelToExperts;
            }
            if a == experts && b == dm {
                return RoutingOrientation::ExpertsToDModel;
            }
        }
        if let Some(experts) = expert_count {
            if a == experts && b == experts {
                return RoutingOrientation::Ambiguous;
            }
            if a == experts {
                return RoutingOrientation::ExpertAxisLeading;
            }
            if b == experts {
                return RoutingOrientation::ExpertAxisTrailing;
            }
        }
        return RoutingOrientation::Unknown;
    }

    let Some(axis) = infer_expert_axis(tensor, inferred_experts, local_expert_count) else {
        return RoutingOrientation::Unknown;
    };
    if axis == 0 {
        return RoutingOrientation::ExpertAxisLeading;
    }
    if axis as usize == dims.len().saturating_sub(1) {
        return RoutingOrientation::ExpertAxisTrailing;
    }
    RoutingOrientation::Ambiguous
}

fn gate_metrics(
    tensor: &TensorInfo,
    orientation: RoutingOrientation,
    linked_expert_count: Option<u64>,
) -> RoutingGateMetrics {
    let dims = tensor.shape.dims();
    let (input_width, output_width, expert_count, logits_per_input) = if dims.len() == 2 {
        match orientation {
            RoutingOrientation::DModelToExperts => {
                (Some(dims[0]), Some(dims[1]), Some(dims[1]), Some(dims[1]))
            }
            RoutingOrientation::ExpertsToDModel => {
                (Some(dims[1]), Some(dims[0]), Some(dims[0]), Some(dims[0]))
            }
            RoutingOrientation::ExpertAxisLeading => {
                (Some(dims[1]), Some(dims[0]), Some(dims[0]), Some(dims[0]))
            }
            RoutingOrientation::ExpertAxisTrailing => {
                (Some(dims[0]), Some(dims[1]), Some(dims[1]), Some(dims[1]))
            }
            RoutingOrientation::Ambiguous | RoutingOrientation::Unknown => {
                (None, None, linked_expert_count, linked_expert_count)
            }
        }
    } else {
        (None, None, linked_expert_count, linked_expert_count)
    };

    RoutingGateMetrics {
        total_elements: tensor.shape.numel(),
        total_nbytes: tensor.nbytes,
        input_width,
        output_width,
        expert_count,
        logits_per_input,
    }
}

fn local_expert_count(members: &[&TensorInfo]) -> Option<u64> {
    mode_u64(
        members
            .iter()
            .filter_map(|tensor| match tensor.kind {
                TensorKind::MoeExpertProjection { .. } if tensor.shape.rank() == 3 => {
                    tensor.shape.dims().first().copied()
                }
                _ => None,
            })
            .collect(),
    )
}

fn select_primary_candidate<'a>(candidates: &'a [Candidate<'a>]) -> Option<&'a Candidate<'a>> {
    candidates
        .iter()
        .find(|candidate| matches!(candidate.tensor.kind, TensorKind::Router))
        .or_else(|| candidates.first())
}

fn tensor_ref_from_candidate(
    candidate: &Candidate<'_>,
    inferred_experts: Option<u64>,
) -> RoutingTensorRef {
    RoutingTensorRef {
        shard_ordinal: candidate.tensor.shard_ordinal,
        in_shard_index: candidate.tensor.in_shard_index,
        block_index: candidate.tensor.block_index,
        block_slot: candidate.tensor.block_slot,
        role: candidate.tensor.role,
        dtype: candidate.tensor.dtype,
        shape: candidate.tensor.shape.clone(),
        kind_label: candidate.tensor.kind.short_label(),
        orientation: candidate.orientation,
        expert_axis: candidate.expert_axis,
        linked_expert_count: candidate.linked_expert_count,
        matches_inferred_expert_count: candidate
            .linked_expert_count
            .map(|count| Some(count) == inferred_experts)
            .unwrap_or(false),
        structural_name: candidate.structural_name.clone(),
        gate_metrics: candidate.gate_metrics.clone(),
    }
}

fn locator_from_candidate(candidate: &Candidate<'_>) -> RoutingTensorLocator {
    RoutingTensorLocator {
        shard_ordinal: candidate.tensor.shard_ordinal,
        in_shard_index: candidate.tensor.in_shard_index,
        block_slot: candidate.tensor.block_slot,
    }
}

fn orientation_summaries(candidates: &[RoutingTensorRef]) -> Vec<RoutingOrientationSummary> {
    #[derive(Default)]
    struct SummaryAcc {
        count: u32,
        shapes: Vec<TensorShape>,
        blocks: BTreeSet<Option<u32>>,
    }

    let mut by_orientation: BTreeMap<RoutingOrientation, SummaryAcc> = BTreeMap::new();
    for candidate in candidates {
        let acc = by_orientation.entry(candidate.orientation).or_default();
        acc.count += 1;
        if !acc.shapes.contains(&candidate.shape) {
            acc.shapes.push(candidate.shape.clone());
        }
        acc.blocks.insert(candidate.block_index);
    }

    by_orientation
        .into_iter()
        .map(|(orientation, acc)| RoutingOrientationSummary {
            orientation,
            count: acc.count,
            observed_shapes: acc.shapes,
            observed_blocks: acc.blocks.len() as u32,
        })
        .collect()
}

fn likely_routing_critical_blocks(blocks: &[RoutingBlockReport]) -> Vec<RoutingCriticalBlock> {
    blocks
        .iter()
        .filter(|block| block.primary_candidate.is_some())
        .map(|block| RoutingCriticalBlock {
            block_index: block.block_index,
            label: block.label.clone(),
            reason: match block.local_expert_count {
                Some(experts) => format!(
                    "contains a primary routing candidate linked to a {}-expert MoE block",
                    experts
                ),
                None => "contains a primary routing candidate without confirmed expert linkage"
                    .to_string(),
            },
            primary_candidate: block.primary_candidate.clone(),
        })
        .collect()
}

fn grok_layout_notes(
    blocks: &[RoutingBlockReport],
    d_model: Option<u64>,
    inferred_experts: Option<u64>,
) -> Vec<String> {
    let mut notes = Vec::new();
    let primaries = blocks
        .iter()
        .filter_map(|block| {
            block
                .primary_candidate
                .as_ref()
                .map(|locator| (block, locator))
        })
        .collect::<Vec<_>>();

    if !primaries.is_empty()
        && primaries.iter().all(|(block, locator)| {
            block.candidates.iter().any(|candidate| {
                candidate.shard_ordinal == locator.shard_ordinal
                    && candidate.in_shard_index == locator.in_shard_index
                    && candidate.dtype == TensorDType::F32
                    && candidate.role == TensorRole::Tensor
                    && candidate.orientation == RoutingOrientation::DModelToExperts
            })
        })
    {
        notes.push(
            "Primary routing candidates are plain f32 tensors oriented from d_model to expert logits."
                .to_string(),
        );
    }

    let primary_slots = primaries
        .iter()
        .filter_map(|(_, locator)| locator.block_slot)
        .collect::<Vec<_>>();
    if !primary_slots.is_empty() && primary_slots.iter().all(|slot| *slot == primary_slots[0]) {
        notes.push(format!(
            "Primary routing candidates occupy a stable block slot ({}) across observed blocks.",
            primary_slots[0]
        ));
    }

    if let (Some(dm), Some(experts)) = (d_model, inferred_experts) {
        let shape = TensorShape::new(vec![dm, experts]).render();
        if primaries.iter().all(|(block, locator)| {
            block.candidates.iter().any(|candidate| {
                candidate.shard_ordinal == locator.shard_ordinal
                    && candidate.in_shard_index == locator.in_shard_index
                    && candidate.shape.dims() == [dm, experts]
            })
        }) {
            notes.push(format!(
                "Observed primary routing tensors match the Grok-style router shape `{shape}`."
            ));
        }
    }

    if blocks.iter().any(|block| block.block_index.is_none()) {
        notes.push(
            "At least one routing candidate is unassigned to a block; this usually indicates a truncated checkpoint scan or an unexpected shard layout."
                .to_string(),
        );
    }

    notes
}

fn shard_range(members: &[&TensorInfo]) -> Option<ShardRange> {
    if members.is_empty() {
        return None;
    }
    let mut start = u32::MAX;
    let mut end_inclusive = 0u32;
    for tensor in members {
        start = start.min(tensor.shard_ordinal);
        end_inclusive = end_inclusive.max(tensor.shard_ordinal);
    }
    Some(ShardRange {
        start,
        end_inclusive,
    })
}

fn block_label(block_index: Option<u32>) -> String {
    match block_index {
        Some(index) => format!("block_{index:03}"),
        None => "unassigned".to_string(),
    }
}

fn mode_u64(values: Vec<u64>) -> Option<u64> {
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(value, count)| (*count, *value))
        .map(|(value, _)| value)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::schema::{
        BlockSummary, InferredHyperparams, InventoryTotals, ModelInventory, RoutingOrientation,
        TensorDType, TensorInfo, TensorKind, TensorRole, TensorShape,
    };

    use super::build_routing_report;

    #[test]
    fn report_discovers_primary_router_and_orientation() {
        let inv = inventory(vec![
            tensor(
                2,
                Some(0),
                Some(11),
                TensorRole::Tensor,
                TensorDType::F32,
                vec![6144, 8],
                TensorKind::Router,
            ),
            tensor(
                3,
                Some(0),
                Some(1),
                TensorRole::QuantWeight,
                TensorDType::I8,
                vec![8, 6144, 32768],
                TensorKind::MoeExpertProjection {
                    projection: crate::schema::MoeProjection::Unresolved,
                },
            ),
        ]);

        let report = build_routing_report(&inv);

        assert_eq!(report.relevant_block_count, 1);
        assert_eq!(report.candidate_tensors.len(), 1);
        assert_eq!(
            report.candidate_tensors[0].orientation,
            RoutingOrientation::DModelToExperts
        );
        assert!(report.anomalies.is_empty());
    }

    #[test]
    fn report_flags_moe_block_without_router() {
        let inv = inventory(vec![tensor(
            3,
            Some(0),
            Some(1),
            TensorRole::QuantWeight,
            TensorDType::I8,
            vec![8, 6144, 32768],
            TensorKind::MoeExpertProjection {
                projection: crate::schema::MoeProjection::Unresolved,
            },
        )]);

        let report = build_routing_report(&inv);

        assert!(
            report
                .anomalies
                .iter()
                .any(|issue| issue.message.contains("no routing candidate"))
        );
    }

    #[test]
    fn report_documents_unassigned_routing_candidate() {
        let inv = inventory(vec![tensor(
            769,
            None,
            None,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![6144, 8],
            TensorKind::Router,
        )]);

        let report = build_routing_report(&inv);

        assert_eq!(report.candidate_tensors.len(), 1);
        assert_eq!(
            report.candidate_tensors[0].structural_name,
            "unassigned.routing_candidate_00"
        );
        assert!(report.anomalies.iter().any(|issue| {
            issue.category == crate::schema::RoutingIssueCategory::LayoutNote
                && issue.message.contains("is unassigned")
        }));
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
                tensor_count: 0,
                total_nbytes: 0,
                dtypes: Vec::new(),
                kinds: Vec::new(),
            }],
            totals: InventoryTotals::default(),
            schema_version: 1,
        }
    }

    fn tensor(
        shard_ordinal: u32,
        block_index: Option<u32>,
        block_slot: Option<u32>,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
        kind: TensorKind,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!("/tmp/tensor{shard_ordinal:05}_000")),
            shard_ordinal,
            in_shard_index: 0,
            role,
            dtype,
            shape: TensorShape::new(shape),
            offset: 0,
            nbytes: 0,
            kind,
            block_index,
            block_slot,
        }
    }
}
