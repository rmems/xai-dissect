#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use xai_dissect::exports::OutputBundle;
use xai_dissect::inventory;
use xai_dissect::schema::{
    BlockSummary, CandidateTensorManifest, ConversionManifest, ConversionManifestTensor,
    ExpertAtlas, ExpertBlock, ExpertNamingCheck, ExpertNamingPattern, ExpertSlice,
    ExpertSliceTensor, ExpertTensorRef, Grok1CoverageCounts, InferredHyperparams, InventoryTotals,
    KindCount, LayerStats, MetricStatus, ModelInventory, MoeProjection, NormSummary,
    OutlierSummary, PilotBlockSelection, PilotQuantizationMode, PilotSelectionPlan, QuantPlan,
    QuantPolicy, RankedTensorStat, RouteMetricStatus, RoutePreservationReport, RoutingBlockReport,
    RoutingCriticalBlock, RoutingGateMetrics, RoutingOrientation, RoutingOrientationSummary,
    RoutingReport, RoutingTensorLocator, RoutingTensorRef, SaaqCandidate, SaaqDisposition,
    SaaqLayerReadiness, SaaqReadinessReport, SaaqRegionClass, ShardRange, StatsProfileReport,
    StatsSamplingConfig, TensorDType, TensorInfo, TensorKind, TensorRole, TensorShape, TensorStats,
    VarianceSummary,
};

pub const SNAPSHOT_ENV: &str = "XAI_DISSECT_WRITE_SNAPSHOTS";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn sample_checkpoint_path() -> PathBuf {
    PathBuf::from("/fixtures/grok-1-official/ckpt-0")
}

pub fn sample_checkpoint_slug() -> &'static str {
    "grok-1-official__ckpt-0"
}

pub fn unique_temp_root(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("xai-dissect-{prefix}-{stamp}-{counter}"))
}

pub fn render_snapshot_sections(sections: &[(String, String)]) -> String {
    let mut rendered = String::new();
    for (idx, (name, body)) in sections.iter().enumerate() {
        if idx > 0 {
            rendered.push('\n');
        }
        rendered.push_str("=== ");
        rendered.push_str(name);
        rendered.push_str(" ===\n");
        rendered.push_str(body.trim_end());
        rendered.push('\n');
    }
    rendered
}

pub fn assert_snapshot_sections(snapshot_path: &str, sections: &[(String, String)]) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(snapshot_path);
    let actual = render_snapshot_sections(sections);
    if std::env::var_os(SNAPSHOT_ENV).is_some() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create snapshot dir");
        }
        fs::write(&path, actual).expect("write snapshot");
        return;
    }

    let expected = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read snapshot {}: {err}", path.display()));
    assert_eq!(expected, actual, "snapshot mismatch for {}", path.display());
}

pub fn bundle_sections(root: &Path, bundle: &OutputBundle) -> Vec<(String, String)> {
    let mut rels = bundle
        .written_paths
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .expect("path under bundle root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<Vec<_>>();
    rels.sort();

    rels.into_iter()
        .map(|rel| {
            let body = fs::read_to_string(root.join(&rel))
                .unwrap_or_else(|err| panic!("read generated artifact {rel}: {err}"));
            (rel, body)
        })
        .collect()
}

pub fn sample_inventory() -> ModelInventory {
    ModelInventory {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        shard_count: 2,
        inferred: InferredHyperparams {
            vocab_size: Some(16),
            d_model: Some(4),
            n_experts: Some(2),
            d_ff: Some(6),
            n_blocks: Some(1),
        },
        tensors: vec![
            TensorInfo {
                shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0000"),
                shard_ordinal: 0,
                in_shard_index: 0,
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![16, 4]),
                offset: 128,
                nbytes: 256,
                kind: TensorKind::TokenEmbedding,
                block_index: None,
                block_slot: None,
            },
            TensorInfo {
                shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0001"),
                shard_ordinal: 1,
                in_shard_index: 0,
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4, 2]),
                offset: 32,
                nbytes: 32,
                kind: TensorKind::Router,
                block_index: Some(0),
                block_slot: Some(0),
            },
            TensorInfo {
                shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0001"),
                shard_ordinal: 1,
                in_shard_index: 1,
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4, 4]),
                offset: 96,
                nbytes: 64,
                kind: TensorKind::AttnProjF32,
                block_index: Some(0),
                block_slot: Some(1),
            },
            TensorInfo {
                shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0001"),
                shard_ordinal: 1,
                in_shard_index: 2,
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4]),
                offset: 176,
                nbytes: 16,
                kind: TensorKind::BlockNorm,
                block_index: Some(0),
                block_slot: Some(2),
            },
        ],
        blocks: vec![
            BlockSummary {
                block_index: None,
                label: "embedding".into(),
                shard_range: Some(ShardRange {
                    start: 0,
                    end_inclusive: 0,
                }),
                tensor_count: 1,
                total_nbytes: 256,
                dtypes: vec![TensorDType::F32],
                kinds: vec![KindCount {
                    kind_label: "token_embedding".into(),
                    count: 1,
                    nbytes: 256,
                }],
            },
            BlockSummary {
                block_index: Some(0),
                label: "block_000".into(),
                shard_range: Some(ShardRange {
                    start: 1,
                    end_inclusive: 1,
                }),
                tensor_count: 3,
                total_nbytes: 112,
                dtypes: vec![TensorDType::F32],
                kinds: vec![
                    KindCount {
                        kind_label: "router".into(),
                        count: 1,
                        nbytes: 32,
                    },
                    KindCount {
                        kind_label: "attn_proj_f32".into(),
                        count: 1,
                        nbytes: 64,
                    },
                    KindCount {
                        kind_label: "block_norm".into(),
                        count: 1,
                        nbytes: 16,
                    },
                ],
            },
        ],
        totals: InventoryTotals {
            tensors: 4,
            quant_tensors: 0,
            f32_tensors: 4,
            i8_tensors: 0,
            total_nbytes: 368,
            total_elements: 92,
        },
        schema_version: inventory::SCHEMA_VERSION,
    }
}

pub fn sample_expert_atlas() -> ExpertAtlas {
    ExpertAtlas {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        shard_count: 3,
        inferred: InferredHyperparams {
            d_model: Some(4),
            n_experts: Some(2),
            d_ff: Some(6),
            n_blocks: Some(1),
            ..Default::default()
        },
        relevant_block_count: 1,
        expected_experts_per_block: Some(2),
        blocks: vec![ExpertBlock {
            block_index: 0,
            shard_range: Some(ShardRange {
                start: 1,
                end_inclusive: 2,
            }),
            expert_count: Some(2),
            tensors: vec![
                ExpertTensorRef {
                    shard_ordinal: 1,
                    in_shard_index: 0,
                    block_slot: Some(3),
                    role: TensorRole::QuantWeight,
                    dtype: TensorDType::I8,
                    shape: TensorShape::new(vec![2, 4, 6]),
                    kind_label: "moe_expert.unresolved".into(),
                    projection: MoeProjection::Unresolved,
                    expert_axis: Some(0),
                    expert_count: Some(2),
                    family_label: "ffn_up_or_gate".into(),
                    structural_name: "block_000.slot_03.ffn_up_or_gate".into(),
                },
                ExpertTensorRef {
                    shard_ordinal: 2,
                    in_shard_index: 0,
                    block_slot: Some(4),
                    role: TensorRole::QuantWeight,
                    dtype: TensorDType::I8,
                    shape: TensorShape::new(vec![2, 6, 4]),
                    kind_label: "moe_expert.down".into(),
                    projection: MoeProjection::Down,
                    expert_axis: Some(0),
                    expert_count: Some(2),
                    family_label: "ffn_down".into(),
                    structural_name: "block_000.slot_04.ffn_down".into(),
                },
            ],
            experts: vec![
                ExpertSlice {
                    expert_index: 0,
                    tensors: vec![
                        ExpertSliceTensor {
                            family_label: "ffn_up_or_gate".into(),
                            structural_name: "block_000.expert_00.ffn_up_or_gate".into(),
                            source_shard_ordinal: 1,
                            source_in_shard_index: 0,
                            source_block_slot: Some(3),
                            projection: MoeProjection::Unresolved,
                            dtype: TensorDType::I8,
                            slice_shape: TensorShape::new(vec![4, 6]),
                        },
                        ExpertSliceTensor {
                            family_label: "ffn_down".into(),
                            structural_name: "block_000.expert_00.ffn_down".into(),
                            source_shard_ordinal: 2,
                            source_in_shard_index: 0,
                            source_block_slot: Some(4),
                            projection: MoeProjection::Down,
                            dtype: TensorDType::I8,
                            slice_shape: TensorShape::new(vec![6, 4]),
                        },
                    ],
                },
                ExpertSlice {
                    expert_index: 1,
                    tensors: vec![
                        ExpertSliceTensor {
                            family_label: "ffn_up_or_gate".into(),
                            structural_name: "block_000.expert_01.ffn_up_or_gate".into(),
                            source_shard_ordinal: 1,
                            source_in_shard_index: 0,
                            source_block_slot: Some(3),
                            projection: MoeProjection::Unresolved,
                            dtype: TensorDType::I8,
                            slice_shape: TensorShape::new(vec![4, 6]),
                        },
                        ExpertSliceTensor {
                            family_label: "ffn_down".into(),
                            structural_name: "block_000.expert_01.ffn_down".into(),
                            source_shard_ordinal: 2,
                            source_in_shard_index: 0,
                            source_block_slot: Some(4),
                            projection: MoeProjection::Down,
                            dtype: TensorDType::I8,
                            slice_shape: TensorShape::new(vec![6, 4]),
                        },
                    ],
                },
            ],
        }],
        naming_patterns: vec![
            ExpertNamingPattern {
                family_label: "ffn_up_or_gate".into(),
                pattern: "block_{block}.slot_03.ffn_up_or_gate".into(),
                projection: MoeProjection::Unresolved,
                block_slots: vec![3],
                observed_shapes: vec![TensorShape::new(vec![2, 4, 6])],
                observed_blocks: 1,
            },
            ExpertNamingPattern {
                family_label: "ffn_down".into(),
                pattern: "block_{block}.slot_04.ffn_down".into(),
                projection: MoeProjection::Down,
                block_slots: vec![4],
                observed_shapes: vec![TensorShape::new(vec![2, 6, 4])],
                observed_blocks: 1,
            },
        ],
        naming_checks: vec![
            ExpertNamingCheck {
                check: "expert_count_consistent".into(),
                passed: true,
                detail: "all expert tensors agree on 2 experts".into(),
            },
            ExpertNamingCheck {
                check: "expert_family_count_consistent".into(),
                passed: true,
                detail: "each relevant block exposes two expert tensor families".into(),
            },
        ],
        anomalies: Vec::new(),
        schema_version: 1,
    }
}

pub fn sample_routing_report() -> RoutingReport {
    RoutingReport {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        shard_count: 2,
        inferred: InferredHyperparams {
            d_model: Some(4),
            n_experts: Some(2),
            n_blocks: Some(1),
            ..Default::default()
        },
        relevant_block_count: 1,
        expected_experts_per_router: Some(2),
        candidate_tensors: vec![RoutingTensorRef {
            shard_ordinal: 1,
            in_shard_index: 0,
            block_index: Some(0),
            block_slot: Some(0),
            role: TensorRole::Tensor,
            dtype: TensorDType::F32,
            shape: TensorShape::new(vec![4, 2]),
            kind_label: "router".into(),
            orientation: RoutingOrientation::DModelToExperts,
            expert_axis: Some(1),
            linked_expert_count: Some(2),
            matches_inferred_expert_count: true,
            structural_name: "block_000.routing_slot_00".into(),
            gate_metrics: RoutingGateMetrics {
                total_elements: 8,
                total_nbytes: 32,
                input_width: Some(4),
                output_width: Some(2),
                expert_count: Some(2),
                logits_per_input: Some(2),
            },
        }],
        blocks: vec![RoutingBlockReport {
            block_index: Some(0),
            label: "block_000".into(),
            shard_range: Some(ShardRange {
                start: 1,
                end_inclusive: 1,
            }),
            local_expert_count: Some(2),
            primary_candidate: Some(RoutingTensorLocator {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_slot: Some(0),
            }),
            candidates: vec![RoutingTensorRef {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_index: Some(0),
                block_slot: Some(0),
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4, 2]),
                kind_label: "router".into(),
                orientation: RoutingOrientation::DModelToExperts,
                expert_axis: Some(1),
                linked_expert_count: Some(2),
                matches_inferred_expert_count: true,
                structural_name: "block_000.routing_slot_00".into(),
                gate_metrics: RoutingGateMetrics {
                    total_elements: 8,
                    total_nbytes: 32,
                    input_width: Some(4),
                    output_width: Some(2),
                    expert_count: Some(2),
                    logits_per_input: Some(2),
                },
            }],
        }],
        orientation_summaries: vec![RoutingOrientationSummary {
            orientation: RoutingOrientation::DModelToExperts,
            count: 1,
            observed_shapes: vec![TensorShape::new(vec![4, 2])],
            observed_blocks: 1,
        }],
        likely_routing_critical_blocks: vec![RoutingCriticalBlock {
            block_index: Some(0),
            label: "block_000".into(),
            reason: "contains a primary routing candidate linked to a 2-expert MoE block".into(),
            primary_candidate: Some(RoutingTensorLocator {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_slot: Some(0),
            }),
        }],
        grok_layout_notes: vec![
            "routers use (d_model, n_experts) orientation in this sample".into(),
        ],
        anomalies: Vec::new(),
        schema_version: 1,
    }
}

pub fn sample_stats_profile() -> StatsProfileReport {
    let embedding_stat = TensorStats {
        shard_ordinal: 0,
        in_shard_index: 0,
        block_index: None,
        block_slot: None,
        structural_name: "embedding.slot_00.token_embedding".into(),
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![16, 4]),
        kind_label: "token_embedding".into(),
        sampled: true,
        total_values: 64,
        sample_values: 64,
        total_nbytes: 256,
        mean: 0.125,
        variance: 0.0625,
        stddev: 0.25,
        min: -0.25,
        max: 0.75,
        max_abs: 0.75,
        l1_norm: 12.0,
        l2_norm: 2.5,
        rms: 0.3125,
        zero_fraction: 0.10,
        near_zero_fraction: 0.25,
        positive_fraction: 0.60,
        negative_fraction: 0.30,
        outlier_fraction: 0.0,
        peak_to_rms: 2.4,
        distribution_label: "dense_balanced".into(),
    };
    let router_stat = TensorStats {
        shard_ordinal: 1,
        in_shard_index: 0,
        block_index: Some(0),
        block_slot: Some(0),
        structural_name: "block_000.routing_slot_00".into(),
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![4, 2]),
        kind_label: "router".into(),
        sampled: true,
        total_values: 8,
        sample_values: 8,
        total_nbytes: 32,
        mean: 0.0,
        variance: 1.25,
        stddev: 1.118_033_988_75,
        min: -1.0,
        max: 2.0,
        max_abs: 2.0,
        l1_norm: 7.0,
        l2_norm: 3.162_277_660_17,
        rms: 1.118_033_988_75,
        zero_fraction: 0.0,
        near_zero_fraction: 0.125,
        positive_fraction: 0.5,
        negative_fraction: 0.375,
        outlier_fraction: 0.125,
        peak_to_rms: 1.788_854_381_99,
        distribution_label: "outlier_heavy".into(),
    };
    StatsProfileReport {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        shard_count: 2,
        inferred: InferredHyperparams {
            vocab_size: Some(16),
            d_model: Some(4),
            n_experts: Some(2),
            n_blocks: Some(1),
            ..Default::default()
        },
        sampling: StatsSamplingConfig {
            max_sample_values: 64,
            f32_near_zero_abs: 1e-3,
            i8_near_zero_abs: 1,
        },
        tensors: vec![embedding_stat.clone(), router_stat.clone()],
        layers: vec![
            LayerStats {
                block_index: None,
                label: "embedding".into(),
                tensor_count: 1,
                sampled_tensor_count: 1,
                total_nbytes: 256,
                mean_rms: 0.3125,
                mean_variance: 0.0625,
                mean_outlier_fraction: 0.0,
                routing_tensor_count: 0,
                compressible_candidate_count: 1,
            },
            LayerStats {
                block_index: Some(0),
                label: "block_000".into(),
                tensor_count: 1,
                sampled_tensor_count: 1,
                total_nbytes: 32,
                mean_rms: 1.118_033_988_75,
                mean_variance: 1.25,
                mean_outlier_fraction: 0.125,
                routing_tensor_count: 1,
                compressible_candidate_count: 0,
            },
        ],
        norm_summary: NormSummary {
            mean_rms: 0.715_266_994_375,
            max_rms: Some(rank(&router_stat, "router", 1.118_033_988_75)),
            max_l2: Some(rank(&router_stat, "router", 3.162_277_660_17)),
            top_rms: vec![
                rank(&router_stat, "router", 1.118_033_988_75),
                rank(&embedding_stat, "token_embedding", 0.3125),
            ],
            top_l2: vec![
                rank(&router_stat, "router", 3.162_277_660_17),
                rank(&embedding_stat, "token_embedding", 2.5),
            ],
        },
        variance_summary: VarianceSummary {
            mean_variance: 0.65625,
            max_variance: Some(rank(&router_stat, "router", 1.25)),
            min_variance: Some(rank(&embedding_stat, "token_embedding", 0.0625)),
            top_variance: vec![
                rank(&router_stat, "router", 1.25),
                rank(&embedding_stat, "token_embedding", 0.0625),
            ],
            lowest_variance: vec![
                rank(&embedding_stat, "token_embedding", 0.0625),
                rank(&router_stat, "router", 1.25),
            ],
        },
        outlier_summary: OutlierSummary {
            mean_outlier_fraction: 0.0625,
            most_outlier_heavy: vec![
                rank(&router_stat, "router", 0.125),
                rank(&embedding_stat, "token_embedding", 0.0),
            ],
            highest_peak_to_rms: vec![
                rank(&embedding_stat, "token_embedding", 2.4),
                rank(&router_stat, "router", 1.788_854_381_99),
            ],
        },
        schema_version: 1,
    }
}

pub fn sample_saaq_readiness() -> SaaqReadinessReport {
    let candidate = SaaqCandidate {
        rank: 1,
        shard_ordinal: 1,
        in_shard_index: 1,
        block_index: Some(0),
        block_slot: Some(1),
        structural_name: "block_000.slot_01.attn_proj".into(),
        kind_label: "attn_proj_f32".into(),
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![4, 4]),
        region_class: SaaqRegionClass::PotentialCompressionTarget,
        disposition: SaaqDisposition::Candidate,
        readiness_score: 0.82,
        opportunity_score: 0.78,
        risk_score: 0.24,
        reasons: vec![
            "f32 tensor with moderate size share".into(),
            "not marked routing-critical".into(),
        ],
    };
    let routing_critical = SaaqCandidate {
        rank: 1,
        shard_ordinal: 1,
        in_shard_index: 0,
        block_index: Some(0),
        block_slot: Some(0),
        structural_name: "block_000.routing_slot_00".into(),
        kind_label: "router".into(),
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![4, 2]),
        region_class: SaaqRegionClass::RoutingCritical,
        disposition: SaaqDisposition::AvoidForNow,
        readiness_score: 0.05,
        opportunity_score: 0.08,
        risk_score: 0.91,
        reasons: vec!["routing tensor linked to expert selection".into()],
    };
    let risky = SaaqCandidate {
        rank: 1,
        shard_ordinal: 1,
        in_shard_index: 2,
        block_index: Some(0),
        block_slot: Some(2),
        structural_name: "block_000.slot_02.norm".into(),
        kind_label: "block_norm".into(),
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![4]),
        region_class: SaaqRegionClass::NormalizationSensitive,
        disposition: SaaqDisposition::AvoidForNow,
        readiness_score: 0.10,
        opportunity_score: 0.12,
        risk_score: 0.74,
        reasons: vec!["normalization-sensitive tensor".into()],
    };
    let deferred = SaaqCandidate {
        rank: 1,
        shard_ordinal: 0,
        in_shard_index: 0,
        block_index: None,
        block_slot: None,
        structural_name: "embedding.slot_00.token_embedding".into(),
        kind_label: "token_embedding".into(),
        dtype: TensorDType::F32,
        shape: TensorShape::new(vec![16, 4]),
        region_class: SaaqRegionClass::EmbeddingHeavy,
        disposition: SaaqDisposition::ObserveOnly,
        readiness_score: 0.08,
        opportunity_score: 0.15,
        risk_score: 0.55,
        reasons: vec!["embedding is deferred from first-pass quantization".into()],
    };
    SaaqReadinessReport {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        shard_count: 2,
        inferred: InferredHyperparams {
            d_model: Some(4),
            n_experts: Some(2),
            n_blocks: Some(1),
            ..Default::default()
        },
        candidate_targets: vec![candidate.clone()],
        quantization_candidates: vec![candidate.clone()],
        routing_critical_tensors: vec![routing_critical.clone()],
        precision_sensitive_tensors: vec![risky.clone()],
        deferred_tensors: vec![deferred.clone()],
        risky_tensors: vec![risky],
        layer_readiness: vec![SaaqLayerReadiness {
            block_index: Some(0),
            label: "block_000".into(),
            routing_critical: true,
            candidate_target_count: 1,
            mean_readiness_score: 0.49,
            max_risk_score: 0.91,
        }],
        notes: vec![
            "routing tensors remain avoid-for-now in this structural profile".into(),
            "attention projection is the top quantization candidate in the synthetic sample".into(),
        ],
        manifest: CandidateTensorManifest {
            model_family: "grok-1".into(),
            checkpoint_path: sample_checkpoint_path(),
            candidates: vec![candidate],
            schema_version: 1,
        },
        schema_version: 2,
    }
}

fn rank(stats: &TensorStats, kind_label: &str, value: f64) -> RankedTensorStat {
    RankedTensorStat {
        shard_ordinal: stats.shard_ordinal,
        in_shard_index: stats.in_shard_index,
        structural_name: stats.structural_name.clone(),
        kind_label: kind_label.into(),
        block_index: stats.block_index,
        value,
    }
}

pub fn sample_quant_plan() -> QuantPlan {
    QuantPlan {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        baseline: "grok1-map-v1-clean".into(),
        required_validation: Grok1CoverageCounts {
            blocks: 64,
            tensors: 770,
            routers: 64,
            expert_families: 192,
            unknown_tensors: 0,
        },
        discovered_validation: Grok1CoverageCounts {
            blocks: 64,
            tensors: 770,
            routers: 64,
            expert_families: 192,
            unknown_tensors: 0,
        },
        keep_fp32: vec!["router".into(), "block_norm".into(), "final_norm".into()],
        pilot_quantize: vec![
            "attn_proj_i8.model_width".into(),
            "attn_proj_i8.narrow".into(),
            "moe_expert.gate".into(),
            "moe_expert.up".into(),
            "moe_expert.down".into(),
        ],
        defer: vec!["token_embedding".into()],
        notes: vec![
            "770 tensors partition into 1 quantization candidates, 1 routing-critical, 1 precision-sensitive, and 1 deferred entries.".into(),
            "Top quantization candidate is `block_000.slot_01.attn_proj` with readiness 0.820.".into(),
        ],
        schema_version: 1,
    }
}

pub fn sample_conversion_manifest() -> ConversionManifest {
    ConversionManifest {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        baseline_profile: "grok1-map-v1-clean".into(),
        required_validation: Grok1CoverageCounts {
            blocks: 64,
            tensors: 770,
            routers: 64,
            expert_families: 192,
            unknown_tensors: 0,
        },
        discovered_validation: Grok1CoverageCounts {
            blocks: 64,
            tensors: 770,
            routers: 64,
            expert_families: 192,
            unknown_tensors: 0,
        },
        relevant_block_count: 64,
        expected_experts_per_block: Some(8),
        expert_tensor_families_per_block: Some(3),
        router_orientation: Some(RoutingOrientation::DModelToExperts),
        router_shape: Some(TensorShape::new(vec![6144, 8])),
        tensors: vec![
            ConversionManifestTensor {
                tensor_name: "tensor0000#0".into(),
                structural_name: "embedding.slot_00.token_embedding".into(),
                model_family: "grok-1".into(),
                block: None,
                slot: None,
                kind: "token_embedding".into(),
                region: SaaqRegionClass::EmbeddingHeavy,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![16, 4]),
                numel: 64,
                byte_len: 256,
                shard_index: 0,
                source_shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0000"),
                source_in_shard_index: 0,
                quant_policy: QuantPolicy::CandidateSaaqEmbedding,
                protected_reason: None,
                deterministic_hash: "fnv1a64:1111111111111111".into(),
                warnings: Vec::new(),
            },
            ConversionManifestTensor {
                tensor_name: "tensor0001#0".into(),
                structural_name: "block_000.routing_slot_00".into(),
                model_family: "grok-1".into(),
                block: Some(0),
                slot: Some(0),
                kind: "router".into(),
                region: SaaqRegionClass::RoutingCritical,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4, 2]),
                numel: 8,
                byte_len: 32,
                shard_index: 1,
                source_shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0001"),
                source_in_shard_index: 0,
                quant_policy: QuantPolicy::PassthroughF32Router,
                protected_reason: Some(
                    "protected router tensor; keep f32 to preserve expert selection".into(),
                ),
                deterministic_hash: "fnv1a64:2222222222222222".into(),
                warnings: Vec::new(),
            },
            ConversionManifestTensor {
                tensor_name: "tensor0001#2".into(),
                structural_name: "block_000.slot_02.block_norm".into(),
                model_family: "grok-1".into(),
                block: Some(0),
                slot: Some(2),
                kind: "block_norm".into(),
                region: SaaqRegionClass::NormalizationSensitive,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![4]),
                numel: 4,
                byte_len: 16,
                shard_index: 1,
                source_shard_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0/tensor0001"),
                source_in_shard_index: 2,
                quant_policy: QuantPolicy::PassthroughF32Norm,
                protected_reason: Some(
                    "protected normalization tensor; keep f32 in first-pass planning".into(),
                ),
                deterministic_hash: "fnv1a64:3333333333333333".into(),
                warnings: Vec::new(),
            },
        ],
        warnings: Vec::new(),
        schema_version: 1,
    }
}

pub fn sample_pilot_selection_plan() -> PilotSelectionPlan {
    PilotSelectionPlan {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        baseline: "grok1-map-v1-clean".into(),
        required_validation: sample_quant_plan().required_validation,
        selected_blocks: vec![
            PilotBlockSelection {
                block_index: 0,
                label: "block_000".into(),
                rationale: "early baseline".into(),
            },
            PilotBlockSelection {
                block_index: 8,
                label: "block_008".into(),
                rationale: "near-zero-sensitive router".into(),
            },
            PilotBlockSelection {
                block_index: 28,
                label: "block_028".into(),
                rationale: "near-zero-sensitive router".into(),
            },
            PilotBlockSelection {
                block_index: 60,
                label: "block_060".into(),
                rationale: "high readiness/routing-critical sample".into(),
            },
            PilotBlockSelection {
                block_index: 63,
                label: "block_063".into(),
                rationale: "late-layer / high peak-to-rms router region".into(),
            },
        ],
        modes: vec![
            PilotQuantizationMode::AttentionOnly,
            PilotQuantizationMode::ExpertOnly,
            PilotQuantizationMode::AttentionPlusExpert,
        ],
        protection_rules: vec![
            "router tensors must remain untouched in every first-pass pilot".into(),
            "block_norm and final_norm tensors must remain untouched in every first-pass pilot".into(),
            "pilot artifacts must be emitted per mode and remain comparable across selected blocks"
                .into(),
        ],
        comparison_artifacts: vec![
            "pilot-selection-plan.json".into(),
            "pilot-selection-plan.md".into(),
            "route-preservation-report.json".into(),
            "route-preservation-report.md".into(),
        ],
        notes: vec![
            "This is a planning artifact only; xai-dissect does not mutate checkpoints or execute a quantization runtime.".into(),
            "Use the selected blocks and protected-family rules to drive downstream bounded pilot runs.".into(),
        ],
        schema_version: 1,
    }
}

pub fn sample_route_preservation_report() -> RoutePreservationReport {
    RoutePreservationReport {
        model_family: "grok-1".into(),
        checkpoint_path: sample_checkpoint_path(),
        baseline: "grok1-map-v1-clean".into(),
        required_validation: sample_quant_plan().required_validation,
        summary: vec![
            RouteMetricStatus {
                name: "router_top1_agreement".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 99.0%".into()),
                observed: None,
                detail: "Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference.".into(),
            },
            RouteMetricStatus {
                name: "router_top2_set_agreement".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 99.5%".into()),
                observed: None,
                detail: "Threshold reserved for downstream pilot comparison artifacts; report as first-class sprint evidence when available.".into(),
            },
            RouteMetricStatus {
                name: "expert_load_distribution_delta".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Capture expert-load distribution drift once bounded pilot routing traces are available.".into(),
            },
            RouteMetricStatus {
                name: "expert_load_js_divergence".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report JS/KL-style divergence over expert-load distributions when downstream pilot evidence exists.".into(),
            },
            RouteMetricStatus {
                name: "router_logit_rank_correlation".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report rank correlation for router logits when logits are captured by downstream pilot comparisons.".into(),
            },
            RouteMetricStatus {
                name: "block_output_cosine".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 0.995".into()),
                observed: None,
                detail: "Tracked as a go/no-go threshold once bounded pilot outputs exist.".into(),
            },
            RouteMetricStatus {
                name: "block_output_rmse".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report alongside cosine similarity for bounded pilot comparisons.".into(),
            },
            RouteMetricStatus {
                name: "residual_stream_drift".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Summarize residual-stream drift once downstream pilot artifacts provide comparable block activations.".into(),
            },
            RouteMetricStatus {
                name: "weight_reconstruction_mse".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Generic reconstruction metrics remain secondary to router-behavior preservation for Grok-1 MoE validation.".into(),
            },
            RouteMetricStatus {
                name: "weight_cosine_similarity".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Useful companion metric, but not sufficient by itself to clear a full quantization run.".into(),
            },
            RouteMetricStatus {
                name: "weight_max_absolute_error".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report max absolute reconstruction error when downstream pilot comparisons include raw tensor deltas.".into(),
            },
            RouteMetricStatus {
                name: "per_channel_scale_error_summary".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Summarize per-channel scale/error drift where quantization metadata is available.".into(),
            },
            RouteMetricStatus {
                name: "logit_kl".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for model/logit KL when downstream pilot inference captures logits.".into(),
            },
            RouteMetricStatus {
                name: "perplexity_delta".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for calibration-data perplexity delta once downstream pilot evaluation exists.".into(),
            },
            RouteMetricStatus {
                name: "generation_sanity_summary".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for short generation sanity checks when pilot inference is available.".into(),
            },
        ],
        router_metrics: vec![
            RouteMetricStatus {
                name: "router_top1_agreement".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 99.0%".into()),
                observed: None,
                detail: "Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference.".into(),
            },
            RouteMetricStatus {
                name: "router_top2_set_agreement".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 99.5%".into()),
                observed: None,
                detail: "Threshold reserved for downstream pilot comparison artifacts; report as first-class sprint evidence when available.".into(),
            },
            RouteMetricStatus {
                name: "expert_load_distribution_delta".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Capture expert-load distribution drift once bounded pilot routing traces are available.".into(),
            },
            RouteMetricStatus {
                name: "expert_load_js_divergence".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report JS/KL-style divergence over expert-load distributions when downstream pilot evidence exists.".into(),
            },
            RouteMetricStatus {
                name: "router_logit_rank_correlation".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report rank correlation for router logits when logits are captured by downstream pilot comparisons.".into(),
            },
        ],
        block_metrics: vec![
            RouteMetricStatus {
                name: "block_output_cosine".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: Some(">= 0.995".into()),
                observed: None,
                detail: "Tracked as a go/no-go threshold once bounded pilot outputs exist.".into(),
            },
            RouteMetricStatus {
                name: "block_output_rmse".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report alongside cosine similarity for bounded pilot comparisons.".into(),
            },
            RouteMetricStatus {
                name: "residual_stream_drift".into(),
                scope: "block_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Summarize residual-stream drift once downstream pilot artifacts provide comparable block activations.".into(),
            },
        ],
        weight_metrics: vec![
            RouteMetricStatus {
                name: "weight_reconstruction_mse".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Generic reconstruction metrics remain secondary to router-behavior preservation for Grok-1 MoE validation.".into(),
            },
            RouteMetricStatus {
                name: "weight_cosine_similarity".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Useful companion metric, but not sufficient by itself to clear a full quantization run.".into(),
            },
            RouteMetricStatus {
                name: "weight_max_absolute_error".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report max absolute reconstruction error when downstream pilot comparisons include raw tensor deltas.".into(),
            },
            RouteMetricStatus {
                name: "per_channel_scale_error_summary".into(),
                scope: "weight_reconstruction".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Summarize per-channel scale/error drift where quantization metadata is available.".into(),
            },
        ],
        model_metrics: vec![
            RouteMetricStatus {
                name: "logit_kl".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for model/logit KL when downstream pilot inference captures logits.".into(),
            },
            RouteMetricStatus {
                name: "perplexity_delta".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for calibration-data perplexity delta once downstream pilot evaluation exists.".into(),
            },
            RouteMetricStatus {
                name: "generation_sanity_summary".into(),
                scope: "model_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "Report-only placeholder for short generation sanity checks when pilot inference is available.".into(),
            },
        ],
        notes: vec![
            "This report defines the required route-preservation surface and thresholds for Grok-1 pilot evidence.".into(),
            "Statuses remain unknown until downstream pilot artifacts supply the observed values.".into(),
        ],
        schema_version: 1,
    }
}
