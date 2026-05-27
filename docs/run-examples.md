# Annotated CLI Walkthrough

This document walks through every `xai-dissect` subcommand with annotated
examples drawn from the real Grok-1 ckpt-0 run at
`out/grok1_run2_after_fixes_20260525T002904Z/`. For each command the
corresponding output artifact is shown with field-level annotations
explaining what each value means and where it comes from.

The checkpoint path used throughout is the real Grok-1 ckpt-0 location on
this system:

```
/home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0
```

The unified output root used in these examples is `out/`, producing the
slug `grok-1-official__ckpt-0`.

---

## `dissect` — Raw Parser Output

`dissect` opens a single shard and prints the raw pickle-grammar tensor
table without any classification or block grouping.

```bash
./target/release/xai-dissect dissect \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --limit 1
```

Output for `tensor00000_000` (the embedding shard):

```text
tensor00000_000:
  offset=151  nbytes=3221225472  dtype=f32  shape=(131072, 6144)
```

| Field | Value | Meaning |
|-------|-------|---------|
| `offset=151` | Byte offset within the shard file where the numpy ndarray payload begins |
| `nbytes=3221225472` | Payload size in bytes (131072 × 6144 × 4 = 3,221,225,472) |
| `dtype=f32` | IEEE 754 binary32 — bare float tensor, no quantization |
| `shape=(131072, 6144)` | Row-major 2-D shape: vocab rows × d_model columns |

`dissect` is the only command that does **not** support `--output-root`
or `--checkpoint-slug`. It is raw parser output only.

---

## `inventory` — Checkpoint Cartography

The `inventory` command is the primary artifact. Its JSON output
(`exports/<slug>/inventory.json`) is the canonical tensor catalog for the
grok-ozempic handoff contract.

```bash
./target/release/xai-dissect inventory \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --output-root out
```

### Annotated `inventory.json` excerpt — first 4 tensor records

```json
{
  "model_family": "grok-1",
  "checkpoint_path": "/home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0",
  "shard_count": 770,
  "inferred": {
    "vocab_size": 131072,
    "d_model": 6144,
    "n_experts": 8,
    "d_ff": 32768,
    "n_blocks": 64
  },
  "tensors": [
    {
      // Absolute path on disk — informative only; downstream consumers
      // use shard_ordinal + in_shard_index as the stable tensor identity
      "shard_path": "/home/raulmc/Downloads/.../ckpt-0/tensor00000_000",
      // 0-based ordinal in the sorted shard list — stable across reruns
      "shard_ordinal": 0,
      // 0-based index within this shard (shards can hold >1 tensor for
      // QuantizedWeight8bit dataclasses)
      "in_shard_index": 0,
      // Parser-level role: bare ndarray vs int8 weight vs f32 scales
      "role": "tensor",
      "dtype": "f32",
      "shape": [131072, 6144],
      // Byte offset of payload within the shard file
      "offset": 151,
      // Payload length in bytes
      "nbytes": 3221225472,
      // Semantic classification — the tagged-union TensorKind
      "kind": {
        "kind": "token_embedding"
      },
      // block_index is null for embedding and final norm singletons
      "block_index": null,
      "block_slot": null
    },
    {
      "shard_path": "/home/raulmc/Downloads/.../ckpt-0/tensor00001_000",
      "shard_ordinal": 1,
      "in_shard_index": 0,
      "role": "tensor",
      "dtype": "f32",
      "shape": [6144],
      "offset": 146,
      "nbytes": 24576,
      // Final pre-head norm — promoted from block_norm by block assignment
      "kind": { "kind": "final_norm" },
      "block_index": null,
      "block_slot": null
    },
    {
      "shard_path": "/home/raulmc/Downloads/.../ckpt-0/tensor00002_000",
      "shard_ordinal": 2,
      "in_shard_index": 0,
      "role": "quant_weight",
      "dtype": "i8",
      // 3-D int8 body for 8-expert × d_model × d_ff MoE projection
      "shape": [8, 6144, 32768],
      "offset": 201,
      "nbytes": 1610612736,
      // gate/down/up disambiguated from official Grok-1 source ordering
      "kind": {
        "kind": "moe_expert_projection",
        "detail": { "projection": "gate" }
      },
      // block_index = 0 for block_000; block_slot = 0 for slot 0
      "block_index": 0,
      "block_slot": 0
    },
    {
      "shard_path": "/home/raulmc/Downloads/.../ckpt-0/tensor00003_000",
      "shard_ordinal": 3,
      "in_shard_index": 0,
      "role": "quant_weight",
      "dtype": "i8",
      // The down projection has transposed inner dims: (E, d_ff, d_model)
      "shape": [8, 32768, 6144],
      "offset": 201,
      "nbytes": 1610612736,
      "kind": {
        "kind": "moe_expert_projection",
        "detail": { "projection": "down" }
      },
      "block_index": 0,
      "block_slot": 1
    }
    // ... continues for all 770 tensors
  ]
  // schema_version: 2 — bump on any incompatible JSON shape change
}
```

### `checkpoint-inventory-snapshot.json` excerpt

This compact manifest is a dashboard-friendly summary. The embedding
block plus the first two transformer blocks:

```json
{
  "model_family": "grok-1",
  "shard_count": 770,
  "inferred": {
    "vocab_size": 131072,
    "d_model": 6144,
    "n_experts": 8,
    "d_ff": 32768,
    "n_blocks": 64
  },
  "total_tensors": 770,
  "total_nbytes": 318114914304,
  "blocks": [
    {
      "label": "embedding",
      "block_index": null,
      "shard_range": { "start": 0, "end_inclusive": 0 },
      "tensor_count": 1,
      "total_nbytes": 3221225472,
      "kind_labels": ["token_embedding"]
    },
    {
      "label": "block_000",
      "block_index": 0,
      "shard_range": { "start": 2, "end_inclusive": 13 },
      "tensor_count": 12,
      "total_nbytes": 4920213504,
      // Every kind appearing in this block
      "kind_labels": [
        "attn_proj_i8.model_width",
        "attn_proj_i8.narrow",
        "block_norm",
        "moe_expert.down",
        "moe_expert.gate",
        "moe_expert.up",
        "router"
      ]
    },
    {
      "label": "block_001",
      "block_index": 1,
      "shard_range": { "start": 14, "end_inclusive": 25 },
      "tensor_count": 12,
      "total_nbytes": 4920213504,
      "kind_labels": [
        "attn_proj_i8.model_width",
        "attn_proj_i8.narrow",
        "block_norm",
        "moe_expert.down",
        "moe_expert.gate",
        "moe_expert.up",
        "router"
      ]
    }
    // blocks 2..63 follow the same pattern
  ]
}
```

---

## `experts` — Expert Atlas

The expert atlas resolves MoE projection identities per block and maps
each of the 8 expert indices to the physical tensors that carry their
parameters.

```bash
./target/release/xai-dissect experts \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --output-root out
```

### Annotated `experts.json` excerpt — block_000

```json
{
  "model_family": "grok-1",
  "relevant_block_count": 64,
  "expected_experts_per_block": 8,
  "blocks": [
    {
      "block_index": 0,
      "shard_range": { "start": 2, "end_inclusive": 13 },
      "expert_count": 8,
      "tensors": [
        {
          "shard_ordinal": 2,
          "in_shard_index": 0,
          "block_slot": 0,
          "role": "quant_weight",
          "dtype": "i8",
          "shape": [8, 6144, 32768],
          "kind_label": "moe_expert.gate",
          "projection": "gate",
          // expert_axis = 0 means the expert dimension is the leading dim
          "expert_axis": 0,
          "expert_count": 8,
          "family_label": "expert_slot_00",
          // Structural name used in routing-critical-tensors.json
          "structural_name": "block_000.expert_slot_00"
        },
        {
          "shard_ordinal": 3,
          "in_shard_index": 0,
          "block_slot": 1,
          "role": "quant_weight",
          "dtype": "i8",
          "shape": [8, 32768, 6144],
          "kind_label": "moe_expert.down",
          "projection": "down",
          "expert_axis": 0,
          "expert_count": 8,
          "family_label": "expert_slot_01",
          "structural_name": "block_000.expert_slot_01"
        },
        {
          "shard_ordinal": 4,
          "in_shard_index": 0,
          "block_slot": 2,
          "role": "quant_weight",
          "dtype": "i8",
          "shape": [8, 6144, 32768],
          "kind_label": "moe_expert.up",
          "projection": "up",
          "expert_axis": 0,
          "expert_count": 8,
          "family_label": "expert_slot_02",
          "structural_name": "block_000.expert_slot_02"
        }
      ],
      "experts": [
        {
          "expert_index": 0,
          "tensors": [
            {
              "family_label": "expert_slot_00",
              "structural_name": "block_000.expert_slot_00.expert_00",
              "source_shard_ordinal": 2,
              "source_in_shard_index": 0,
              "source_block_slot": 0,
              "projection": "gate",
              "dtype": "i8",
              // Slice shape is the per-expert slice: (d_model, d_ff)
              "slice_shape": [6144, 32768]
            },
            {
              "family_label": "expert_slot_01",
              "structural_name": "block_000.expert_slot_01.expert_00",
              "source_shard_ordinal": 3,
              "source_in_shard_index": 0,
              "source_block_slot": 1,
              "projection": "down",
              "dtype": "i8",
              // Slice shape is (d_ff, d_model) for the down projection
              "slice_shape": [32768, 6144]
            }
            // expert_index 1..7 follow identically
          ]
        }
        // expert_index 1..7
      ]
    }
    // blocks 1..63 follow the same structure
  ]
}
```

---

## `routing-report` — Routing Structure Inspection

```bash
./target/release/xai-dissect routing-report \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --output-root out
```

### Annotated `routing-report.json` excerpt — block_000 router

```json
{
  "relevant_block_count": 64,
  "expected_experts_per_router": 8,
  "candidate_tensors": [
    {
      "shard_ordinal": 13,
      "in_shard_index": 0,
      "block_index": 0,
      "block_slot": 11,
      "role": "tensor",
      "dtype": "f32",
      "shape": [6144, 8],
      "kind_label": "router",
      // d_model (6144) maps to expert logits (8) — DModelToExperts orientation
      "orientation": "d_model_to_experts",
      // The leading dim is d_model, the trailing dim is n_experts
      "expert_axis": 1,
      "linked_expert_count": 8,
      "matches_inferred_expert_count": true,
      "structural_name": "block_000.routing_slot_11",
      "gate_metrics": {
        "total_elements": 49152,
        "total_nbytes": 196608,
        "input_width": 6144,
        "output_width": 8,
        "expert_count": 8,
        "logits_per_input": 8
      }
    }
    // 63 more candidates (one per block) follow
  ]
}
```

### `routing-critical-tensors.json` — the guardrail manifest

This is the machine-ingest artifact for grok-ozempic. All 64 router
tensors are listed here:

```json
{
  "model_family": "grok-1",
  "tensors": [
    {
      "shard_ordinal": 13,
      "in_shard_index": 0,
      "block_index": 0,
      "block_slot": 11,
      "structural_name": "block_000.routing_slot_11",
      "kind_label": "router",
      "orientation": "d_model_to_experts",
      "linked_expert_count": 8,
      "total_nbytes": 196608,
      // Why this tensor is critical — used for human review, not machine ingest
      "criticality_reason": "contains a primary routing candidate linked to a 8-expert MoE block"
    },
    {
      "shard_ordinal": 25,
      "in_shard_index": 0,
      "block_index": 1,
      "block_slot": 11,
      "structural_name": "block_001.routing_slot_11",
      "kind_label": "router",
      "orientation": "d_model_to_experts",
      "linked_expert_count": 8,
      "total_nbytes": 196608,
      "criticality_reason": "contains a primary routing candidate linked to a 8-expert MoE block"
    }
    // All 64 router tensors follow — blocks 0..63, each at slot 11
  ]
}
```

---

## `stats` — Offline Tensor Statistics

```bash
./target/release/xai-dissect stats \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --sample-values 65536 \
    --output-root out
```

### Annotated `stats.json` excerpt — token embedding tensor

```json
{
  "sampling": {
    "max_sample_values": 65536,
    "f32_near_zero_abs": 0.001,
    "i8_near_zero_abs": 1
  },
  "tensors": [
    {
      "shard_ordinal": 0,
      "in_shard_index": 0,
      "block_index": null,
      "block_slot": null,
      "structural_name": "embedding.slot_00.token_embedding",
      "role": "tensor",
      "dtype": "f32",
      "shape": [131072, 6144],
      "kind_label": "token_embedding",
      "sampled": true,
      // How many values were read vs. total in the tensor
      "total_values": 805306368,
      "sample_values": 65536,
      "total_nbytes": 3221225472,
      // Statistical moments over the sampled values
      "mean": 0.005448298090801567,
      "variance": 0.00012176985980541508,
      "stddev": 0.011034938142346566,
      "min": -0.039775047451257706,
      "max": 0.05811597779393196,
      "max_abs": 0.05811597779393196,
      "l1_norm": 634.2734907355923,
      "l2_norm": 3.1505042479151633,
      "rms": 0.012306657218418606,
      // Fractions of the sampled distribution
      "zero_fraction": 0.0,
      "near_zero_fraction": 0.067474365234375,
      "positive_fraction": 0.6834564208984375,
      "negative_fraction": 0.3165435791015625,
      "outlier_fraction": 0.0,
      // peak_to_rms > 4.0 suggests heavy-tailed distribution
      "peak_to_rms": 4.722320347637001,
      "distribution_label": "dense_balanced"
    }
    // ... continues for all 770 tensors
  ]
}
```

---

## `saaq-readiness` — SAAQ Candidate Scouting

```bash
./target/release/xai-dissect saaq-readiness \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --sample-values 65536 \
    --output-root out
```

### Annotated `candidate-saaq-targets.json` excerpt

```json
{
  "model_family": "grok-1",
  "candidates": [
    {
      "rank": 1,
      "shard_ordinal": 0,
      "in_shard_index": 0,
      "block_index": null,
      "block_slot": null,
      "structural_name": "embedding.slot_00.token_embedding",
      "kind_label": "token_embedding",
      "dtype": "f32",
      "shape": [131072, 6144],
      // The embedding shard is the only embedding-heavy candidate
      "region_class": "embedding_heavy",
      "disposition": "candidate",
      // Composite scoring — see docs/metric-definitions.md for methodology
      "readiness_score": 0.17649918328848646,
      "opportunity_score": 0.33145386533049104,
      "risk_score": 0.39055624243345904,
      "reasons": [
        "distribution=dense_balanced",
        "sampled_values=65536/805306368",
        "zero_fraction=0.0000",
        "near_zero_fraction=0.0675",
        "outlier_fraction=0.0000",
        "peak_to_rms=4.722"
      ]
    }
  ],
  "schema_version": 1
}
```

---

## `pilot-plan` — Pilot Selection Plan

The `pilot-plan` planning artifact is from the test fixture snapshot
(`tests/fixtures/exports/pilot-plan.snap`) since the real Grok-1 run
did not generate a pilot plan.

```bash
./target/release/xai-dissect pilot-plan \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --output-root out
```

### Annotated `pilot-selection-plan.json` from fixture

```json
{
  "model_family": "grok-1",
  "baseline": "grok1-map-v1-clean",
  "required_validation": {
    "blocks": 64,
    "tensors": 770,
    "routers": 64,
    "expert_families": 192,
    "unknown_tensors": 0
  },
  "selected_blocks": [
    { "block_index": 0,  "label": "block_000", "rationale": "early baseline" },
    { "block_index": 8,  "label": "block_008", "rationale": "near-zero-sensitive router" },
    { "block_index": 28, "label": "block_028", "rationale": "near-zero-sensitive router" },
    { "block_index": 60, "label": "block_060", "rationale": "high readiness/routing-critical sample" },
    { "block_index": 63, "label": "block_063", "rationale": "late-layer / high peak-to-rms router region" }
  ],
  "modes": [
    "attention_only",
    "expert_only",
    "attention_plus_expert"
  ],
  "protection_rules": [
    "router tensors must remain untouched in every first-pass pilot",
    "block_norm and final_norm tensors must remain untouched in every first-pass pilot",
    "pilot artifacts must be emitted per mode and remain comparable across selected blocks"
  ],
  "comparison_artifacts": [
    "pilot-selection-plan.json",
    "pilot-selection-plan.md",
    "route-preservation-report.json",
    "route-preservation-report.md"
  ],
  "notes": [
    "This is a planning artifact only; xai-dissect does not mutate checkpoints or execute a quantization runtime.",
    "Use the selected blocks and protected-family rules to drive downstream bounded pilot runs."
  ],
  "schema_version": 1
}
```

---

## `route-preservation` — Route Preservation Gate Report

The `route-preservation` artifact is from the test fixture snapshot
(`tests/fixtures/exports/route-preservation.snap`).

```bash
./target/release/xai-dissect route-preservation \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --output-root out
```

### Annotated `route-preservation-report.json` — key metrics

```json
{
  "model_family": "grok-1",
  "baseline": "grok1-map-v1-clean",
  "summary": [
    {
      "name": "router_top1_agreement",
      "scope": "router_behavior",
      "status": "unknown",
      "threshold": ">= 99.0%",
      "observed": null,
      // xai-dissect defines the gate but cannot execute pilot inference
      "detail": "Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference."
    },
    {
      "name": "block_output_cosine",
      "scope": "block_behavior",
      "status": "unknown",
      "threshold": ">= 0.995",
      "observed": null,
      "detail": "Tracked as a go/no-go threshold once bounded pilot outputs exist."
    }
    // ... 15 total metrics across router, block, weight, and model scopes
  ],
  "router_metrics": [
    {
      "name": "router_top1_agreement",
      "scope": "router_behavior",
      "status": "unknown",
      "threshold": ">= 99.0%",
      "observed": null,
      "detail": "Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference."
    }
    // ... 5 router metrics total
  ],
  "block_metrics": [
    {
      "name": "block_output_cosine",
      "scope": "block_behavior",
      "status": "unknown",
      "threshold": ">= 0.995",
      "observed": null,
      "detail": "Tracked as a go/no-go threshold once bounded pilot outputs exist."
    }
    // ... 3 block metrics total
  ],
  "notes": [
    "This report defines the required route-preservation surface and thresholds for Grok-1 pilot evidence.",
    "Statuses remain unknown until downstream pilot artifacts supply the observed values."
  ],
  "schema_version": 1
}
```

All `status: "unknown"` entries will be populated with real observed values
when `grok-ozempic` executes the bounded pilot runs and writes the
comparison artifacts.

---

## `quant-plan` — Conversion and Quant Planning

The `quant-plan` artifacts are from the test fixture snapshot.

```bash
./target/release/xai-dissect quant-plan \
    /home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0 \
    --sample-values 65536 \
    --output-root out
```

### Annotated `conversion-manifest.json` excerpt

```json
{
  "baseline_profile": "grok1-map-v1-clean",
  "relevant_block_count": 64,
  "expected_experts_per_block": 8,
  "expert_tensor_families_per_block": 3,
  "router_orientation": "d_model_to_experts",
  "router_shape": [6144, 8],
  "tensors": [
    {
      "tensor_name": "tensor0000#0",
      "structural_name": "embedding.slot_00.token_embedding",
      "model_family": "grok-1",
      "block": null,
      "slot": null,
      "kind": "token_embedding",
      // Region class drives disposition in the SAAQ readiness pipeline
      "region": "embedding_heavy",
      "dtype": "f32",
      "shape": [16, 4],
      "numel": 64,
      "byte_len": 256,
      "shard_index": 0,
      "source_shard_path": "/fixtures/grok-1-official/ckpt-0/tensor0000",
      "source_in_shard_index": 0,
      // QuantPolicy determines what downstream packing/repacking does with this tensor
      "quant_policy": "candidate_saaq_embedding",
      "protected_reason": null,
      // Deterministic hash over canonical tensor representation
      "deterministic_hash": "fnv1a64:1111111111111111",
      "warnings": []
    },
    {
      "tensor_name": "tensor0001#0",
      "structural_name": "block_000.routing_slot_00",
      "block": 0,
      "slot": 0,
      "kind": "router",
      "region": "routing_critical",
      "dtype": "f32",
      "shape": [4, 2],
      "numel": 8,
      "byte_len": 32,
      "shard_index": 1,
      "quant_policy": "passthrough_f32_router",
      // Router must stay FP32 to preserve expert selection
      "protected_reason": "protected router tensor; keep f32 to preserve expert selection",
      "deterministic_hash": "fnv1a64:2222222222222222",
      "warnings": []
    }
  ]
}
```

### `quant-plan.json` — policy summary

```json
{
  "model_family": "grok-1",
  "baseline": "grok1-map-v1-clean",
  "required_validation": {
    "blocks": 64, "tensors": 770, "routers": 64,
    "expert_families": 192, "unknown_tensors": 0
  },
  // FP32 tensors that must not be quantized in any first-pass pilot
  "keep_fp32": ["router", "block_norm", "final_norm"],
  // Tensor families recommended for pilot quantization
  "pilot_quantize": [
    "attn_proj_i8.model_width",
    "attn_proj_i8.narrow",
    "moe_expert.gate",
    "moe_expert.up",
    "moe_expert.down"
  ],
  // Tensors deferred beyond the first pilot pass
  "defer": ["token_embedding"],
  "notes": [
    "770 tensors partition into 1 quantization candidates, 1 routing-critical, 1 precision-sensitive, and 1 deferred entries.",
    "Top quantization candidate is `block_000.slot_01.attn_proj` with readiness 0.820."
  ],
  "schema_version": 1
}
```

---

## `grok1-coverage.json` — the validation gate

The coverage manifest is the fail-closed gate for the grok-ozempic handoff.
This is the complete 22-line file from the real Grok-1 run:

```json
{
  "model_family": "grok-1",
  "schema_version": 2,
  "coverage_schema_version": 1,
  "validation": "pass",
  // FNV-1a 64-bit checksum over canonical structural tensor representation
  "checksum": "fnv1a64:de5a1c978121c62c",
  "expected": {
    "blocks": 64,
    "tensors": 770,
    "routers": 64,
    "expert_families": 192,
    "unknown_tensors": 0
  },
  "discovered": {
    "blocks": 64,
    "tensors": 770,
    "routers": 64,
    "expert_families": 192,
    "unknown_tensors": 0
  },
  "unknown_slots": []
}
```

For a grok-ozempic bundle to be accepted: `validation` must be `"pass"`,
`expected` must equal `discovered`, and `unknown_slots` must be empty.
See `docs/grok1-coverage-manifest.md` for the full algorithm documentation.