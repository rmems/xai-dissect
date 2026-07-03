# Grok-1 Architecture Reference

This document describes the Grok-1 model architecture and how `xai-dissect`
maps raw checkpoint shards to structural descriptions of it. It does not
describe the codebase — that is covered in `docs/architecture.md`. This
document is about the model.

## Model overview

Grok-1 is a 64-layer decoder-only transformer with a Mixture-of-Experts
(moe) feed-forward block. The architecture was released by xAI as open
weights in early 2025.

| Hyperparameter | Value |
|---------------|-------|
| `d_model` (hidden width) | 6144 |
| `vocab_size` | 131,072 |
| `n_blocks` (layers) | 64 |
| `n_experts` | 8 |
| `top_k` (experts active per token) | 2 |
| `d_ff` (per-expert inner width) | 32,768 |

The model uses RMSNorm (no layer norm), rotary position embeddings, and a
MoE layer in every transformer block. The router is a simple linear layer
mapping `d_model → n_experts` that selects the top-2 experts per token.

## Per-layer composition

Each of the 64 transformer blocks contains exactly **12 shards** on disk.
This is the `K = 12` constant hard-coded in the Grok-1 coverage validator
(`src/inventory/grok1_coverage.rs`). The 12 slots per block are:

| Slot(s) | Count | Tensor type | Shape | dtype | Role |
|--------|------:|------------|-------|-------|------|
| 0 | 1 | MoE expert projection | `(8, 6144, 32768)` | int8 | quantized `quant.weight` |
| 1 | 1 | MoE expert projection | `(8, 32768, 6144)` | int8 | quantized `quant.weight` |
| 2 | 1 | MoE expert projection | `(8, 6144, 32768)` | int8 | quantized `quant.weight` |
| 3, 6 | 2 | Attention projection | `(6144, 1024)` | int8 | quantized `quant.weight` (narrow) |
| 4–5 | 2 | Attention projection | `(6144, 6144)` | int8 | quantized `quant.weight` (model-width) |
| 7–10 | 4 | RMSNorm | `(6144,)` | f32 | bare `tensor` |
| 11 | 1 | Router / gate | `(6144, 8)` | f32 | bare `tensor` |
| (norm singletons) | 1 | Final pre-head norm | `(6144,)` | f32 | bare `tensor` |
| (embedding) | 1 | Token embedding | `(131072, 6144)` | f32 | bare `tensor` |

The top-level shard accounting reconciles exactly:

```
  1 embedding shard
+ 64 layers × 12 shards/layer  = 768
+ 1 final pre-head norm shard
= 770 total shards
```

## How MoE routing works

The router is a single `(d_model=6144, n_experts=8)` f32 weight matrix.
For each token, it computes an 8-element logit vector, takes `softmax`,
and selects the top-2 expert indices. The selected expert FFNs process the
token in parallel and their outputs are summed with a weighted combination.

This means:

- **The router weights are routing-critical**: quantization of the router
  directly changes which experts process which tokens. A small rounding
  error in the router can cascade into a completely different expert
  selection pattern. This is why `routing-critical-tensors.json` contains
  all 64 router tensors and why the GO/NO-GO gate requires router tensors
  to remain untouched in the first pilot pass.
- **The 8 expert weight matrices are compression targets**: the up/gate
  projections are the primary compression opportunity (they are large,
  `(8, 6144, 32768)`, and the activation distribution is favorable).
- **Router top-1 / top-2 agreement** (GO/NO-GO thresholds of 99.0% /
  99.5%) measures whether quantized router weights select the same experts
  as the FP32 baseline on a calibration dataset. These are runtime metrics
  computed by `grok-ozempic`, not by `xai-dissect`.

## The QuantizedWeight8bit structure

Grok-1 does not store MoE expert or attention weight tensors as bare
FP32. Routers, norms, and the token embedding remain bare FP32. The MoE
expert projections and attention projections are stored as `QuantizedWeight8bit`
dataclass instances — a JAX-specific packing of an int8 weight body plus
per-element f32 quantization scales.

When this dataclass is saved to a pickle shard, it appears as three
separate `numpy.ndarray` reduce sites on disk:

1. **The int8 weight body** — the ndarray with `dtype = i8` in each
   `QuantizedWeight8bit` reduce site. The parser records this as
   `role = quant.weight`. Shape is `(E, d_model, d_ff)` for gate/up and
   `(E, d_ff, d_model)` for down, where `E = n_experts = 8`.
2. **The f32 scales / zero-points** — the first f32 ndarray after the
   int8 weight body in the same reduce site. The parser records this as
   `role = quant.scales`. Shape varies (the two 1.5 GiB shard variants
   differ by exactly 262,144 bytes — 65,536 f32 values — due to different
   quantization side-data shapes across the three expert projection
   families).
3. **Additional f32 ndarrays** (for example per-expert minimum values in
   the down projection's asymmetric quantization layout) remain bare
   `role = tensor` records in the same shard.

`xai-dissect` **does not decode the internal structure of the
QuantizedWeight8bit dataclass**. It emits one `quant.weight` and at most
one `quant.scales` record per reduce site, using dtype-aware role
assignment. The pairing is preserved by the shard grouping (all records
share the same shard file), but the exact internal layout is opaque to
the parser.

## How xai-dissect maps shards to architecture

The 770 shards fall into distinct size buckets. File size alone (verified
from the actual Grok-1 ckpt-0 run) is sufficient to identify the tensor
type before reading any shard body:

| File size | Interpretation | Count |
|----------:|----------------|------:| 
| 3,221,225,637 B | Token embedding `(131072, 6144) f32` | 1 |
| 1,611,137,347 B | MoE `QuantizedWeight8bit` shard, variant A | 128 |
| 1,611,399,491 B | MoE `QuantizedWeight8bit` shard, variant B (262,144 B larger in quantization side-data) | 64 |
| 37,847,359 B | Attention projection int8, model-width bucket | 64 |
| 37,761,334 B | Sibling model-width attention projection | 64 |
| 6,293,814 B | Attention projection int8, narrow bucket | 128 |
| 196,770 B | Router `(6144, 8) f32` | 64 |
| 24,727 B | RMSNorm vector `(6144,) f32` | 257 |

The two model-width attention buckets (37,847,359 and 37,761,334) both
have shape `(6144, 6144)`. Both narrow attention projections share the
single narrow bucket size (6,293,814 B) and shape `(6144, 1024)`. Q/K/V/O
disambiguation is not
attempted by `xai-dissect` — those roles are downstream disambiguation
problems for `grok-ozempic`.

## Source-backed slot disambiguation

The MoE expert projection slots (0, 1, 2) are disambiguated using the
official Grok-1 Haiku model source code:

```text
slot 0 → moe/linear/w   → gate   (GELU gate branch)
slot 1 → moe/linear_1/w → down   (consumes gated product as down projection)
slot 2 → moe/linear_v/w → up     (ungated up/value branch)
```

This ordering is from `xai-org/grok-1/model.py` and the flattened pytree
checkpoint restore path in `xai-org/grok-1/checkpoint.py`.

Because the three projection families share the same `(E=8, d_model=6144,
d_ff=32768)` tensor shape in their outer dimensions, shape alone cannot
distinguish gate from up. Only source-backed block-slot assignment (using
the official checkpoint flattening order) resolves this. If Grok-1 had
used a different ordering in its checkpoint, both would be reported as
`MoeExpertProjection { projection: unresolved }`.

## Inventory output — real Grok-1 ckpt-0 excerpt

The following is the first block table excerpt from the actual
`out/grok1_run2_after_fixes_20260525T002904Z/reports/grok-1-official__ckpt-0/inventory.md`
run against the real Grok-1 ckpt-0 checkpoint:

```markdown
# xai-dissect inventory

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/Downloads/SNN_Quantization/grok-1-official/ckpt-0`
- **shards**: 770
- **schema_version**: 2

## Inferred hyperparameters

| Field | Value |
| ----- | ----- |
| vocab_size | 131072 |
| d_model | 6144 |
| n_experts | 8 |
| d_ff | 32768 |
| n_blocks | 64 |

## Tensor kinds

| Kind | Count | Bytes |
| ---- | ----: | ----: |
| attn_proj_i8.model_width | 128 | 4831838208 (4.50 GiB) |
| attn_proj_i8.narrow | 128 | 805306368 (768.00 MiB) |
| block_norm | 256 | 6291456 (6.00 MiB) |
| final_norm | 1 | 24576 (24.00 KiB) |
| moe_expert.down | 64 | 103079215104 (96.00 GiB) |
| moe_expert.gate | 64 | 103079215104 (96.00 GiB) |
| moe_expert.up | 64 | 103079215104 (96.00 GiB) |
| router | 64 | 12582912 (12.00 MiB) |
| token_embedding | 1 | 3221225472 (3.00 GiB) |

## Blocks

| Label | Block | Shards | Tensors | Bytes | Kinds |
| ----- | ----: | ------ | ------: | ----: | ----- |
| embedding | - | 0..=0 | 1 | 3221225472 (3.00 GiB) | 1xtoken_embedding |
| block_000 | 0 | 2..=13 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_001 | 1 | 14..=25 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_002 | 2 | 26..=37 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
```

The complete 770-tensor inventory is in `exports/<slug>/inventory.json`. The
coverage manifest validating this as a complete clean Grok-1 run is at
`manifests/<slug>/grok1-coverage.json` (see `docs/grok1-coverage-manifest.md`).

## What is not determined by xai-dissect

The following Grok-1 properties cannot be inferred from shard structure
alone and are left as downstream responsibilities:

- **Attention head structure**: Q/K/V head counts and KV-head grouping.
  The `(6144, 6144)` and `(6144, 1024)` attention shards are reported as
  `QuantizedAttentionProjection { width: model_width | narrow }` without
  Q/K/V/O disambiguation.
- **Top-k parameter**: the routing top-k of 2 is a training hyperparameter,
  not a structural property. The router shape `(6144, 8)` is consistent
  with any top-k ≤ 8.
- **Rotary position embedding details**: RoPE frequency bases and
  sequence length configuration are not stored in checkpoint shards.
- **Activation functions**: GELU is referenced in the model source but
  not stored in the weights.

## Grok-1 vs. Grok-2

Grok-2 is not yet supported by `xai-dissect`. Expected extension points
when public Grok-2 weights are released are documented in
`docs/grok2-future-support.md`. The primary architectural differences to
anticipate are likely: different expert count, different d_model, and
potentially a different MoE layout.