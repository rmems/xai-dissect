# Tensor schema and naming heuristics

This document defines the exported inventory data model of `xai-dissect` and
the rules the inventory layer uses to classify Grok-1 tensors by shape. The
JSON export is produced by `report::write_json` and matches the
`ModelInventory` struct in `src/schema/mod.rs` byte-for-byte via serde.

Schema version: **1** (`ModelInventory.schema_version`).

## Core types

### `TensorDType`

Narrow, additive enum.

- `f32` - IEEE 754 binary32.
- `i8` - signed 8-bit integer.

No other dtypes are emitted. A checkpoint that contains an unsupported
dtype is a parser error, not a schema extension.

### `TensorShape`

A row-major shape tuple serialized as a JSON array of non-negative
integers. A rank-0 tensor serializes as `[]`.

### `TensorRole`

Parser-level tag describing how a tensor appeared on disk. Orthogonal to
semantic classification.

- `tensor` - a bare `numpy.ndarray`.
- `quant.weight` - the int8 half of a `QuantizedWeight8bit` dataclass.
- `quant.scales` - the f32 scales half of a `QuantizedWeight8bit` dataclass.

### `TensorKind`

Semantic classification. Inferred from `(rank, dtype, dims, role)` plus a
small set of hyperparameters inferred from the inventory itself
(`vocab_size`, `d_model`, `n_experts`). Emitted as a tagged JSON object
`{ "kind": "...", "detail": ... }`.

Variants:

- `token_embedding`
- `final_norm`
- `block_norm`
- `router`
- `moe_expert_projection` with `detail = { "projection": "up" | "gate" | "down" | "unresolved" }`
- `moe_scales`
- `attn_proj_f32`
- `unknown` with `detail = { "reason": "..." }`

### `TensorInfo`

One record per tensor found on disk. One shard may produce one or two
records (two for `QuantizedWeight8bit`).

Fields:

- `shard_path` - absolute path of the shard file.
- `shard_ordinal` - 0-based index of the shard in the sorted shard list.
- `in_shard_index` - 0-based index of the tensor inside the shard.
- `role` - `TensorRole`.
- `dtype` - `TensorDType`.
- `shape` - `TensorShape`.
- `offset` - byte offset of the raw payload within the shard file.
- `nbytes` - payload length in bytes.
- `kind` - `TensorKind`.
- `block_index` - optional transformer-block index.
- `block_slot` - optional position within the block.

### `BlockSummary`

Aggregate view of one transformer block or one non-block singleton (the
embedding and the final norm get their own rows). Contains: label,
`block_index`, `shard_range`, tensor/byte counts, the set of dtypes seen,
and per-kind counts.

### `ModelInventory`

Top-level document. Carries `model_family`, `checkpoint_path`,
`shard_count`, `inferred` hyperparameters, the full `tensors` array, the
`blocks` summary list, `totals`, and `schema_version`.

## Hyperparameter inference

Two-pass, conservative, no hardcoded Grok-1 constants:

1. **Embedding**: the largest 2-D f32 `tensor` (not quant) wins. Its
   shape fixes `vocab_size = dims[0]` and `d_model = dims[1]`.
2. **Experts**: the first 3-D int8 `quant.weight` tensor encountered fixes
   `n_experts = dims[0]`. If one of the inner dims equals `d_model`, the
   other is recorded as `d_ff`.
3. **Blocks**: after tensors are assembled, the shard layout is checked
   against the Grok-1 pattern below to derive `n_blocks`.

If a step cannot be resolved, the corresponding field is `null` and
classification falls back to `Unknown { reason }` for the affected
tensors.

## Classification rules (Grok-1)

Applied in order; the first matching rule wins.

1. `(role = quant.weight, dtype = i8, rank = 3)`:
   - If `dims[0] == n_experts`:
     - If `dims[1] == d_model` -> `MoeExpertProjection { projection: unresolved }`
       (the `(E, d_model, d_ff)` signature covers both up and gate; they
       cannot be told apart by shape alone on Grok-1).
     - Else if `dims[2] == d_model` -> `MoeExpertProjection { projection: down }`
       (the `(E, d_ff, d_model)` signature).
     - Else -> `MoeExpertProjection { projection: unresolved }`.
   - Else -> `Unknown`.
2. `(role = quant.scales, dtype = f32)` -> `MoeScales`.
3. `(role = tensor, dtype = f32, rank = 2)`:
   - If `dims == (vocab_size, d_model)` -> `TokenEmbedding`.
   - Else if `dims == (d_model, n_experts)` -> `Router`.
   - Else -> `AttnProjF32`.
4. `(role = tensor, dtype = f32, rank = 1)`:
   - If `dims[0] == d_model` -> `BlockNorm` (may be promoted to
     `FinalNorm` by the block-assignment pass).
   - Else -> `Unknown`.
5. `(role = tensor, dtype = f32, rank >= 3)` -> `AttnProjF32`.
6. Anything else -> `Unknown`.

## Block assignment

Grok-1 emits one shard per top-level JAX leaf. The observed layout for
`ckpt-0` is one token-embedding singleton, one norm singleton, and 64
equal-sized transformer-block windows:

```
  shard   0              : token embedding          (1 shard)
  one edge singleton     : final/pre-head norm      (1 shard)
  remaining 64*K shards  : 64 transformer blocks    (K = 12 shards / block)
                           total = 770
```

The inventory layer computes:

```
  interior = shard_count - 2
  if interior % 12 == 0 and shard_count >= 3:
      k_per_block = 12
      n_blocks    = interior / 12
      choose the edge norm singleton from observed tensor kinds
      block_index = (shard_ordinal - first_block_shard) / k_per_block
      block_slot  = (shard_ordinal - first_block_shard) % k_per_block
```

If the divisor check fails the assignment is skipped; `block_index` and
`block_slot` remain `null` and downstream consumers should fall back to
`shard_ordinal` and `kind`.

The norm singleton's `BlockNorm` record is promoted to `FinalNorm` once
block assignment succeeds. For Grok-1 router canonicalization, the layout
choice is conservative: if the norm singleton appears immediately after the
embedding and the tail shard is router-shaped, the block window starts after
that singleton so all 64 `(d_model, n_experts)` routers receive canonical
`block_NNN.routing_slot_SS` names. If the evidence is insufficient, router
candidates remain unassigned and the routing report records a layout note.

## What is *not* inferred here

- The ordering of `up` vs. `gate` within a block. Both share the
  `(E, d_model, d_ff)` shape; disambiguation requires a later pass over
  the block's shard order and is explicitly out of scope for the
  cartography layer.
- Attention head count, head dimension, KV-head grouping. The f32
  attention projections are reported as `AttnProjF32` without further
  split.
- Any numerical property of the weights inside the inventory layer. The
  parser/inventory path does not read tensor bodies; the separate stats layer
  may sample payload values for offline profiling.
- Per-layer routing top-k, dropout layout, or anything that depends on
  training-time hyperparameters.

## Forward compatibility

Grok-2 is not yet supported. When it lands, the expected extensions are:

- A new `model_family` tag (`"grok-2"`).
- Additional `TensorKind` variants where Grok-2 introduces new tensor
  roles (e.g. a separate value projection) or new dtypes.
- A new block-layout entry alongside the Grok-1 `K = 12` rule.

The schema is designed to absorb these without breaking existing
consumers: the JSON `kind` enum is tagged, unknown variants round-trip as
`Unknown { reason }`, and `schema_version` bumps on any incompatible
change.
