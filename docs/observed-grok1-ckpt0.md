# Observed: Grok-1 `ckpt-0`

Empirical shard survey of the official xAI Grok-1 release, pinned to the
known public architecture. Every number below comes from looking at the
shard files on disk; no shard bodies are read.

## Shard size distribution in `ckpt-0/`

Before running `xai-dissect`, a simple

```bash
find ckpt-0 -type f -printf '%s\n' | sort | uniq -c
```

gives a clean histogram of the 770 shards (~297 GiB total). Every ndarray
payload inside a shard is stored verbatim; pickle adds only ~150-400 B of
framing, so file size alone already tells you roughly what is inside.

| Count | Size (bytes)   | Size (human) | Interpretation |
| ----: | -------------: | ------------ | -------------- |
|     1 |  3,221,225,637 | 3.0 GiB      | `tensor00000_000` - token embedding `(131072, 6144) f32`. Payload = 131072 x 6144 x 4 = 3,221,225,472 B + ~165 B pickle framing. |
|   128 |  1,611,137,347 | 1.5 GiB      | MoE / attention `QuantizedWeight8bit` shards, variant A. Int8 body is 8 x 6144 x 32768 = 1,610,612,736 B; remaining ~524 KB is f32 scales + pickle framing. 2 per layer. |
|    64 |  1,611,399,491 | 1.5 GiB      | Same class as variant A but exactly 262,144 B (= 65,536 f32) larger - consistent with a differently-shaped `scales` block on one of the three expert projections. 1 per layer. |
|    64 |     37,847,359 | 36 MiB       | f32 tensor sized ~9.46M elements; f32-attention projection companion, 1 per layer. |
|    64 |     37,761,334 | 36 MiB       | Sibling shape to the row above, 1 per layer. |
|   128 |      6,293,814 | 6.0 MiB      | f32 scales for the large quantized expert shards, 2 per layer. |
|    64 |        196,770 | 192 KiB      | Small f32 tensor, 1 per layer. Size matches `(6144, 8) f32 = 196,608 B + ~162 B framing` - the per-layer router / gate matrix. |
|   257 |         24,727 | 24 KiB       | f32 vector of 6144 elements (6144 x 4 = 24,576 B + framing) - per-layer RMSNorms + the final pre-head norm. |

Totals reconcile exactly:

```
1 + 128 + 64 + 64 + 64 + 128 + 64 + 257 = 770
```

The two buckets at the 1.5 GiB tier (`1,611,137,347` vs `1,611,399,491`,
delta = 262,144 = 64 KiB in f32) are consistent with the same base
ndarray shape carrying a different-sized `scales` block inside the
`QuantizedWeight8bit` dataclass - exactly what the parser already
targets.

## Per-bucket architectural mapping

Grok-1 is a 64-layer MoE with 8 experts (2 active per token), hidden size
6144, FFN inner size 32768, vocab 131072. Pinning each bucket to that
architecture:

- **1 x 3.0 GiB**: the token embedding, `(131072, 6144) f32`.
- **128 + 64 = 192 x ~1.5 GiB**: the three MoE expert feed-forward
  projections (up / gate / down) stacked across 8 experts per layer, for
  64 layers. `3 x 64 = 192`.
- **64 + 64 = 128 x 36 MiB**: two f32 attention-projection companions per
  layer for 64 layers. `2 x 64 = 128`.
- **128 x 6 MiB**: f32 scales for the large quantized expert shards, 2
  per layer for 64 layers. `2 x 64 = 128`.
- **64 x 192 KiB**: one router / gate tensor per layer, `(d_model,
  n_experts) = (6144, 8) f32`, 64 layers. `1 x 64 = 64`.
- **257 x 24 KiB**: per-layer RMSNorms of width `d_model = 6144`. The
  count decomposes as `4 x 64 + 1 = 257` - four norms per block
  (attention-Q, attention-K, pre-MLP, post-MLP) plus one final pre-head
  norm.

Every 1.5 GiB shard is a `QuantizedWeight8bit` dataclass (int8 weight +
f32 scales), so `xai-dissect` emits two rows per shard (`quant.weight`
and `quant.scales`). The plain f32 shards emit a single `tensor` row.

## Grok-1 shape sanity

Known public architecture:

- `d_model` = 6144
- `vocab_size` = 131072
- `num_layers` = 64
- `num_experts` = 8 (top-2 active per token)
- `d_ff` = 32768 (per expert)

Shard accounting:

```
770 shards = 1 embedding
           + 64 layers * 12 shards/layer
           + 1 final norm
         = 1 + 768 + 1 = 770
```

The 12 shards per layer decompose as:

```
3 quantized MoE expert projections (up / gate / down, 8 experts stacked)
+ 2 f32 attention projection companions
+ 2 f32 scales blocks for the large quantized experts
+ 1 router / gate
+ 4 RMSNorms
= 12
```

Close enough without a full `run.py` trace; the remaining ambiguity is
which quantized projection is `up` vs `gate` (they share the
`(8, 6144, 32768)` signature on disk). That is a downstream disambiguation
problem for a later analysis pass, not a cartography problem.

## Dissecting a single shard

To see exact shapes and byte offsets for the embedding alone without
pulling weights into RAM:

```bash
./target/release/xai-dissect dissect /path/to/grok-1-official/ckpt-0 --limit 1
```

This opens only `tensor00000_000`, memory-maps it, and prints the token
embedding's dtype, shape, payload offset, and `Nbytes`. Drop `--limit` to
sweep all 770 shards; the tool does not allocate tensor bodies, so the
whole scan is I/O-bound (a few seconds on NVMe even at 297 GiB).

For the full classified view across the checkpoint, use the `inventory`
subcommand instead:

```bash
./target/release/xai-dissect inventory /path/to/grok-1-official/ckpt-0 \
    --json out/inventory.json --md out/summary.md
```

See `docs/tensor-schema.md` for the classification rules applied to the
shapes observed above.
