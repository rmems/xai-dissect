# QuantizedWeight8bit Pickle Internals

This document explains how Grok-1's expert and attention projection
tensors are stored on disk as `QuantizedWeight8bit` JAX dataclass
instances, and how `xai-dissect` parses them without a Python interpreter.

## The JAX quantization packing

JAX models that support int8 quantized inference typically store
quantized weights as a dataclass with two fields:

- `weight`: the int8 weight matrix, stored as a flat ndarray
- `quantize_state`: the f32 quantization scales and zero-points

When Grok-1 saves its weights, the `QuantizedWeight8bit` dataclass is
flattened into a pytree and each leaf is saved as a separate
`numpy.ndarray` in the checkpoint. This means a single quantized projection
weight matrix appears as **three separate ndarray reduce sites** in the
pickle stream:

| Reduce site order | Role | dtype | Shape | xai-dissect role |
|-----------------|------|-------|-------|-----------------|
| 1st in shard | int8 weight body | i8 | `(E, d_model, d_ff)` for gate/up; `(E, d_ff, d_model)` for down | `quant.weight` |
| 2nd in shard | quantization scales | f32 | varies | `quant.scales` |
| 3rd in shard | min values (asymmetric) | f32 | varies | bare `tensor` |

`xai-dissect` does **not** decode the internal structure of this
dataclass. It records each top-level ndarray reduce site as a separate
`TensorInfo` record, preserving the pairing through the shard boundary
(the `quant.weight` and `quant.scales` records share the same `shard_path`;
additional f32 ndarrays remain bare `tensor` records).

## PROTO 4 pickle framing

The Grok-1 checkpoint shards are pickle files using PROTO version 4. The
format is:

```text
\x80\x04           PROTO magic (version 4)
<opcode stream>   variable-length opcode encoding
\x2e              STOP opcode
```

Each ndarray in the stream is encoded as a `GLOBAL` or `REDUCE` opcode
followed by metadata and a byte payload. The key opcodes `xai-dissect`
recognizes are:

| Opcode | Value | Meaning |
|--------|-------|---------|
| `PROTO` | `\x80` | Version marker — must be `\x80\x04` for Grok-1 shards |
| `GLOBAL` | `\x71` | Module + name reference (e.g. `numpy.core.multiarray`) |
| `REDUCE` | `\x72` | Callable + state (the dataclass reducer) |
| `BINPUT` | `\x61`..`\x80` | Short int marker for small memo indices |
| `MEMOIZE` | `\x94` | Store top of stack in memo |
| `STOP` | `\x2e` | End of frame |

`xai-dissect` walks the opcode stream looking for `numpy.ndarray` reduce
sites. It matches the magic `\x80\x04`, then scans forward through opcodes
counting `REDUCE` + `BINPUT` pairs (for the `__reduce__` protocol used by
JAX's pytree flattening). Every `numpy.core.multiarray` reducer it
encounters produces one raw tensor record.

## How QuantizedWeight8bit pairing works in xai-dissect

The parser (`src/parser/mod.rs`) processes each shard as follows:

1. Memory-map the shard file.
2. Verify `\x80\x04` at the head — reject if not PROTO 4.
3. Walk the opcode stream left-to-right.
4. For each `QuantizedWeight8bit` reduce site, assign roles by dtype:
   - `role = quant.weight` for the `i8` ndarray in the site
   - `role = quant.scales` for the first `f32` ndarray after that `i8`
     body within the same site
   - any additional `f32` ndarrays in the site remain bare `tensor`
5. The role assignment is dtype-aware — it does not rely on ndarray
   order alone.

This means every Grok-1 MoE/attention shard produces multiple records
(typically three ndarray reduce sites, with one `quant.weight`, one
`quant.scales`, and any remaining f32 leaves as bare `tensor`). Grok-1
never stores bare weight tensors mixed with quantized ones in the same
shard file — the pairing is always within a single shard.

## Why shape alone cannot disambiguate gate from up

The gate and up projections both have the outer shape
`(E=8, d_model=6144, d_ff=32768)` in Grok-1. When `xai-dissect` first
encounters a 3-D int8 `quant.weight`
with `dims[0] == n_experts == 8`, it initially assigns
`MoeExpertProjection { projection: unresolved }`. The resolved projection
labels (gate, down, up) are applied later by the block-slot assignment
pass, which uses the official Grok-1 Haiku checkpoint ordering:

```text
slot 0 → moe/linear/w   → gate   (GELU gate branch, input × d_ff)
slot 1 → moe/linear_1/w → down   (consumes gated input, d_ff × d_model)
slot 2 → moe/linear_v/w → up     (ungated value, d_model × d_ff)
```

The shape for gate and up is identical: `(8, 6144, 32768)`. The down
projection is transposed: `(8, 32768, 6144)`, so it is always identified
unambiguously.

## The two 1.5 GiB shard variants

In the real Grok-1 ckpt-0 run (`out/grok1_run2_after_fixes_20260525T002904Z/`),
the MoE `QuantizedWeight8bit` shards fall into two size buckets:

| Size | Bytes | Delta | Interpretation |
|-----:|------:|------:|-----------------|
| variant A | 1,611,137,347 | — | Base int8 body + standard quantization side-data |
| variant B | 1,611,399,491 | +262,144 | Same int8 body + 262,144 extra f32 values in the quantization state |

262,144 bytes = 65,536 f32 values. This is exactly the size of one
additional quantization axis (e.g., per-min-max range vs per-channel
scales) on one of the three projection families. The specific variant
distribution across the 192 MoE shards (128 × variant A, 64 × variant B)
is consistent with one projection family using a different per-axis
quantization scheme than the other two.

`xai-dissect` does not decode the internal quantization scheme. The
variant delta is observable in file size, but its meaning is opaque to
the parser. Both variants produce identical `role = quant.weight` records.

## Why int8 instead of FP16 or BF16?

Int8 quantization was chosen for the expert projections because:

- Memory: 8 experts × 32,768 FFN width × 6,144 model width × 1 byte =
  1.5 GiB per projection family per layer, vs 6.0 GiB for FP32. The
  memory savings enable fitting the full model in GPU memory.
- Bandwidth: Router computations are memory-bandwidth-bound. Int8
  weight reads reduce bandwidth by 4×.
- Accuracy: The per-expert scales capture the distribution shape well
  enough that the quantization error is dominated by the clipping noise,
  not the int8 representation.

The router weights (`(d_model, n_experts) = (6144, 8)`) are stored as
bare FP32 (192 KiB per layer). Quantizing the router changes the expert
selection behavior in a non-linear way, which is why the GO/NO-GO gate
requires `passthrough_f32_router` in the first pilot pass.