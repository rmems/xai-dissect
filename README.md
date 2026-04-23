# xai-dissect

Standalone Rust CLI for inspecting raw JAX / Pickle shards of the Grok-1 open
weights. Uses `memmap2` for zero-copy file reads and a targeted byte scanner
over the Pickle Protocol 4 grammar - no real unpickler, no ML framework, no
allocation of tensor bodies.

## What it reports

For every ndarray located inside a `tensor*` shard the tool prints:

| Column | Meaning |
| ------ | ------- |
| `Idx` | Per-file index (0-based, in byte order) |
| `Role` | `tensor`, `quant.weight`, or `quant.scales` |
| `Dtype` | `f32` or `int8` |
| `Shape` | Numpy-style shape tuple |
| `Offset` | Absolute byte offset of the raw payload within the shard |
| `Nbytes` | Payload length in bytes (verified against dtype * prod(shape)) |

`QuantizedWeight8bit` dataclasses (int8 weight + f32 scales) are detected and
their two inner ndarrays are labeled `quant.weight` / `quant.scales`.

## Usage

```bash
cargo build --release

# Sweep an entire checkpoint directory.
./target/release/xai-dissect /path/to/grok-1/ckpt-0

# Peek at just the first shard while iterating on the parser.
./target/release/xai-dissect /path/to/grok-1/ckpt-0 --limit 1

# Non-default filename prefix.
./target/release/xai-dissect /path/to/dump --prefix shard_
```

## Parsing strategy (summary)

1. Memory-map each shard file; never copy payload bytes.
2. Validate the leading `\x80\x04` PROTO 4 magic.
3. Locate every `\x8c\x02f4` / `\x8c\x02i1` short_binunicode whose forward
   post-amble matches numpy's `dtype(str, False, True)` reduce shape:
   `[\x94?] \x89 \x88 \x87`. This anchors one ndarray per hit and works even
   when CPython's pickler elides individual `\x94` MEMOIZE opcodes.
4. Walk backward through the variable-length "dtype class push" (full
   `STACK_GLOBAL` form or a `BINGET` / `LONG_BINGET` memo re-fetch) to find the
   shape tuple terminator (`\x85` / `\x86` / `\x87` / `)` / `t`) and decode the
   shape ints.
5. Walk forward looking for `[\x88|\x89]` (fortran-order bool) immediately
   followed by a bytes-payload opcode (`C`, `B`, or `\x8e`); read the payload
   length and record the absolute offset.
6. Mark any `__main__.QuantizedWeight8bit` class reference sites and label the
   adjacent int8/f32 pair as `quant.weight` / `quant.scales`.

## Observed: Grok-1 `ckpt-0` (xAI official release)

Before running the tool, a simple `find -printf '%s\n' | sort | uniq -c` over
`ckpt-0/` gives a clean histogram of the 770 shards (~297 GiB total). Because
every ndarray payload inside a shard is stored verbatim (pickle adds only
~150-400 bytes of framing), the file size alone already tells you roughly
what's inside.

| Count | Size (bytes)   | Size (human) | Likely contents |
| ----: | -------------: | ------------ | --------------- |
|     1 |  3,221,225,637 | 3.0 GiB      | `tensor00000_000` - token embedding `(131072, 6144) f32`. Payload = 131072 x 6144 x 4 = 3,221,225,472 B; the extra ~165 B is pickle framing. |
|   128 |  1,611,137,347 | 1.5 GiB      | MoE / attention `QuantizedWeight8bit` shards, variant A. Int8 body is 8 x 6144 x 32768 = 1,610,612,736 B; remaining ~524 KB holds the f32 scales + pickle framing. |
|    64 |  1,611,399,491 | 1.5 GiB      | Same class as variant A but exactly 262,144 B (= 65,536 f32) larger - consistent with a differently-shaped `scales` array on one of the three expert projections (up / gate / down). |
|    64 |     37,847,359 | 36 MiB       | f32 tensor sized ~9.46M elements; compatible with a `(vocab_size, ?)` router / head slice or an attention-projection f32 companion, 1 per layer. |
|    64 |     37,761,334 | 36 MiB       | Sibling shape to the row above, 1 per layer. |
|   128 |      6,293,814 | 6.0 MiB      | f32 scales for the large quantized expert shards, 2 per layer. |
|    64 |        196,770 | 192 KiB      | Small f32 tensor, 1 per layer - likely attention norm / bias. |
|   257 |         24,727 | 24 KiB       | f32 vector of 6144 elements (6144 x 4 = 24,576 B + pickle framing) - per-layer RMSNorms / residual norms. |

Totals cross-check: 1 + 128 + 64 + 64 + 64 + 128 + 64 + 257 = 770 shards.

Interpretation at a glance:

- Grok-1 is a 64-layer MoE with 8 experts; the counts align: most buckets are
  multiples of 64 (per-layer) and the two 1.5 GiB buckets together sum to
  3 x 64 = 192 shards, matching 3 MoE projections (up / gate / down) per layer.
- The 257 x 24 KiB count = 64 x 4 + 1 extra; the "+1" is the final pre-head
  norm and 4x per-layer matches attention-q-norm, attention-k-norm, pre-mlp
  norm, and post-mlp norm.
- Every 1.5 GiB shard is a `QuantizedWeight8bit` dataclass (int8 weight +
  f32 scales), so the tool will emit two rows per shard (`quant.weight`
  and `quant.scales`). The plain f32 shards emit a single `tensor` row.

### Dissecting a single shard

To see exact shapes and byte offsets without pulling weights into RAM:

```bash
./target/release/xai-dissect /path/to/grok-1-official/ckpt-0 --limit 1
```

This opens only `tensor00000_000`, memory-maps it, and prints the token
embedding's dtype, shape, payload offset, and `Nbytes`. Drop `--limit` to
sweep all 770 shards; the tool does not allocate tensor bodies, so the whole
scan is I/O-bound (a few seconds on NVMe even at 297 GiB).

## Non-goals

- Full Pickle deserialization.
- Decoding tensor values, dequantization, or file conversion.
- Dtypes beyond `float32` and `int8` (the only ones present in raw Grok-1).

## License

GPL-3.0-only. See [LICENSE](LICENSE).

The Grok-1 model weights themselves are distributed by xAI under
Apache-2.0 and are NOT included in or redistributed by this repository;
this tool only reads them out-of-band.
