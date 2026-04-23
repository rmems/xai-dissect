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

## Non-goals

- Full Pickle deserialization.
- Decoding tensor values, dequantization, or file conversion.
- Dtypes beyond `float32` and `int8` (the only ones present in raw Grok-1).

## License

GPL-3.0-only. See [LICENSE](LICENSE).

The Grok-1 model weights themselves are distributed by xAI under
Apache-2.0 and are NOT included in or redistributed by this repository;
this tool only reads them out-of-band.
