# xai-dissect

Static structural analysis of the Grok family of open-weight checkpoints.

`xai-dissect` parses raw xAI weight shards (Pickle Protocol 4 / JAX dumps),
builds a tensor inventory, and emits architecture, expert, and routing
reports. It is a read-only structural tool: it never loads tensor bodies,
never runs the model, and never produces inference output.

Current target:

- **Grok-1** (xAI official release, Apache-2.0 weights). First-class, actively
  developed.
- **Grok-2** (if/when public weights are released under a compatible license).
  Not supported yet. Scoped as a deliberate follow-on, not a promise.

Only open/public weights are in scope. This repo is for lawful analysis of
weights you already have legitimate access to. No weights are redistributed
here.

## What this repo owns

- Parsing xAI Pickle/JAX shard files without a Python unpickler.
- A normalized tensor schema (dtype, shape, role, offsets, shard provenance).
- A full tensor inventory across a checkpoint directory.
- Architecture reports: layer count, embedding shapes, attention/MoE shape
  families, norm placement, head/vocab geometry.
- Expert analysis: per-layer expert count, expert-projection shapes
  (up / gate / down), quantization layout (`int8` weight + `f32` scales).
- Routing analysis: router/gate tensor identification, shape consistency
  across layers, top-k inference from shape alone where possible.
- Summary statistics: parameter counts (raw / effective / per-expert),
  quantized vs. unquantized byte budgets, shard-size histograms.
- Exportable findings: stable, machine-readable artifacts (JSON / CSV /
  Markdown tables) suitable for downstream consumers.

## What this repo does NOT own

- A full inference runtime. No forward pass, no sampler, no decode loop.
- Projector logic (neuromorphic or otherwise).
- SAAQ latent calibration, SAAQ scoring, or any latent-space experiments.
- GPU kernels, CUDA/Metal/ROCm code, or perf tuning of matmul paths.
- Symbolic regression over weights or activations.
- Plot-heavy / dashboard-heavy interactive analysis.
- Orchestration, hybrid runtime glue, or multi-repo scheduling.
- Dequantization, format conversion, or emission of runnable checkpoints.
- Redistribution of model weights.

Anything in that list belongs in a sibling repo (see below) or is explicitly
out of scope.

## Relationship to sibling repos

### corinth-canal
`corinth-canal` owns orchestration and hybrid-runtime glue: it wires models,
projectors, and downstream consumers together. `xai-dissect` is strictly
upstream of that: it produces structural descriptions of a frozen
checkpoint. If `corinth-canal` needs "what is the shape of expert 3's
down-projection in layer 17", it consumes an `xai-dissect` export. It does
not call into this repo at runtime.

### snn-projector
`snn-projector` owns projector logic, including any spiking / neuromorphic
projection of activations. `xai-dissect` does not implement, test, or depend
on projector math. It may describe the *shape* of tensors that a projector
would later consume (e.g. embedding width, expert output dimension), but it
never projects anything itself.

### SAAQ-latent
`SAAQ-latent` owns SAAQ latent calibration and latent-space analysis.
`xai-dissect` does not compute SAAQ scores, does not calibrate latents, and
does not load tensor values into memory. The boundary is sharp: `xai-dissect`
stops at "this tensor lives at this offset with this shape and dtype"; any
interpretation of its numerical contents belongs in `SAAQ-latent`.

### Surrogate_Viz.jl
`Surrogate_Viz.jl` owns visualization and dashboarding. `xai-dissect` emits
structured, exportable findings (JSON / CSV / Markdown). It does not render
plots, does not ship a UI, and does not embed a plotting stack.
`Surrogate_Viz.jl` is a downstream consumer of `xai-dissect` exports.

## Usage (current)

```bash
cargo build --release

# Inventory an entire checkpoint directory.
./target/release/xai-dissect /path/to/grok-1/ckpt-0

# Peek at a single shard.
./target/release/xai-dissect /path/to/grok-1/ckpt-0 --limit 1

# Non-default shard filename prefix.
./target/release/xai-dissect /path/to/dump --prefix shard_
```

The CLI memory-maps each shard, validates the PROTO 4 header, scans the
pickle byte grammar for ndarray anchors, and prints one row per tensor:
`Idx | Role | Dtype | Shape | Offset | Nbytes`. `QuantizedWeight8bit`
dataclasses are split into `quant.weight` (`int8`) and `quant.scales`
(`f32`) rows. No tensor bodies are read.

See `docs/architecture.md` for the intended layer breakdown and
`docs/non_goals.md` for the full non-goals list.

## Legal / ethical scope

- Analyze only weights you have lawful access to under their original
  license (Grok-1: Apache-2.0, distributed by xAI).
- This repository does not contain, mirror, or redistribute model weights.
- This repository does not ship circumvention tooling; it reads files the
  operator already possesses.

## License

GPL-3.0-only. See [LICENSE](LICENSE).

Grok model weights are the property of their respective rights holders and
are not covered by this repository's license.
