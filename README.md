# xai-dissect

Static structural analysis of the Grok family of open-weight checkpoints.

`xai-dissect` parses raw xAI weight shards (Pickle Protocol 4 / JAX dumps),
builds a tensor inventory, and emits architecture, expert, routing, and
offline profiling reports. It is a read-only analysis tool: it never mutates
weights, never runs the model, and never produces inference output. The
inventory/expert/routing paths stay parser-driven; the stats/SAAQ-readiness
paths may sample tensor payload values for profiling.

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
- Offline stats profiling: sampled norm / variance / outlier / sparsity-ish
  heuristics per tensor and per layer.
- SAAQ-readiness profiling: routing-critical vs. potentially compressible
  regions, plus ranked candidate target manifests for future experiments.
- Summary statistics: parameter counts (raw / effective / per-expert),
  quantized vs. unquantized byte budgets, shard-size histograms.
- Exportable findings: stable, machine-readable artifacts (JSON / CSV /
  Markdown tables) suitable for downstream consumers.
- Unified output conventions for reusable `reports/`, `exports/`, and
  `manifests/` trees keyed by checkpoint slug.

## What this repo does NOT own

- A full inference runtime. No forward pass, no sampler, no decode loop.
- Projector logic (neuromorphic or otherwise).
- SAAQ latent calibration, SAAQ scoring, or latent-space experiments beyond
  offline readiness profiling.
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
`xai-dissect` does not compute SAAQ scores and does not calibrate latents.
Its role is upstream reconnaissance: sampled tensor statistics, routing-risk
flags, and candidate-target manifests that help decide where future SAAQ work
should focus.

### Surrogate_Viz.jl
`Surrogate_Viz.jl` owns visualization and dashboarding. `xai-dissect` emits
structured, exportable findings (JSON / CSV / Markdown). It does not render
plots, does not ship a UI, and does not embed a plotting stack.
`Surrogate_Viz.jl` is a downstream consumer of `xai-dissect` exports.

## Usage (current)

```bash
cargo build --release

# Inventory an entire checkpoint directory.
./target/release/xai-dissect inventory /path/to/grok-1/ckpt-0

# Peek at a single shard.
./target/release/xai-dissect dissect /path/to/grok-1/ckpt-0 --limit 1

# Expert and routing structure.
./target/release/xai-dissect experts /path/to/grok-1/ckpt-0
./target/release/xai-dissect routing-report /path/to/grok-1/ckpt-0

# Offline profiling for future SAAQ work.
./target/release/xai-dissect stats /path/to/grok-1/ckpt-0
./target/release/xai-dissect saaq-readiness /path/to/grok-1/ckpt-0

# Write a predictable artifact tree for downstream tooling.
./target/release/xai-dissect routing-report /path/to/grok-1/ckpt-0 \
  --output-root out
```

The parser-oriented commands memory-map each shard, validate the PROTO 4
header, scan the pickle byte grammar for ndarray anchors, and classify the
resulting tensors without executing model code. The stats-oriented commands
reuse those offsets and sample tensor payload values for offline profiling;
they still do not mutate weights or run inference.

See `docs/architecture.md` for the intended layer breakdown and
`docs/non_goals.md` for the full non-goals list. Unified artifact naming and
directory conventions are documented in `docs/output-conventions.md`.

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
