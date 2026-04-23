# Non-goals

`xai-dissect` is deliberately narrow. If a request falls into any of the
categories below, it belongs in a different repo or is out of scope
entirely. Pull requests expanding into these areas will be rejected.

## Out of scope

### Inference runtime
- No forward pass, no decode loop, no sampler, no KV cache.
- No model execution of any kind, even for "just a sanity check".

Offline tensor-value sampling for stats profiling is in scope; runtime model
execution is not.

### Projector logic
- No spiking / neuromorphic projection of activations or weights.
- No dimensionality reduction intended to feed a projector.
- Projector work lives in `snn-projector`.

### SAAQ latent calibration
- No SAAQ scoring, calibration, or latent-space analysis.
- No numerical interpretation that turns into a latent-space method or a
  quantization runtime.
- Latent work lives in `SAAQ-latent`.

### GPU kernels
- No CUDA / Metal / ROCm / SYCL code.
- No hand-tuned matmul, attention, or MoE kernels.
- No perf work beyond what is needed to scan shards on CPU.

### Symbolic regression
- No symbolic fitting over weights or activations.
- No program synthesis from tensor statistics.

### Plotting and dashboards
- No interactive UI, no web dashboard, no notebook-first workflows.
- No embedded plotting stack.
- Visualization lives in `Surrogate_Viz.jl`, which consumes our exports.

### Orchestration / hybrid runtime
- No multi-process scheduling, no job graph, no runtime glue across repos.
- No "run dissect then run projector then run SAAQ" wrappers.
- Orchestration lives in `corinth-canal`.

### Weight transformation
- No dequantization.
- No format conversion (Pickle -> safetensors, GGUF, etc.).
- No checkpoint rewriting or re-sharding.
- No production of runnable artifacts.

### Weight distribution
- This repo does not contain, mirror, vendor, or link-through any model
  weights.
- Users must obtain weights directly from the rights holder under the
  original license.

## Scope-adjacent items deferred, not rejected

These are not non-goals; they are "not yet":

- **Grok-2** support. Depends on a public release under a compatible
  license. No work starts until that exists.
- Additional dtypes beyond `f32` and `int8`. Added only when a supported
  checkpoint actually requires them.
- A stable Rust library API. The export schema is the stable surface
  first; an in-process API hardens later.

## Legal posture

`xai-dissect` is a tool for lawful analysis of weights the operator already
has legitimate access to. It is not a circumvention tool, not a scraper,
and not a redistribution channel. Contributions that assume or enable
unauthorized access to proprietary weights are out of scope and will be
rejected.
