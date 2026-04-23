# Architecture

`xai-dissect` is organized as a pipeline of thin, composable layers. Each
layer has one job, consumes only the layer below it, and is individually
testable against a single shard or a full checkpoint directory.

```
  raw shard bytes
        |
        v
  [ parser ]        -> pickle-grammar byte scan, no unpickler
        |
        v
  [ schema ]        -> normalized tensor records
        |
        v
  [ inventory ]     -> per-checkpoint tensor table
        |
        +--> [ expert analysis ]   -> MoE expert geometry
        +--> [ routing analysis ]  -> router / gate geometry
        +--> [ stats ]             -> offline tensor-value profiling
        |
        v
  [ reports / exports ]  -> JSON, CSV, Markdown
```

## Layers

### 1. parser
Memory-maps each shard, validates the `\x80\x04` PROTO 4 magic, and walks
the pickle byte grammar to locate every `numpy.ndarray` reduce site. Emits
raw `TensorAnchor` records: shard path, byte offset, payload length, dtype
token, shape tuple, and any enclosing `QuantizedWeight8bit` marker. Never
decodes tensor bodies. Never depends on a Python interpreter.

### 2. schema
Normalizes `TensorAnchor` records into a stable `TensorRecord`:

- `shard_path`, `shard_index`
- `role` (`tensor` | `quant.weight` | `quant.scales`)
- `dtype` (`f32` | `int8`, extensible)
- `shape` (`Vec<u64>`)
- `offset`, `nbytes`
- optional `group_id` linking paired quant weight + scales

This is the canonical on-disk / on-wire record format for every other layer
and every export.

### 3. inventory
Aggregates `TensorRecord`s across a checkpoint directory into a single
queryable table. Responsibilities:

- dedup and order shards deterministically
- verify `nbytes == dtype_size * prod(shape)`
- build indices by shard, by role, and by shape family
- detect missing/extra shards vs. an expected count

The inventory is the single source of truth consumed by every downstream
analyzer.

### 4. expert analysis
Identifies Mixture-of-Experts structure from inventory records alone:

- per-layer expert count (inferred from leading shape dimension of expert
  projections, e.g. `(8, 6144, 32768)` -> 8 experts)
- up / gate / down projection families and their quantization layout
- expert-local parameter counts (raw, quantized, effective)
- consistency checks across layers (same expert count, same inner dims)

Shape-only. No activation, no routing decision, no runtime execution.

### 5. routing analysis
Identifies router / gate tensors and per-layer routing geometry:

- locate router weight tensors by shape signature `(d_model, n_experts)`
  or equivalent
- cross-check against expert count from (4)
- report top-k structure where it can be inferred from shapes
- flag layers whose router shape disagrees with sibling layers

Explicitly does not execute routing, does not rank experts, and does not
touch activation data.

### 6. stats
Offline profiling over the inventory plus sampled tensor payloads:

- norm, variance, outlier, and sparsity-ish summaries
- per-layer and per-tensor metrics
- SAAQ-readiness scouting: likely routing-critical vs. potentially
  compressible regions
- ranked candidate-target manifests for future experiments

Read-only payload sampling is allowed here. No weight mutation, no
quantization runtime, and no model execution.

### 7. reports / exports
Emits stable, machine-readable artifacts:

- `inventory.json` - full `TensorRecord` array
- `architecture.md` - human-readable summary (layer count, d_model,
  n_experts, head geometry, norm placement)
- `experts.json` / `routing.json` - structured findings from (4) and (5)
- `stats.json` / `saaq-readiness.json` - structured profiling outputs from (6)
- `candidate-manifest.json` - ranked machine-readable target list for future experiments

Exports are the only supported integration surface for downstream repos
(`corinth-canal`, `SAAQ-latent`, `Surrogate_Viz.jl`). No in-process API is
guaranteed across versions; the export schema is.

## Minimal initial file tree (proposed)

```
xai-dissect/
  Cargo.toml
  README.md
  LICENSE
  src/
    main.rs                  # CLI entry (today's binary, preserved)
    lib.rs                   # re-exports for the layers below
    parser/
      mod.rs
      pickle_scan.rs         # PROTO 4 byte-grammar scanner
      quantized.rs           # QuantizedWeight8bit anchor pairing
    schema/
      mod.rs                 # TensorRecord, Role, Dtype
    inventory/
      mod.rs                 # checkpoint walk, dedup, verification
    analysis/
      mod.rs
      experts.rs             # MoE geometry from shapes
      routing.rs             # router/gate identification
      stats.rs               # parameter / byte accounting
    report/
      mod.rs
      json.rs                # inventory.json, experts.json, routing.json
      csv.rs                 # stats.csv
      markdown.rs            # architecture.md renderer
  docs/
    architecture.md          # this file
    non_goals.md
  tests/
    fixtures/                # tiny synthetic shards (no real weights)
    parser_smoke.rs
    inventory_smoke.rs
```

The tree is aspirational; today's `src/` contains a single-file CLI. The
migration is additive: `main.rs` stays, `lib.rs` + submodules grow under
it, and the CLI is rewritten to call into the library once the schema is
stable.
