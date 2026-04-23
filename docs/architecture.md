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
        +--> [ reports ]           -> human-readable Markdown
        +--> [ exports ]           -> full JSON + findings summaries
        |
        v
  [ manifests ]          -> focused machine-readable inventories
```

## Layers

### 1. parser
Memory-maps each shard, validates the `\x80\x04` PROTO 4 magic, and walks
the pickle byte grammar to locate every `numpy.ndarray` reduce site. Emits
raw `parser::RawTensor` records: role, dtype, shape, payload offset, and
payload length, with `QuantizedWeight8bit` pairing preserved in the role
assignment. Never
decodes tensor bodies. Never depends on a Python interpreter.

### 2. schema
Normalizes parser output into stable serializable types such as
`TensorInfo`, `ModelInventory`, `ExpertAtlas`, `RoutingReport`,
`StatsProfileReport`, and `SaaqReadinessReport`:

- `shard_path`, `shard_index`
- `role` (`tensor` | `quant.weight` | `quant.scales`)
- `dtype` (`f32` | `int8`, extensible)
- `shape` (`Vec<u64>`)
- `offset`, `nbytes`
- optional `group_id` linking paired quant weight + scales

This is the canonical on-disk / on-wire record format for every other layer
and every export.

### 3. inventory
Aggregates parser records across a checkpoint directory into a single
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

- `reports/<slug>/*.md` - human-readable summaries for inspection
- `exports/<slug>/*.json` - full structured outputs plus compact findings summaries
- `manifests/<slug>/*.json` - focused machine-readable lists such as:
  checkpoint inventory snapshots, routing-critical tensors, and ranked SAAQ
  target candidates

Exports are the only supported integration surface for downstream repos
(`corinth-canal`, `SAAQ-latent`, `Surrogate_Viz.jl`). No in-process API is
guaranteed across versions; the export schema is.

## Current file tree

```
xai-dissect/
  Cargo.toml
  README.md
  CHANGELOG.md
  LICENSE
  src/
    main.rs                  # CLI entry
    lib.rs                   # library entry + module exports
    parser/
      mod.rs                 # PROTO 4 byte-grammar scanner
    schema/
      mod.rs                 # stable serializable schema types
    inventory/
      mod.rs                 # checkpoint walk, dedup, classification
    experts/
      mod.rs                 # MoE geometry from inventory records
    routing/
      mod.rs                 # router / gate structure analysis
    stats/
      mod.rs                 # offline tensor-statistics profiling
    report/
      mod.rs                 # Markdown / JSON writers and renderers
    exports/
      mod.rs                 # output-tree planning and manifest bundles
  docs/
    architecture.md          # this file
    export-contracts.md      # stable artifact contract
    output-conventions.md    # bundle paths and filenames
    non_goals.md
    tensor-schema.md         # inventory schema details
  tests/
    fixtures/
      parser/                # tiny synthetic pickle fixtures
      exports/               # golden snapshot files
    support/
      mod.rs                 # synthetic sample builders
    cli_help.rs              # CLI ergonomics coverage
    export_contracts.rs      # bundle snapshot tests
    parser_inventory.rs      # parser/inventory fixture tests
```

This tree reflects the current milestone, not an aspirational future layout.
The repo is already organized around parser, schema, inventory, expert,
routing, stats, report, and export modules, with the CLI in `src/main.rs`
acting as a thin entry point over those layers.
