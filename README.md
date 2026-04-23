# xai-dissect

Static structural analysis of Grok-family open-weight checkpoints.

`xai-dissect` is a read-only checkpoint dissector. It parses raw xAI shard
files, builds a normalized tensor inventory, and emits structural reports for
experts, routing, and future SAAQ-oriented profiling. It does not run the
model, mutate weights, or act as an inference runtime.

Current release target:

- **Grok-1**: supported now
- **Grok-2**: not supported yet; tracked as a future follow-on only if public
  weights are released under a compatible license

Only open/public weights are in scope. This repo analyzes weights you already
have lawful access to and does not redistribute them.

## What It Does

- Parses raw Grok shard files without a Python unpickler
- Builds a stable tensor inventory with dtype, shape, role, offsets, and shard
  provenance
- Maps MoE expert structure and block-to-expert organization
- Identifies likely routing tensors and routing-critical regions
- Profiles offline tensor statistics for future SAAQ experimentation
- Writes predictable Markdown, JSON, and manifest artifacts for downstream
  tooling

## What It Does Not Do

- No forward pass, logits, decode loop, or runtime inference
- No quantization runtime, checkpoint mutation, or format conversion
- No projector logic, dashboard UI, or orchestration layer
- No redistribution of model weights

See [docs/non_goals.md](docs/non_goals.md) for the full non-goals list.

## Quick Start

```bash
cargo build --release
cargo test

# Show the available commands
./target/release/xai-dissect --help
```

Main commands:

- `inventory`: full checkpoint inventory and architecture-oriented summary
- `experts`: expert atlas for MoE block structure
- `routing-report`: routing/gating structure inspection
- `stats`: offline tensor-statistics profiling
- `saaq-readiness`: candidate scouting for future SAAQ experiments

## Usage Examples

All examples assume a checkpoint directory such as
`/path/to/grok-1/ckpt-0`.

### Inventory

```bash
./target/release/xai-dissect inventory /path/to/grok-1/ckpt-0 \
  --json out/inventory.json \
  --md out/inventory.md
```

### Experts

```bash
./target/release/xai-dissect experts /path/to/grok-1/ckpt-0 \
  --json out/experts.json \
  --md out/experts.md
```

### Routing Report

```bash
./target/release/xai-dissect routing-report /path/to/grok-1/ckpt-0 \
  --json out/routing-report.json \
  --md out/routing-report.md
```

### Stats

```bash
./target/release/xai-dissect stats /path/to/grok-1/ckpt-0 \
  --sample-values 65536 \
  --json out/stats.json \
  --md out/stats.md
```

### SAAQ Readiness

```bash
./target/release/xai-dissect saaq-readiness /path/to/grok-1/ckpt-0 \
  --sample-values 65536 \
  --json out/saaq-readiness.json \
  --md out/saaq-readiness.md \
  --manifest out/candidate-saaq-targets.json
```

### Unified Output Tree

```bash
./target/release/xai-dissect routing-report /path/to/grok-1/ckpt-0 \
  --output-root out
```

That produces a predictable artifact layout such as:

```text
out/
  reports/<checkpoint_slug>/
  exports/<checkpoint_slug>/
  manifests/<checkpoint_slug>/
```

See [docs/output-conventions.md](docs/output-conventions.md) for the full
artifact naming convention.

## Outputs

The repo writes three artifact families:

- `reports/`: human-readable Markdown for inspection and review
- `exports/`: full JSON plus compact findings summaries
- `manifests/`: focused machine-readable lists for downstream selection and
  orchestration

Examples:

- `inventory` writes a checkpoint inventory plus an inventory snapshot manifest
- `routing-report` writes a routing report plus a routing-critical tensor list
- `saaq-readiness` writes a readiness report plus a ranked candidate manifest

## Stability Notes

This milestone is intended to feel like a coherent tool, not a runtime:

- CLI-first workflow
- parser/analysis orientation
- stable export schema favored over a broad in-process Rust API
- current checkpoint support centered on Grok-1 `f32` and `int8` shard layouts

Release notes live in [CHANGELOG.md](CHANGELOG.md).

## Future Grok-2 Support

Grok-2 is not yet in scope for implementation, but the repo now includes a
future-support checklist and issue template to keep that work bounded when the
time comes:

- [docs/grok2-future-support.md](docs/grok2-future-support.md)
- [.github/ISSUE_TEMPLATE/grok2-support.md](.github/ISSUE_TEMPLATE/grok2-support.md)

## Relationship To Sibling Repos

- `corinth-canal`: orchestration and runtime glue
- `snn-projector`: projector logic
- `SAAQ-latent`: latent calibration and latent-space analysis
- `Surrogate_Viz.jl`: visualization and dashboards

`xai-dissect` stays upstream of those projects. Its job is to emit structural
artifacts from a frozen checkpoint.

## Architecture

See:

- [docs/architecture.md](docs/architecture.md)
- [docs/tensor-schema.md](docs/tensor-schema.md)
- [docs/output-conventions.md](docs/output-conventions.md)

## Legal / Ethical Scope

- Analyze only weights you have lawful access to under the original license
- This repository does not contain or mirror model weights
- This repository is not a circumvention or scraping tool

## License

GPL-3.0-only. See [LICENSE](LICENSE).

Grok model weights remain the property of their respective rights holders and
are not covered by this repository's license.
