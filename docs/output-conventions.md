# Output Conventions

`xai-dissect` supports two output modes:

- explicit per-file flags such as `--json`, `--md`, and `--manifest`
- a unified output tree via `--output-root <dir>`

The unified tree exists so downstream tooling can consume artifacts from
different analysis commands without guessing filenames or stitching together
ad hoc directories.

## Directory layout

When `--output-root out` is provided, artifacts are written under:

```text
out/
  reports/<checkpoint_slug>/
  exports/<checkpoint_slug>/
  manifests/<checkpoint_slug>/
```

`<checkpoint_slug>` is inferred from the checkpoint path by default using the
last two path components, sanitized for filesystem safety. Example:

```text
/home/user/grok-1-official/ckpt-0
-> grok-1-official__ckpt-0
```

Use `--checkpoint-slug <slug>` to override that value when a repo or
pipeline needs a stable custom name.

## Artifact conventions

### `inventory`

- `reports/<slug>/inventory.md`
- `exports/<slug>/inventory.json`
- `exports/<slug>/inventory-findings.json`
- `manifests/<slug>/checkpoint-inventory-snapshot.json`

### `experts`

- `reports/<slug>/experts.md`
- `exports/<slug>/experts.json`
- `exports/<slug>/experts-findings.json`

### `routing-report`

- `reports/<slug>/routing-report.md`
- `exports/<slug>/routing-report.json`
- `exports/<slug>/routing-report-findings.json`
- `manifests/<slug>/routing-critical-tensors.json`

### `stats`

- `reports/<slug>/stats.md`
- `exports/<slug>/stats.json`
- `exports/<slug>/stats-findings.json`

### `saaq-readiness`

- `reports/<slug>/saaq-readiness.md`
- `exports/<slug>/saaq-readiness.json`
- `exports/<slug>/saaq-readiness-findings.json`
- `manifests/<slug>/candidate-saaq-targets.json`

## Summary and manifest intent

- `reports/`: human-readable Markdown meant for inspection and PR review
- `exports/`: full JSON exports plus compact findings summaries for scripts
- `manifests/`: focused machine-readable lists intended for downstream
  selection, orchestration, and snapshot comparison

The output tree is additive. Existing explicit file flags remain supported
and are not replaced by the unified tree.

See `docs/export-contracts.md` for the stable schema types behind each
artifact.
