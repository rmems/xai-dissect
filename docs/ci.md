# CI for xai-dissect

GitHub Actions workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).  
Tracked as [issue #33](https://github.com/rmems/xai-dissect/issues/33) / Linear **RM-148**.

## Jobs

| Job | When | Required for merge? | What it does |
|-----|------|---------------------|--------------|
| **rust-ci** | PR + `main` | **Yes** | `cargo fmt --check`, `cargo test --locked`, `cargo clippy -D warnings`, CLI `--help` smokes |
| **coverage** | After rust-ci | Coverage generation yes; Codecov upload soft | `cargo llvm-cov` → `lcov.info` → Codecov (token or OIDC) |
| **qodana** | PR + `main` | Soft without token; **hard when `QODANA_TOKEN` set** | JetBrains Qodana for Rust (`qodana.yaml`) |
| **release-observability** | `main` push only | Soft | Optional Sentry release via `scripts/observability/sentry_release.sh` |

**Out of scope:** New Relic, Aikido, checkpoint downloads, GPU runners.

## Optional repository secrets

Set under GitHub → Settings → Secrets and variables → Actions:

| Secret | Job | Required? |
|--------|-----|-----------|
| `CODECOV_TOKEN` | coverage | Optional (OIDC tokenless upload enabled via `use_oidc: true`; upload still soft-fails) |
| `QODANA_TOKEN` | qodana | Optional. When **set**, scan runs and can fail the job. When **unset**, job skips with a note. |
| `SENTRY_AUTH_TOKEN` | release-observability | Optional |
| `SENTRY_ORG` | release-observability | Optional (with token + project) |
| `SENTRY_PROJECT_XAI_DISSECT` | release-observability | Optional |

### Disable Sentry

Omit any of the three Sentry secrets. The job prints a skip message and exits 0.

### Disable / soften Qodana

Omit `QODANA_TOKEN`. The Qodana job skips analysis and stays green; only **rust-ci** is the required merge gate.

## Local commands (same as CI)

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo run --locked -- --help
cargo run --locked -- quant-plan --help
cargo run --locked -- inventory --help
cargo run --locked -- saaq-readiness --help
```

Coverage (optional locally):

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --locked --lcov --output-path lcov.info
```

## Security

- No tokens, DSNs, or private paths in the tree
- Workflow default permissions are `contents: read`; Qodana alone gets `checks`/`pull-requests` write
- `sentry-cli` is installed from a **version-pinned** GitHub release binary (no `curl \| bash`)
- Secret-backed steps skip when secrets are missing
- Fork PRs should not receive repository secrets from GitHub
- Concurrency cancels only PR runs (not in-flight `main` Sentry releases)
