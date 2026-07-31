# CI for xai-dissect

GitHub Actions workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).  
Tracked as [issue #33](https://github.com/rmems/xai-dissect/issues/33) / Linear **RM-148**.

## Jobs

| Job | When | Required for merge? | What it does |
|-----|------|---------------------|--------------|
| **rust-ci** | PR + `main` | **Yes** | `cargo fmt --check`, `cargo test --locked`, `cargo clippy -D warnings`, CLI `--help` smokes |
| **coverage** | After rust-ci | Coverage generation yes; Codecov upload soft | `cargo llvm-cov` → `lcov.info` → Codecov |
| **qodana** | PR + `main` | Soft without token | JetBrains Qodana for Rust (`qodana.yaml`) |
| **release-observability** | `main` push only | Soft | Optional Sentry release via `scripts/observability/sentry_release.sh` |

**Out of scope:** New Relic, Aikido, checkpoint downloads, GPU runners.

## Optional repository secrets

Set under GitHub → Settings → Secrets and variables → Actions:

| Secret | Job | Required? |
|--------|-----|-----------|
| `CODECOV_TOKEN` | coverage | Optional (public repos often upload without it; upload step soft-fails) |
| `QODANA_TOKEN` | qodana | Required for full Qodana for Rust (Ultimate/Cloud). Job uses `continue-on-error` when unset |
| `SENTRY_AUTH_TOKEN` | release-observability | Optional |
| `SENTRY_ORG` | release-observability | Optional (with token + project) |
| `SENTRY_PROJECT_XAI_DISSECT` | release-observability | Optional |

### Disable Sentry

Omit any of the three Sentry secrets. The job prints a skip message and exits 0.

### Disable / soften Qodana

Omit `QODANA_TOKEN`. The Qodana job is allowed to fail without blocking the workflow conclusion for required checks (only **rust-ci** is the hard gate today).

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
- Secret-backed steps skip or soft-fail when secrets are missing
- Fork PRs should not receive repository secrets from GitHub
