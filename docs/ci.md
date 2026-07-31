# CI for xai-dissect

GitHub Actions workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).  
Tracked as [issue #33](https://github.com/rmems/xai-dissect/issues/33) / Linear **RM-148**.

## Jobs

| Job | When | Required for merge? | What it does |
|-----|------|---------------------|--------------|
| **rust-ci** | PR + `main` | **Yes** (branch-protection gate) | `cargo fmt --check`, `cargo test --locked`, `cargo clippy -D warnings`, CLI `--help` smokes |
| **coverage** | After rust-ci | Coverage generation yes; upload soft | `cargo llvm-cov` → `lcov.info` → Codecov (`CODECOV_TOKEN` if set, else OIDC) |
| **qodana** | PR + `main` | Not a required merge gate; job fails only when token is set and scan fails | JetBrains Qodana for Rust (`qodana.yaml`); skips when token unset |
| **release-observability** | `main` push only | Soft / skip if unconfigured | Optional Sentry release via `scripts/observability/sentry_release.sh` |

**Out of scope:** New Relic, Aikido, checkpoint downloads, GPU runners.

## Optional repository secrets

Set under GitHub → Settings → Secrets and variables → Actions:

| Secret | Job | Required? |
|--------|-----|-----------|
| `CODECOV_TOKEN` | coverage | Optional. When set, used for upload; when empty, OIDC (`use_oidc`) is enabled. Upload still soft-fails. |
| `QODANA_TOKEN` | qodana | Optional. When **set**, scan runs and the **job** can fail. When **unset**, job skips. Does not replace `rust-ci` as the required merge check unless you add it to branch protection. |
| `SENTRY_AUTH_TOKEN` | release-observability | Optional |
| `SENTRY_ORG` | release-observability | Optional (with token + project) |
| `SENTRY_PROJECT_XAI_DISSECT` | release-observability | Optional |

### Disable Sentry

Omit any of the three Sentry secrets. Install and release steps are skipped (no binary download).

### Disable / soften Qodana

Omit `QODANA_TOKEN`. The Qodana job skips analysis and stays green. Only **rust-ci** is the required merge gate by default.

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
- `sentry-cli` is installed only when Sentry is configured, from a **version-pinned** GitHub release binary with **SHA-256 verification** (no `curl | bash`)
- Secret-backed steps skip when secrets are missing
- Fork PRs should not receive repository secrets from GitHub
- Concurrency cancels only PR runs (not in-flight `main` Sentry releases)
