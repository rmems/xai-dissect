# CI for xai-dissect

GitHub Actions workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).  
Tracked as [issue #33](https://github.com/rmems/xai-dissect/issues/33) / Linear **RM-148**.

## Jobs

| Job | When | Required for merge? | What it does |
|-----|------|---------------------|--------------|
| **rust-ci** | PR + `main` | **Yes** (branch-protection gate) | `cargo fmt --check`, `cargo test --locked`, `cargo clippy -D warnings`, CLI `--help` smokes |
| **coverage** | After rust-ci | Coverage generation yes; upload soft | `cargo llvm-cov` → `lcov.info` → Codecov (`CODECOV_TOKEN` if set, else OIDC) |
| **qodana** | PR + `main` | Not a required merge gate; scan step uses `continue-on-error` (Rust linter is EAP) | JetBrains Qodana for Rust (`qodana.yaml`); skips when `QODANA_TOKEN` unset |
| **release-observability** | `main` push only | Not a merge gate; skips if unconfigured; configured failures fail the job | Optional Sentry release via `scripts/observability/sentry_release.sh` |

**Out of scope:** New Relic, Aikido, checkpoint downloads, GPU runners.

## Optional repository secrets

Set under GitHub → Settings → Secrets and variables → Actions:

| Secret | Job | Required? |
|--------|-----|-----------|
| `CODECOV_TOKEN` | coverage | Optional. When set, used for upload; when empty, OIDC (`use_oidc`) is enabled. Upload still soft-fails. |
| `QODANA_TOKEN` | qodana | Optional. JetBrains Cloud **project** token from the [project card](https://qodana.cloud/). When set, the scan runs (soft-fail on EAP timeout). When unset, the job skips. Not a merge gate. |
| `SENTRY_AUTH_TOKEN` | release-observability | Optional |
| `SENTRY_ORG` | release-observability | Optional (with token + project); org slug is **`limen-neural`** |
| `SENTRY_PROJECT_XAI_DISSECT` | release-observability | Optional; project slug **`xai-dissect`** (dedicated Rust project) |

Do **not** use `QODANA_CONFIGURATIONS_TOKEN` as the scan token — that is an uploader/config token, not a Cloud project token.

### Disable Sentry (CI release markers)

Omit any of the three Sentry secrets. Install and release steps are skipped (no binary download).

## Opt-in Sentry for real-weight CLI runs

Runtime capture is **off by default** (public CLI must not phone home).

Enable only on machines where you intentionally want crash/error reports during
Grok-1 weight campaigns:

```bash
export XAI_DISSECT_SENTRY=1
# DSN for limen-neural / xai-dissect (not the shared liquidcortex/rust projects):
export SENTRY_DSN='https://…@….ingest.us.sentry.io/…'
# optional — use full SHA so runtime release matches CI markers:
export SENTRY_ENVIRONMENT=local-weights
export AGENTOS_GIT_SHA="$(git rev-parse HEAD)"
# optional stable correlation id for logs + Sentry:
export AGENTOS_RUN_ID="weights-$(date -u +%Y%m%d)-1"

./target/release/xai-dissect inventory /path/to/grok-1/ckpt-0
```

Local DSN helper (gitignored machine config, never commit):

```bash
# after creating the project key in Sentry UI/API:
#   ~/.config/xai-dissect/sentry_dsn.env  → SENTRY_DSN=...
# Always clear allexport even if source fails (missing/invalid file).
set -a
source ~/.config/xai-dissect/sentry_dsn.env
src_status=$?
set +a
if [ "$src_status" -ne 0 ]; then
  printf 'failed to source sentry_dsn.env (status=%s)\n' "$src_status" >&2
  return "$src_status" 2>/dev/null || exit "$src_status"
fi
```

What is sent:

- Panics (via Sentry panic integration) with tags `repo`, `run_id` (and `command` after CLI parse)
- Top-level command `anyhow` failures via `capture_anyhow` (full error chain) with
  the same tags **plus** `error_category` (only on command failures). The full
  error chain is application-controlled text; `before_send` redacts `$HOME`
  prefixes from messages / exception values / stack paths / breadcrumbs, but
  does **not** strip arbitrary error content
- Release name: `xai-dissect@<AGENTOS_GIT_SHA|unknown>` (same fallback as
  `scripts/observability/sentry_release.sh` and `observability::git_sha`)
- With the `contexts` feature: device/OS/rustc metadata (not weight data). `server_name` is fixed to
  `xai-dissect` (machine hostname is not advertised)

SDK defaults that stay off (does **not** claim application error strings are scrubbed):

- Weight tensors / checkpoint bytes
- Default SDK PII only (`send_default_pii = false` — IPs/headers via HTTP integrations)
- Performance transactions (traces strategy remains **Disabled**; no sample rate configured)
- Events when `XAI_DISSECT_SENTRY` is unset, or when `SENTRY_DSN` is empty/invalid
CI release markers (main only) use `SENTRY_AUTH_TOKEN` + org/project secrets and
do not require a DSN. Runtime capture uses `SENTRY_DSN` + the enable flag.
Invalid DSNs soft-disable Sentry instead of panicking the CLI.

## Qodana Cloud setup

`qodana-rust` is **Ultimate-only** — it needs a [Qodana Cloud](https://qodana.cloud/) project token. Without `QODANA_TOKEN`, CI **skips** the scan (green job, no Cloud report).

1. Create account / org / team / project on [qodana.cloud](https://qodana.cloud/) for `rmems/xai-dissect`
2. Copy the **project token** from the project card
3. GitHub → Settings → Secrets and variables → Actions → add **`QODANA_TOKEN`**
4. Re-run CI; expect a long Docker scan when the EAP linter can open the project

Rust image tags on Docker Hub (as of 2026-08): `latest`, `2026.2-eap`, `2026.1-eap`. CI passes `--image jetbrains/qodana-rust:2026.2-eap`. Scan step uses `continue-on-error` because project-open timeouts are common on GitHub-hosted runners.

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
- Third-party Actions are pinned to full commit SHAs (checkout, rust-toolchain, rust-cache, install-action, codecov, qodana), not floating major tags
- Secret-backed steps skip when secrets are missing
- Fork PRs should not receive repository secrets from GitHub
- Concurrency cancels only PR runs (not in-flight `main` Sentry releases)
