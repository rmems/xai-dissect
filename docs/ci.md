# CI for xai-dissect

GitHub Actions workflows: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
and [`.github/workflows/security.yml`](../.github/workflows/security.yml).  
Tracked as [issue #33](https://github.com/rmems/xai-dissect/issues/33) / Linear **RM-148**.

## Jobs

| Job | When | Required for merge? | What it does |
|-----|------|---------------------|--------------|
| **rust-ci** | PR + `main` | **Yes** (branch-protection gate) | `cargo fmt --check`, `cargo test --locked`, `cargo clippy -D warnings`, CLI `--help` smokes |
| **msrv** | PR + `main` | Not a required merge gate (recommended) | `cargo check --locked --all-targets --all-features` on the toolchain floor in `Cargo.toml` `rust-version`, plus an assertion that the installed `rustc` matches the manifest |
| **coverage** | After rust-ci | Coverage generation yes; upload soft | `cargo llvm-cov` → `lcov.info` → Codecov (`CODECOV_TOKEN` if set, else OIDC) |
| **qodana** | PR + `main` | Not a required merge gate; scan step uses `continue-on-error` (Rust linter is EAP) | JetBrains Qodana for Rust (`qodana.yaml`); skips when `QODANA_TOKEN` unset |
| **cargo-audit** (`security.yml`) | PR + `main` + daily 06:00 UTC + manual | No — advisory only | `cargo audit` against `Cargo.lock`; log uploaded as an artifact and rendered into the run summary |
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
# Under `set -e`, bare `source` would abort before `src_status`/`set +a`;
# `if source` is exempt from errexit so we can always restore allexport.
set -a
if source ~/.config/xai-dissect/sentry_dsn.env; then
  src_status=0
else
  src_status=$?
fi
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

There **is** a Qodana for Rust product — CI uses **`jetbrains/qodana-rust:2026.2-eap`**
(`qodana.yaml` sets `linter: qodana-rust`). It is **not** free Community edition:
Rust support is **Ultimate / Cloud** and needs a [Qodana Cloud](https://qodana.cloud/)
project token. Without `QODANA_TOKEN`, CI **skips** the scan (green job, no Cloud report).

1. Create account / org / team / project on [qodana.cloud](https://qodana.cloud/) for `rmems/xai-dissect`
2. Copy the **project token** from the project card
3. GitHub → Settings → Secrets and variables → Actions → add **`QODANA_TOKEN`**
4. Re-run CI; expect a long Docker scan when the EAP linter can open the project

Rust image tags on Docker Hub (as of 2026-08): `latest`, `2026.2-eap`, `2026.1-eap`.
CI passes `--image jetbrains/qodana-rust:2026.2-eap` and
`qd.rust.configuration.timeout.minutes=90`. Scans on GHA commonly take
**~1.5 hours** (not stuck — slow project-open + EAP). The scan step uses
`continue-on-error: true` so timeouts do not block merge; **Rust** remains
the required quality gate.

### Disable / soften Qodana

Omit `QODANA_TOKEN`. The Qodana job skips analysis and stays green. Only **rust-ci** is the required merge gate by default.

## Dependency vulnerability scanning

`security.yml` runs `cargo audit` against `Cargo.lock`. It lives in its own
workflow rather than as a job in `ci.yml` for one reason: it carries a daily
`schedule` trigger so advisories published *between* PRs are still caught, and
putting a cron on `ci.yml` would drag the ~1.5 h Qodana scan along with it.

The job is **advisory and never a merge gate**. A fresh RUSTSEC advisory
against a transitive dependency is not a defect in whichever PR happens to be
open when it lands, so the audit step is `continue-on-error` and **rust-ci**
remains the only required check. Three details make that advisory posture
honest rather than merely quiet:

- `set -euo pipefail` runs **before** the `tee`. Without it the pipeline
  reports `tee`'s status and a failing audit is recorded as a pass.
- `--deny warnings` is passed. By default `cargo audit` exits 0 for
  unmaintained / unsound / yanked advisories and fails only on
  vulnerabilities, so the step would report success while printing advisories
  — which is precisely the state this lockfile was in when the job was added.
- the log upload is `if: always()`, and its artifact name carries
  `run_attempt` as well as `run_id` — v4 artifact names are immutable and a
  rerun keeps the same `run_id`, so without it the second attempt fails on a
  name conflict and turns an advisory job red.
- the run summary distinguishes **three** outcomes, not two, keyed on
  cargo-audit's exit code: `0` clean, `1` advisories reported, anything else
  the scan itself failed (database unreachable, `Cargo.lock` unparseable).
  That third state is reported as *unknown*, never as "no advisories" — a
  scheduled run that dies fetching the database has told you nothing, and
  silently reading as clean is the worst thing an advisory job can do.

Reproduce locally:

```bash
cargo install cargo-audit --locked
cargo audit --deny warnings   # same strictness as CI
```

`.github/dependabot.yml` covers the other half of supply-chain hygiene. Every
third-party action here is pinned to a full commit SHA, which never moves on
its own — including past a security fix. Dependabot rewrites the SHA and its
trailing `# vX.Y.Z` comment together, monthly and grouped, so the pins stay
both reproducible and current.

`tempfile` is explicitly ignored there. `Cargo.toml` pins it exactly
(`= "=3.27.0"`), and Dependabot rewrites exact requirements just as it does
ranges — restricting the group to `minor`/`patch` does not exempt them, so the
pin needs a real `ignore` entry rather than a comment asking for one.

## MSRV

`Cargo.toml` declares `rust-version = "1.88"`. Every other job floats on
`stable`, so the **msrv** job is the only thing that makes that declaration
true.

It compares the manifest's `rust_version` against the **`rustc` that is
actually installed**, rather than against a third hard-coded copy of the
version. That matters in one specific direction: raising the job's
`toolchain:` pin while leaving `rust-version` behind would keep a
constant-vs-manifest assertion green while the job quietly stopped exercising
the declared floor.

Two manifest rules follow from supporting a toolchain this old:

- **No multi-line inline tables.** TOML 1.0 forbids a newline inside
  `{ ... }`, and cargo 1.88 rejects one outright with `error: invalid inline
  table`. A dependency that needs several lines gets a
  `[dependencies.<name>]` table section instead (see `sentry`).
- Reproduce a suspected MSRV break locally with
  `cargo +1.88 check --locked --all-targets --all-features`.

Raising the floor is a deliberate change, and there are exactly two places to
change: `rust-version` in `Cargo.toml` and the `toolchain:` pin in the **msrv**
job. Changing one without the other fails the job.

## Local commands (same as CI)

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo run --locked -- --help
cargo run --locked -- quant-plan --help
cargo run --locked -- inventory --help
cargo run --locked -- saaq-readiness --help

# Toolchain floor (same as the msrv job)
cargo +1.88 check --locked --all-targets --all-features
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
- Third-party Actions are pinned to full commit SHAs (checkout, rust-toolchain, rust-cache, install-action, codecov, qodana, upload-artifact), not floating major tags — and Dependabot keeps those pins current
- `Cargo.lock` is audited against the RUSTSEC advisory database on every PR and daily (advisory, not a gate)
- Secret-backed steps skip when secrets are missing
- Fork PRs should not receive repository secrets from GitHub
- Concurrency cancels only PR runs (not in-flight `main` Sentry releases)
