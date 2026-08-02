#!/usr/bin/env bash
# Create/finalize a Sentry release + deploy for xai-dissect (main CI only).
#
# Required env:
#   SENTRY_AUTH_TOKEN
#   SENTRY_ORG
#   SENTRY_PROJECT_XAI_DISSECT  (or SENTRY_PROJECT)
#
# Optional:
#   SENTRY_ENVIRONMENT  (default: local)
#   AGENTOS_GIT_SHA     (default: short git HEAD)
#
# Note: sentry-cli 2.46+ rejects global `--org` / `--project` before the
# subcommand. Prefer env vars (SENTRY_ORG / SENTRY_PROJECT), which work on
# both 2.x and 3.x.
set -euo pipefail

repo="xai-dissect"
org="${SENTRY_ORG:-}"
project="${SENTRY_PROJECT_XAI_DISSECT:-${SENTRY_PROJECT:-}}"
environment="${SENTRY_ENVIRONMENT:-local}"
git_sha="${AGENTOS_GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')}"
release="${repo}@${git_sha}"

if [[ -z "${SENTRY_AUTH_TOKEN:-}" ]]; then
  printf 'SENTRY_AUTH_TOKEN is required\n' >&2
  exit 2
fi

if [[ -z "${org}" ]]; then
  printf 'SENTRY_ORG is required\n' >&2
  exit 2
fi

if [[ -z "${project}" ]]; then
  printf 'SENTRY_PROJECT_XAI_DISSECT (or SENTRY_PROJECT) is required\n' >&2
  exit 2
fi

export SENTRY_ORG="${org}"
export SENTRY_PROJECT="${project}"

if ! sentry-cli releases info "${release}" >/dev/null 2>&1; then
  sentry-cli releases new "${release}"
fi

sentry-cli releases set-commits "${release}" --auto --ignore-missing
sentry-cli releases finalize "${release}"
sentry-cli deploys new --release "${release}" -e "${environment}"

printf '%s\n' "${release}"
