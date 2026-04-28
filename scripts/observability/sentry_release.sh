#!/usr/bin/env bash
set -euo pipefail

repo="xai-dissect"
org="${SENTRY_ORG:-}"
project="${SENTRY_PROJECT_XAI_DISSECT:-}"
environment="${SENTRY_ENVIRONMENT:-local}"
git_sha="${AGENTOS_GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')}"
release="${repo}@${git_sha}"

if [[ -z "${org}" ]]; then
  printf 'SENTRY_ORG is required\n' >&2
  exit 2
fi

if [[ -z "${project}" ]]; then
  printf 'SENTRY_PROJECT_XAI_DISSECT is required\n' >&2
  exit 2
fi

sentry_cmd() {
  sentry-cli --org "${org}" --project "${project}" "$@"
}

if ! sentry_cmd releases info "${release}" >/dev/null 2>&1; then
  sentry_cmd releases new "${release}"
fi

sentry_cmd releases set-commits "${release}" --auto --ignore-missing
sentry_cmd releases finalize "${release}"
sentry_cmd deploys new --release "${release}" -e "${environment}"

printf '%s\n' "${release}"
