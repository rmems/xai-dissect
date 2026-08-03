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
#   AGENTOS_GIT_SHA     (must match runtime when correlating events;
#                       when unset, falls back to `unknown` like git_sha() in
#                       src/observability.rs — not short HEAD)
#
# Note: sentry-cli 2.46+ rejects global `--org` / `--project` before the
# subcommand. Prefer env vars (SENTRY_ORG / SENTRY_PROJECT), which work on
# both 2.x and 3.x.
set -euo pipefail

repo="xai-dissect"
org="${SENTRY_ORG:-}"
project="${SENTRY_PROJECT_XAI_DISSECT:-${SENTRY_PROJECT:-}}"
environment="${SENTRY_ENVIRONMENT:-local}"
# Match runtime: xai-dissect@<AGENTOS_GIT_SHA|unknown> (see observability::git_sha).
git_sha="${AGENTOS_GIT_SHA:-unknown}"
# Trim leading/trailing whitespace to match runtime git_sha_from_value (str::trim).
git_sha="$(printf '%s' "${git_sha}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [[ -z "${git_sha}" ]]; then
  git_sha="unknown"
fi
fi
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

# Preflight auth/org/project. sentry-cli 2.46+/3.x often returns empty stderr on a
# missing release (HTTP 404) with no version string, so we cannot parse "not found"
# text reliably. After list succeeds, a failed `releases info` means the version
# is absent — create it. If list fails, surface that error (token/org/project).
list_err="$(mktemp)"
set +e
sentry-cli releases list >/dev/null 2>"${list_err}"
list_status=$?
set -e
if [[ "${list_status}" -ne 0 ]]; then
  printf 'sentry-cli releases list failed (status=%s); check token/org/project:\n%s\n' \
    "${list_status}" "$(cat "${list_err}" 2>/dev/null || true)" >&2
  rm -f "${list_err}"
  exit "${list_status}"
fi
rm -f "${list_err}"

info_err="$(mktemp)"
set +e
sentry-cli releases info "${release}" >/dev/null 2>"${info_err}"
info_status=$?
set -e
if [[ "${info_status}" -ne 0 ]]; then
  # Auth/project already validated via list; treat info failure as missing release.
  # (sentry-cli may print nothing at default log level on 404.)
  rm -f "${info_err}"
  sentry-cli releases new "${release}"
else
  rm -f "${info_err}"
fi

sentry-cli releases set-commits "${release}" --auto --ignore-missing
sentry-cli releases finalize "${release}"
sentry-cli deploys new --release "${release}" -e "${environment}"

printf '%s\n' "${release}"
