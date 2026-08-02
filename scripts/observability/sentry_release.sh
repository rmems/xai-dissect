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
if [[ -z "${git_sha// }" ]]; then
  git_sha="unknown"
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

# Only create when the release is confirmed missing. Auth/network/project
# errors from `releases info` must surface, not be treated as "not found".
info_err="$(mktemp)"
set +e
sentry-cli releases info "${release}" >/dev/null 2>"${info_err}"
info_status=$?
set -e
if [[ "${info_status}" -ne 0 ]]; then
  info_msg="$(cat "${info_err}" 2>/dev/null || true)"
  rm -f "${info_err}"
  # Only treat a *version-specific* missing-release response as create-eligible.
  # Generic "not found" / bare 404 can mean bad org, project, or token — fail closed.
  # Typical missing-release text includes the version string and "release".
  if printf '%s' "${info_msg}" | grep -qiF "${release}" \
    && printf '%s' "${info_msg}" | grep -qiE 'could not find release|release not found|no such release'; then
    sentry-cli releases new "${release}"
  else
    printf 'sentry-cli releases info failed (status=%s); not creating release:\n%s\n' \
      "${info_status}" "${info_msg}" >&2
    exit "${info_status}"
  fi
else
  rm -f "${info_err}"
fi

sentry-cli releases set-commits "${release}" --auto --ignore-missing
sentry-cli releases finalize "${release}"
sentry-cli deploys new --release "${release}" -e "${environment}"

printf '%s\n' "${release}"
