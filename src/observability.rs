use std::borrow::Cow;
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Error;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();
static RUN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Opt-in Sentry for real-weight campaigns.
///
/// Enabled only when **both** are set:
/// - `XAI_DISSECT_SENTRY=1` (or `true` / `yes`)
/// - `SENTRY_DSN` non-empty and parseable
///
/// Default is off so public CLI clones never phone home.
///
/// Init follows the official Rust SDK pattern (`ClientOptions` + keep the
/// guard alive). Release names use `xai-dissect@<git_sha|version>` so they
/// match CI release markers (`scripts/observability/sentry_release.sh`).
/// Invalid DSNs soft-fail (return `None`) instead of panicking — `sentry::init`
/// panics on bad DSNs, which would break the CLI for a misconfigured opt-in.
pub fn init_sentry() -> Option<sentry::ClientInitGuard> {
    if !env_flag_enabled("XAI_DISSECT_SENTRY") {
        return None;
    }
    let dsn_raw = std::env::var("SENTRY_DSN")
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    // Parse before init: invalid DSNs must not panic the process.
    let dsn: sentry::types::Dsn = dsn_raw.trim().parse().ok()?;

    let environment = std::env::var("SENTRY_ENVIRONMENT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "local".to_owned());

    let release = format!(
        "xai-dissect@{}",
        std::env::var("AGENTOS_GIT_SHA")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_owned())
    );

    let guard = sentry::init(sentry::ClientOptions {
        dsn: Some(dsn),
        release: Some(Cow::Owned(release)),
        environment: Some(Cow::Owned(environment)),
        // Error reporting only for opt-in weight runs (no perf transactions).
        traces_sample_rate: 0.0,
        // CLI: never send IPs/headers/PII (docs enable this for HTTP servers only).
        send_default_pii: false,
        before_send: Some(std::sync::Arc::new(|mut event| {
            scrub_event_paths(&mut event);
            Some(event)
        })),
        ..Default::default()
    });

    if !guard.is_enabled() {
        return None;
    }

    sentry::configure_scope(|scope| {
        scope.set_tag("repo", "xai-dissect");
        scope.set_tag("run_id", run_id());
    });

    Some(guard)
}

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref().map(str::trim),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
    )
}

/// Redact home-directory prefixes so checkpoint paths do not leak layout.
fn scrub_event_paths(event: &mut sentry::protocol::Event<'_>) {
    if let Some(home) = std::env::var_os("HOME").and_then(|h| h.into_string().ok()) {
        let home = home.trim_end_matches('/');
        if home.is_empty() {
            return;
        }
        if let Some(message) = event.message.as_mut() {
            *message = message.replace(home, "$HOME");
        }
        for exception in &mut event.exception.values {
            if let Some(value) = exception.value.as_mut() {
                *value = value.replace(home, "$HOME");
            }
        }
    }
}

/// Report a top-level command failure (no-op when Sentry is disabled).
///
/// Uses `capture_anyhow` so Sentry gets the full error chain (not a flat
/// message). Path scrubbing is applied in `before_send` only.
pub fn capture_error(error: &Error) {
    let category = error_category(Some(error));
    sentry::configure_scope(|scope| {
        scope.set_tag("error_category", category);
    });
    sentry::integrations::anyhow::capture_anyhow(error);
}

pub fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let builder = tracing_subscriber::fmt().with_env_filter(filter);

        if std::env::var("AGENTOS_JSON_TRACING").as_deref() == Ok("1") {
            builder.json().init();
        } else {
            builder.init();
        }
    });
}

pub fn run_id() -> String {
    std::env::var("AGENTOS_RUN_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            let pid = std::process::id();
            let counter = RUN_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("xai-dissect-{nanos}-{pid}-{counter}")
        })
}

pub fn git_sha() -> String {
    std::env::var("AGENTOS_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

pub fn error_category(error: Option<&Error>) -> &'static str {
    let Some(error) = error else {
        return "none";
    };
    let message = format!("{error:#}").to_ascii_lowercase();

    if message.contains("not a directory") || message.contains("no shards found") {
        "config_error"
    } else if message.contains("parse")
        || message.contains("shard")
        || message.contains("tensor")
        || message.contains("mmap")
        || message.contains("stat ")
    {
        "checkpoint_io_error"
    } else if message.contains("json")
        || message.contains("markdown")
        || message.contains("manifest")
        || message.contains("write")
        || message.contains("export")
    {
        "artifact_error"
    } else {
        "unknown_error"
    }
}

#[cfg(test)]
mod tests {
    use super::error_category;
    use anyhow::Error;

    fn classify(message: &str) -> &'static str {
        let error = Error::msg(message.to_owned());
        error_category(Some(&error))
    }

    #[test]
    fn categorizes_none_as_none() {
        assert_eq!(error_category(None), "none");
    }

    #[test]
    fn categorizes_config_errors_from_representative_messages() {
        assert_eq!(classify("Not a directory"), "config_error");
        assert_eq!(classify("No shards found in checkpoint"), "config_error");
    }

    #[test]
    fn categorizes_checkpoint_io_errors_from_representative_messages() {
        assert_eq!(
            classify("failed to parse checkpoint header"),
            "checkpoint_io_error"
        );
        assert_eq!(classify("tensor mmap failed"), "checkpoint_io_error");
        assert_eq!(classify("shard stat failed"), "checkpoint_io_error");
    }

    #[test]
    fn categorizes_artifact_errors_from_representative_messages() {
        assert_eq!(classify("json serialization failed"), "artifact_error");
        assert_eq!(classify("failed to write manifest"), "artifact_error");
        assert_eq!(classify("markdown export failed"), "artifact_error");
    }

    #[test]
    fn categorizes_unknown_errors_when_no_keywords_match() {
        assert_eq!(classify("connection reset by peer"), "unknown_error");
    }

    #[test]
    fn preserves_branch_precedence_for_overlapping_messages() {
        assert_eq!(classify("manifest parse failed"), "checkpoint_io_error");
        assert_eq!(classify("no shards found while exporting"), "config_error");
    }

    #[test]
    fn does_not_match_unrelated_words_containing_stat() {
        // Guards the narrowed `"stat "` predicate against false positives
        // such as "statistics" or the `stats` subcommand name.
        assert_eq!(classify("stats subcommand misconfigured"), "unknown_error");
    }
}
