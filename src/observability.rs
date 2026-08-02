use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Error;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();
static RUN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Single process run id shared by Sentry tags and tracing spans.
static CACHED_RUN_ID: OnceLock<String> = OnceLock::new();

/// Opt-in Sentry for real-weight campaigns.
///
/// Enabled only when **both** are set:
/// - `XAI_DISSECT_SENTRY=1` (or `true` / `yes` / case-insensitive)
/// - `SENTRY_DSN` non-empty and parseable
///
/// Default is off so public CLI clones never phone home.
///
/// Init follows the official Rust SDK pattern (`ClientOptions` + keep the
/// guard alive). Release names prefer `xai-dissect@<AGENTOS_GIT_SHA>` so they
/// match CI release markers (`scripts/observability/sentry_release.sh`); when
/// no SHA is set, falls back to `unknown` (not crate version) so events are not
/// mis-attributed to a deploy that was never created.
/// Invalid DSNs soft-fail (return `None`) instead of panicking — `sentry::init`
/// panics on bad DSNs, which would break the CLI for a misconfigured opt-in.
///
/// # What leaves the machine (when enabled)
///
/// `capture_error` / `capture_anyhow` send the **full application error chain**.
/// `before_send` only redacts `$HOME` prefixes from messages, exception values,
/// stack frame paths, and breadcrumbs — it does **not** strip arbitrary error
/// text. `send_default_pii(false)` only disables SDK default PII (IPs/headers
/// via HTTP integrations), not app-provided error content.
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

    // Align with scripts/observability/sentry_release.sh: same git_sha fallback.
    let release = format!("xai-dissect@{}", git_sha());

    // sentry 0.49+: ClientOptions is non_exhaustive — use builder methods.
    // Default traces strategy is Disabled (errors only; no performance product).
    // Soft-parsed Dsn is assigned after builders (builder `.dsn(&str)` panics on bad input).
    let mut options = sentry::ClientOptions::new()
        .release(Cow::Owned(release))
        .environment(Cow::Owned(environment))
        // CLI: never send IPs/headers/PII (HTTP-server docs enable true).
        .send_default_pii(false)
        // Avoid advertising hostname when contexts are compiled in.
        .server_name("xai-dissect")
        .before_send(|mut event| {
            // Path scrub only; error chain text from capture_anyhow still ships.
            scrub_event_paths(&mut event);
            Some(event)
        });
    options.dsn = Some(dsn);
    let guard = sentry::init(options);

    if !guard.is_enabled() {
        return None;
    }

    // Same cached run_id as tracing (OnceLock); set before main continues.
    sentry::configure_scope(|scope| {
        scope.set_tag("repo", "xai-dissect");
        scope.set_tag("run_id", run_id());
    });

    Some(guard)
}

fn env_flag_enabled(name: &str) -> bool {
    let Ok(raw) = std::env::var(name) else {
        return false;
    };
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn scrub_str(value: &str, home: &str) -> String {
    value.replace(home, "$HOME")
}

fn scrub_optional(value: &mut Option<String>, home: &str) {
    if let Some(text) = value.as_mut() {
        *text = scrub_str(text, home);
    }
}

fn scrub_stacktrace(stacktrace: &mut sentry::protocol::Stacktrace, home: &str) {
    for frame in &mut stacktrace.frames {
        scrub_optional(&mut frame.abs_path, home);
        scrub_optional(&mut frame.filename, home);
    }
}

/// Redact home-directory prefixes from messages, exceptions, frames, breadcrumbs.
fn scrub_event_paths(event: &mut sentry::protocol::Event<'_>) {
    let Some(home) = std::env::var_os("HOME").and_then(|h| h.into_string().ok()) else {
        return;
    };
    let home = home.trim_end_matches('/');
    if home.is_empty() {
        return;
    }

    scrub_optional(&mut event.message, home);
    for exception in &mut event.exception.values {
        scrub_optional(&mut exception.value, home);
        if let Some(stacktrace) = exception.stacktrace.as_mut() {
            scrub_stacktrace(stacktrace, home);
        }
    }
    if let Some(stacktrace) = event.stacktrace.as_mut() {
        scrub_stacktrace(stacktrace, home);
    }
    for breadcrumb in &mut event.breadcrumbs.values {
        scrub_optional(&mut breadcrumb.message, home);
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

/// Process-stable run id for tracing + Sentry correlation.
pub fn run_id() -> String {
    CACHED_RUN_ID
        .get_or_init(|| {
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
        })
        .clone()
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
    use super::{
        env_flag_enabled, error_category, git_sha, init_sentry, scrub_event_paths, scrub_str,
    };
    use anyhow::Error;
    use sentry::protocol::{Event, Exception, LogEntry, Values};
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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

    #[test]
    fn scrub_str_replaces_home_prefix() {
        assert_eq!(
            scrub_str("/home/dev/ckpt/tensor00000_000", "/home/dev"),
            "$HOME/ckpt/tensor00000_000"
        );
        assert_eq!(scrub_str("no home here", "/home/dev"), "no home here");
    }

    #[test]
    fn scrub_event_paths_redacts_message_and_exception() {
        let _lock = env_lock();
        // SAFETY: serialized by env_lock; restored below.
        unsafe {
            std::env::set_var("HOME", "/home/scrub-test-user");
        }
        let mut event = Event {
            message: Some("/home/scrub-test-user/weights/ckpt".into()),
            exception: Values {
                values: vec![Exception {
                    value: Some("stat /home/scrub-test-user/missing".into()),
                    ..Default::default()
                }],
            },
            logentry: Some(LogEntry {
                message: "ok".into(),
                params: vec![],
            }),
            ..Default::default()
        };
        scrub_event_paths(&mut event);
        assert_eq!(event.message.as_deref(), Some("$HOME/weights/ckpt"));
        assert_eq!(
            event.exception.values[0].value.as_deref(),
            Some("stat $HOME/missing")
        );
        unsafe {
            std::env::remove_var("HOME");
        }
    }

    #[test]
    fn env_flag_accepts_common_truthy_values() {
        let _lock = env_lock();
        let key = "XAI_DISSECT_TEST_FLAG_TRUTHY";
        for truthy in ["1", "true", "YES", "On", " true "] {
            unsafe {
                std::env::set_var(key, truthy);
            }
            assert!(env_flag_enabled(key), "expected truthy for {truthy:?}");
        }
        for falsy in ["0", "false", "no", ""] {
            unsafe {
                std::env::set_var(key, falsy);
            }
            assert!(!env_flag_enabled(key), "expected false for {falsy:?}");
        }
        unsafe {
            std::env::remove_var(key);
        }
        assert!(!env_flag_enabled(key));
    }

    #[test]
    fn git_sha_prefers_env_then_unknown() {
        let _lock = env_lock();
        unsafe {
            std::env::set_var("AGENTOS_GIT_SHA", "abc1234");
        }
        assert_eq!(git_sha(), "abc1234");
        unsafe {
            std::env::remove_var("AGENTOS_GIT_SHA");
        }
        assert_eq!(git_sha(), "unknown");
        unsafe {
            std::env::set_var("AGENTOS_GIT_SHA", "   ");
        }
        assert_eq!(git_sha(), "unknown");
        unsafe {
            std::env::remove_var("AGENTOS_GIT_SHA");
        }
    }

    #[test]
    fn init_sentry_off_when_flag_unset() {
        let _lock = env_lock();
        unsafe {
            std::env::remove_var("XAI_DISSECT_SENTRY");
            std::env::set_var("SENTRY_DSN", "https://public@o0.ingest.sentry.io/0");
        }
        assert!(init_sentry().is_none());
        unsafe {
            std::env::remove_var("SENTRY_DSN");
        }
    }

    #[test]
    fn init_sentry_off_when_dsn_invalid() {
        let _lock = env_lock();
        unsafe {
            std::env::set_var("XAI_DISSECT_SENTRY", "1");
            std::env::set_var("SENTRY_DSN", "not-a-dsn");
        }
        assert!(init_sentry().is_none());
        unsafe {
            std::env::remove_var("XAI_DISSECT_SENTRY");
            std::env::remove_var("SENTRY_DSN");
        }
    }

    #[test]
    fn init_sentry_off_when_dsn_empty() {
        let _lock = env_lock();
        unsafe {
            std::env::set_var("XAI_DISSECT_SENTRY", "1");
            std::env::set_var("SENTRY_DSN", "   ");
        }
        assert!(init_sentry().is_none());
        unsafe {
            std::env::remove_var("XAI_DISSECT_SENTRY");
            std::env::remove_var("SENTRY_DSN");
        }
    }
}
