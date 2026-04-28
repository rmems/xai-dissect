use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Error;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

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
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            format!("xai-dissect-{millis}")
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
        assert_eq!(classify("failed to parse checkpoint header"), "checkpoint_io_error");
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
        assert_eq!(
            classify("stats subcommand misconfigured"),
            "unknown_error"
        );
    }
}
