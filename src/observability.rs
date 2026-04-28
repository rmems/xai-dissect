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
        || message.contains("stat")
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
