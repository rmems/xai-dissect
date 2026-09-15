// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! SAAQ-readiness Markdown/JSON writers (`saaq-readiness.md`, candidate manifest).

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use crate::schema::{CandidateTensorManifest, SaaqDisposition, SaaqReadinessReport};

use super::common::{fmt_opt_u32, write_pretty_json, write_text};

/// Write the full SAAQ-readiness report as pretty-printed JSON.
pub fn write_saaq_readiness_json(report_doc: &SaaqReadinessReport, out: &Path) -> Result<()> {
    write_pretty_json(report_doc, "serialize saaq-readiness report to json", out)
}

/// Write the candidate manifest as pretty-printed JSON.
pub fn write_candidate_manifest_json(manifest: &CandidateTensorManifest, out: &Path) -> Result<()> {
    write_pretty_json(manifest, "serialize candidate manifest to json", out)
}

/// Render a Markdown summary report for humans from a SAAQ-readiness report.
pub fn render_saaq_readiness_markdown(report_doc: &SaaqReadinessReport) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect SAAQ-readiness report");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", report_doc.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        report_doc.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **shards**: {}", report_doc.shard_count);
    let _ = writeln!(
        md,
        "- **quantization_candidates**: {}",
        report_doc.quantization_candidates.len()
    );
    let _ = writeln!(
        md,
        "- **precision_sensitive_tensors**: {}",
        report_doc.precision_sensitive_tensors.len()
    );
    let _ = writeln!(
        md,
        "- **deferred_tensors**: {}",
        report_doc.deferred_tensors.len()
    );
    let _ = writeln!(
        md,
        "- **routing_critical_tensors**: {}",
        report_doc.routing_critical_tensors.len()
    );
    let _ = writeln!(md, "- **schema_version**: {}", report_doc.schema_version);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Quantization candidates");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Rank | Tensor | Kind | Region | Readiness | Opportunity | Risk | Disposition |"
    );
    let _ = writeln!(
        md,
        "| ---: | ------ | ---- | ------ | --------: | ----------: | ---: | ----------- |"
    );
    for candidate in &report_doc.quantization_candidates {
        let _ = writeln!(
            md,
            "| {} | `{}` | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
            candidate.rank,
            candidate.structural_name,
            candidate.kind_label,
            saaq_region_label(candidate.region_class),
            candidate.readiness_score,
            candidate.opportunity_score,
            candidate.risk_score,
            saaq_disposition_label(candidate.disposition)
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Routing-critical tensors");
    let _ = writeln!(md);
    if report_doc.routing_critical_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Readiness | Risk | Reasons |");
        let _ = writeln!(md, "| ------ | --------: | ---: | ------- |");
        for candidate in &report_doc.routing_critical_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {:.3} | {:.3} | {} |",
                candidate.structural_name,
                candidate.readiness_score,
                candidate.risk_score,
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Precision-sensitive tensors");
    let _ = writeln!(md);
    if report_doc.precision_sensitive_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Risk | Reasons |");
        let _ = writeln!(md, "| ------ | ---: | ------- |");
        for candidate in &report_doc.precision_sensitive_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {:.3} | {} |",
                candidate.structural_name,
                candidate.risk_score,
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Deferred tensors");
    let _ = writeln!(md);
    if report_doc.deferred_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Kind | Disposition | Reasons |");
        let _ = writeln!(md, "| ------ | ---- | ----------- | ------- |");
        for candidate in &report_doc.deferred_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {} | {} | {} |",
                candidate.structural_name,
                candidate.kind_label,
                saaq_disposition_label(candidate.disposition),
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Highest-risk tensors");
    let _ = writeln!(md);
    if report_doc.risky_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Region | Risk | Reasons |");
        let _ = writeln!(md, "| ------ | ------ | ---: | ------- |");
        for candidate in &report_doc.risky_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {} | {:.3} | {} |",
                candidate.structural_name,
                saaq_region_label(candidate.region_class),
                candidate.risk_score,
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Layer readiness");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Label | Block | Routing critical | Candidate targets | Mean readiness | Max risk |"
    );
    let _ = writeln!(
        md,
        "| ----- | ----: | ---------------- | ----------------: | -------------: | -------: |"
    );
    for layer in &report_doc.layer_readiness {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {:.3} | {:.3} |",
            layer.label,
            fmt_opt_u32(layer.block_index),
            if layer.routing_critical { "yes" } else { "no" },
            layer.candidate_target_count,
            layer.mean_readiness_score,
            layer.max_risk_score
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Notes");
    let _ = writeln!(md);
    if report_doc.notes.is_empty() {
        let _ = writeln!(md, "None.");
    } else {
        for note in &report_doc.notes {
            let _ = writeln!(md, "- {}", note);
        }
    }

    md
}

/// Write the SAAQ-readiness Markdown summary to `out`.
pub fn write_saaq_readiness_markdown(report_doc: &SaaqReadinessReport, out: &Path) -> Result<()> {
    let s = render_saaq_readiness_markdown(report_doc);
    write_text(&s, out)
}

fn saaq_region_label(region: crate::schema::SaaqRegionClass) -> &'static str {
    match region {
        crate::schema::SaaqRegionClass::RoutingCritical => "routing_critical",
        crate::schema::SaaqRegionClass::NormalizationSensitive => "normalization_sensitive",
        crate::schema::SaaqRegionClass::AlreadyCompressed => "already_compressed",
        crate::schema::SaaqRegionClass::PotentialCompressionTarget => "potential_target",
        crate::schema::SaaqRegionClass::EmbeddingHeavy => "embedding_heavy",
        crate::schema::SaaqRegionClass::Unknown => "unknown",
    }
}

fn saaq_disposition_label(disposition: SaaqDisposition) -> &'static str {
    match disposition {
        SaaqDisposition::Candidate => "candidate",
        SaaqDisposition::ObserveOnly => "observe_only",
        SaaqDisposition::AvoidForNow => "avoid_for_now",
    }
}
