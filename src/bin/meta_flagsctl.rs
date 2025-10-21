use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use prism_ai::features::{
    registry, MetaFeatureId, MetaFeatureState, MetaFlagError, MetaFlagManifest,
};

#[derive(Parser, Debug)]
#[command(
    name = "meta-flagsctl",
    about = "Governance-grade controller for PRISM meta feature flags",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Display current meta feature status
    Status {
        /// Emit manifest in JSON format
        #[arg(long)]
        json: bool,
    },

    /// Emit a signed manifest snapshot to stdout or a file
    Snapshot {
        /// Optional output path; defaults to stdout
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Transition a feature into shadow mode
    Shadow {
        #[arg(value_parser = parse_feature)]
        feature: MetaFeatureId,
        #[arg(long)]
        actor: String,
        #[arg(long, value_parser = parse_datetime)]
        planned: Option<DateTime<Utc>>,
        #[arg(long)]
        rationale: String,
        #[arg(long)]
        evidence: Option<String>,
    },

    /// Promote a feature with a gradual rollout
    Gradual {
        #[arg(value_parser = parse_feature)]
        feature: MetaFeatureId,
        #[arg(long)]
        actor: String,
        #[arg(long, value_parser = parse_percentage)]
        current_pct: f32,
        #[arg(long, value_parser = parse_percentage)]
        target_pct: f32,
        #[arg(long, value_parser = parse_datetime)]
        eta: Option<DateTime<Utc>>,
        #[arg(long)]
        rationale: String,
        #[arg(long)]
        evidence: Option<String>,
    },

    /// Enable a feature with full production justification
    Enable {
        #[arg(value_parser = parse_feature)]
        feature: MetaFeatureId,
        #[arg(long)]
        actor: String,
        #[arg(long)]
        justification: String,
        #[arg(long)]
        evidence: Option<String>,
    },

    /// Disable a feature and document the rationale
    Disable {
        #[arg(value_parser = parse_feature)]
        feature: MetaFeatureId,
        #[arg(long)]
        actor: String,
        #[arg(long)]
        rationale: String,
        #[arg(long)]
        evidence: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Status { json } => {
            let manifest = registry().snapshot();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&manifest)
                        .context("serialising manifest to JSON")?
                );
            } else {
                print_table(&manifest);
            }
        }
        Command::Snapshot { out } => {
            let manifest = registry().snapshot();
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?)
                    .with_context(|| format!("writing snapshot {}", path.display()))?;
                println!(
                    "Snapshot written to {} (merkle={})",
                    path.display(),
                    manifest.merkle_root
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&manifest)
                        .context("serialising manifest to JSON")?
                );
            }
        }
        Command::Shadow {
            feature,
            actor,
            planned,
            rationale,
            evidence,
        } => {
            let state = MetaFeatureState::Shadow {
                actor: actor.clone(),
                planned_activation: planned,
            };
            apply_state(feature, state, actor, rationale, evidence)?;
        }
        Command::Gradual {
            feature,
            actor,
            current_pct,
            target_pct,
            eta,
            rationale,
            evidence,
        } => {
            if target_pct < current_pct {
                anyhow::bail!(
                    "target percentage ({target_pct:.2}%) must be >= current percentage ({current_pct:.2}%)"
                );
            }
            let state = MetaFeatureState::Gradual {
                actor: actor.clone(),
                current_pct: current_pct / 100.0,
                target_pct: target_pct / 100.0,
                eta,
            };
            apply_state(feature, state, actor, rationale, evidence)?;
        }
        Command::Enable {
            feature,
            actor,
            justification,
            evidence,
        } => {
            let state = MetaFeatureState::Enabled {
                actor: actor.clone(),
                activated_at: Utc::now(),
                justification: justification.clone(),
            };
            apply_state(feature, state, actor, justification, evidence)?;
        }
        Command::Disable {
            feature,
            actor,
            rationale,
            evidence,
        } => {
            let state = MetaFeatureState::Disabled;
            apply_state(feature, state, actor, rationale, evidence)?;
        }
    }
    Ok(())
}

fn apply_state(
    feature: MetaFeatureId,
    state: MetaFeatureState,
    actor: String,
    rationale: String,
    evidence: Option<String>,
) -> Result<()> {
    let manifest = registry()
        .update_state(feature, state, actor.clone(), rationale.clone(), evidence)
        .map_err(anyhow::Error::from)?;
    println!(
        "✔ Feature {} updated by {} at {} (merkle={})",
        feature,
        actor,
        manifest.generated_at.to_rfc3339(),
        manifest.merkle_root
    );
    Ok(())
}

fn print_table(manifest: &MetaFlagManifest) {
    println!(
        "{:<24} {:<16} {:<24} {:<20} {}",
        "feature", "state", "updated_at", "actor", "rationale"
    );
    println!("{}", "-".repeat(96));
    for record in &manifest.records {
        let state = state_label(&record.state);
        println!(
            "{:<24} {:<16} {:<24} {:<20} {}",
            record.id,
            state,
            record.updated_at.to_rfc3339(),
            record.updated_by,
            record.rationale
        );
    }
    println!("merkle_root: {}", manifest.merkle_root);
}

fn state_label(state: &MetaFeatureState) -> String {
    match state {
        MetaFeatureState::Disabled => "disabled".to_string(),
        MetaFeatureState::Shadow {
            actor,
            planned_activation,
        } => {
            if let Some(ts) = planned_activation {
                format!("shadow:{actor}→{}", ts.to_rfc3339())
            } else {
                format!("shadow:{actor}")
            }
        }
        MetaFeatureState::Gradual {
            actor,
            current_pct,
            target_pct,
            eta,
        } => {
            let pct = format!("{:.1}%→{:.1}%", current_pct * 100.0, target_pct * 100.0);
            if let Some(ts) = eta {
                format!("gradual:{actor}:{pct}@{}", ts.to_rfc3339())
            } else {
                format!("gradual:{actor}:{pct}")
            }
        }
        MetaFeatureState::Enabled {
            actor,
            activated_at,
            ..
        } => format!("enabled:{actor}@{}", activated_at.to_rfc3339()),
    }
}

fn parse_feature(raw: &str) -> Result<MetaFeatureId, String> {
    MetaFeatureId::from_str(raw).map_err(|err| match err {
        MetaFlagError::UnknownFeature(name) => format!(
            "unknown feature '{name}'. Expected one of: {}",
            MetaFeatureId::variants().join(", ")
        ),
        other => other.to_string(),
    })
}

fn parse_datetime(raw: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| format!("invalid RFC3339 timestamp '{raw}': {err}"))
}

fn parse_percentage(raw: &str) -> Result<f32, String> {
    let value: f32 = raw
        .parse()
        .map_err(|err| format!("invalid percentage '{raw}': {err}"))?;
    if !(0.0..=100.0).contains(&value) {
        return Err(format!("percentage '{value}' must be within [0, 100]"));
    }
    Ok(value)
}
