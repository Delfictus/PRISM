use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use prism_ai::meta::ontology::{AlignmentEngine, OntologyService, OntologyServiceError};
use serde_json;

#[derive(Parser, Debug)]
#[command(
    name = "meta-ontologyctl",
    about = "Meta ontology ledger controller",
    version
)]
struct Cli {
    /// Path to the ontology ledger JSONL file
    #[arg(
        long,
        default_value = "PRISM-AI-UNIFIED-VAULT/meta/ontology_ledger.jsonl"
    )]
    ledger: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Emit ontology snapshot to stdout or file
    Snapshot {
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Inspect alignment for a concept id
    Align {
        #[arg(long)]
        concept: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let service = OntologyService::new(cli.ledger.clone())?.with_alignment(AlignmentEngine::new());

    match cli.command {
        Command::Snapshot { out } => {
            let snapshot = service.snapshot();
            let json = serde_json::to_string_pretty(&snapshot)
                .context("serialising ontology snapshot to JSON")?;
            if let Some(path) = out {
                write_snapshot(&path, &json)?;
                println!(
                    "Ontology snapshot written to {} ({})",
                    path.display(),
                    snapshot
                        .digest
                        .as_ref()
                        .map(|d| d.concept_root.as_str())
                        .unwrap_or("no-digest")
                );
            } else {
                println!("{json}");
            }
        }
        Command::Align { concept } => {
            let result = service.align(&concept).map_err(|err| match err {
                OntologyServiceError::Alignment(msg) => anyhow::anyhow!(msg),
                other => other.into(),
            })?;
            println!(
                "Target: {} | Primary: {} | Candidates: {}",
                result.target,
                result
                    .primary_match
                    .as_ref()
                    .map(|m| format!(
                        "{} (score {:.3}, attr {:.3}, rel {:.3})",
                        m.id, m.score, m.attribute_overlap, m.relation_overlap
                    ))
                    .unwrap_or_else(|| "None".into()),
                result.candidates.len()
            );
            if !result.explainability.is_empty() {
                println!("Explainability: {}", result.explainability);
            }
        }
    }
    Ok(())
}

fn write_snapshot(path: &Path, json: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating snapshot directory {}", parent.display()))?;
    }
    std::fs::write(path, json).with_context(|| format!("writing snapshot {}", path.display()))?;
    Ok(())
}
