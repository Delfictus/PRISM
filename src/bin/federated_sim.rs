//! CLI helper that produces synthetic federated readiness artifacts.
//!
//! This binary can be invoked manually or by automation to generate the Phase
//! M5 governance evidence (simulation summary + ledger anchors). It supports
//! custom scenarios so engineers can model different node topologies while
//! keeping the artifacts deterministic.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use clap::Parser;
use prism_ai::meta::federated::{
    compute_ledger_merkle, sign_summary_digest, verify_signature, verify_summary_signature,
    FederatedInterface, FederationConfig, LedgerEntry, NodeProfile, NodeRole,
};
use serde::Deserialize;
use serde_json::json;

const DEFAULT_OUTPUT_DIR: &str = "PRISM-AI-UNIFIED-VAULT/artifacts/mec/M5";

#[derive(Debug, Parser)]
#[command(
    name = "federated-sim",
    about = "Generate synthetic Phase M5 federation artifacts."
)]
struct Args {
    /// Output directory for federated artifacts.
    #[arg(long, value_name = "PATH")]
    output_dir: Option<PathBuf>,

    /// Scenario configuration (JSON) describing nodes and quorum requirements.
    #[arg(long, value_name = "FILE")]
    scenario: Option<PathBuf>,

    /// Number of epochs to simulate.
    #[arg(long, default_value_t = 3)]
    epochs: u32,

    /// Remove existing simulation/ledger outputs before writing new ones.
    #[arg(long)]
    clean: bool,

    /// Optional label used to namespace outputs (defaults to scenario name).
    #[arg(long, value_name = "LABEL")]
    label: Option<String>,

    /// Verify mode: summary JSON to validate.
    #[arg(long, value_name = "PATH", requires = "verify_ledger", conflicts_with_all = ["scenario", "epochs", "clean", "output_dir", "label"])]
    verify_summary: Option<PathBuf>,

    /// Verify mode: ledger directory containing epoch_XXX.json files.
    #[arg(long, value_name = "PATH", requires = "verify_summary", conflicts_with_all = ["scenario", "epochs", "clean", "output_dir", "label"])]
    verify_ledger: Option<PathBuf>,

    /// Expected label when running in verify mode.
    #[arg(long, value_name = "LABEL", conflicts_with_all = ["scenario", "epochs", "clean", "output_dir", "label"])]
    expect_label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScenarioNode {
    id: String,
    region: String,
    role: String,
    stake: u32,
}

#[derive(Debug, Deserialize)]
struct ScenarioConfig {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    quorum: Option<usize>,
    #[serde(default)]
    max_ledger_drift: Option<u64>,
    #[serde(default)]
    epoch_start: Option<u64>,
    nodes: Vec<ScenarioNode>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(summary) = args.verify_summary.as_ref() {
        let ledger = args
            .verify_ledger
            .as_ref()
            .ok_or_else(|| anyhow!("--verify-ledger is required when --verify-summary is used"))?;
        let expected_label = args.expect_label.as_deref();
        verify_mode(summary, ledger, expected_label)?;
        println!("✅ Federated artifacts verified for {}", summary.display());
        return Ok(());
    }

    let base_dir = args
        .output_dir
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    let sim_dir = base_dir.join("simulations");
    let ledger_dir = base_dir.join("ledger");

    if args.clean {
        remove_dir_contents(&sim_dir)?;
        remove_dir_contents(&ledger_dir)?;
    }

    fs::create_dir_all(&sim_dir)?;
    fs::create_dir_all(&ledger_dir)?;

    let (mut interface, scenario_name) = load_interface(args.scenario.as_deref())?;
    let mut label = args
        .label
        .or_else(|| scenario_name.clone())
        .unwrap_or_else(|| "default".to_string());
    if label == "baseline-triple-site" {
        label = "default".to_string();
    }
    let epochs_to_run = args.epochs.max(1) as usize;

    let mut reports = Vec::with_capacity(epochs_to_run);
    for _ in 0..epochs_to_run {
        reports.push(interface.simulate_epoch());
    }

    let mut epoch_docs = Vec::with_capacity(reports.len());
    let mut merkle_roots = Vec::with_capacity(reports.len());
    for report in &reports {
        let merkle = compute_ledger_merkle(&report.ledger_entries);
        if !verify_signature(&merkle, &report.signature) {
            return Err(anyhow!(
                "Ledger signature verification failed for epoch {}",
                report.epoch
            ));
        }
        merkle_roots.push(merkle.clone());
        epoch_docs.push(json!({
            "epoch": report.epoch,
            "quorum_reached": report.quorum_reached,
            "aggregated_delta": report.aggregated_delta,
            "ledger_merkle": merkle,
            "signature": report.signature,
            "aligned_updates": report
                .aligned_updates
                .iter()
                .map(|update| {
                    json!({
                        "node_id": update.node_id,
                        "ledger_height": update.ledger_height,
                        "delta_score": update.delta_score,
                        "anchor_hash": update.anchor_hash,
                    })
                })
                .collect::<Vec<_>>(),
        }));
    }

    let summary_signature = sign_summary_digest(&merkle_roots);

    let summary = json!({
        "generated_at": Utc::now().to_rfc3339(),
        "scenario": scenario_name,
        "label": label,
        "epoch_count": reports.len(),
        "summary_signature": summary_signature,
        "epochs": epoch_docs,
    });

    let summary_path = if label == "default" {
        sim_dir.join("epoch_summary.json")
    } else {
        sim_dir.join(format!("epoch_summary_{}.json", label))
    };
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    for report in &reports {
        let target_dir = if label == "default" {
            ledger_dir.clone()
        } else {
            ledger_dir.join(&label)
        };
        fs::create_dir_all(&target_dir)?;
        let ledger_path = target_dir.join(format!("epoch_{:03}.json", report.epoch));
        let merkle_root = compute_ledger_merkle(&report.ledger_entries);
        if !verify_signature(&merkle_root, &report.signature) {
            return Err(anyhow!(
                "Ledger signature verification failed for epoch {}",
                report.epoch
            ));
        }
        let ledger_doc = json!({
            "epoch": report.epoch,
            "signature": report.signature,
            "merkle_root": merkle_root,
            "entries": report.ledger_entries.iter().map(|entry| {
                json!({
                    "node_id": entry.node_id,
                    "anchor_hash": entry.anchor_hash,
                })
            }).collect::<Vec<_>>(),
        });
        fs::write(&ledger_path, serde_json::to_string_pretty(&ledger_doc)?)?;
    }

    println!(
        "Wrote {} epochs to {}",
        reports.len(),
        summary_path.display()
    );

    Ok(())
}

fn load_interface(scenario_path: Option<&Path>) -> Result<(FederatedInterface, Option<String>)> {
    if let Some(path) = scenario_path {
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read scenario: {}", path.display()))?;
        let config: ScenarioConfig = serde_json::from_slice(&bytes)
            .with_context(|| format!("Failed to parse scenario JSON: {}", path.display()))?;
        let federation_config = FederationConfig {
            quorum: config.quorum.unwrap_or(2),
            max_ledger_drift: config.max_ledger_drift.unwrap_or(2),
            epoch: config.epoch_start.unwrap_or(1),
        };

        if config.nodes.is_empty() {
            return Err(anyhow!("Scenario must define at least one node"));
        }
        let mut profiles = Vec::with_capacity(config.nodes.len());
        for node in config.nodes {
            let role = parse_role(&node.role).ok_or_else(|| {
                anyhow!(
                    "Invalid node role '{}' (expected 'core' or 'edge')",
                    node.role
                )
            })?;
            profiles.push(NodeProfile::new(node.id, node.region, role, node.stake));
        }

        let interface = FederatedInterface::new(federation_config, profiles);
        Ok((interface, config.name))
    } else {
        Ok((FederatedInterface::placeholder(), None))
    }
}

fn verify_mode(summary_path: &Path, ledger_dir: &Path, expected_label: Option<&str>) -> Result<()> {
    let summary_text = fs::read_to_string(summary_path)
        .with_context(|| format!("Failed to read summary: {}", summary_path.display()))?;
    let summary_data: serde_json::Value = serde_json::from_str(&summary_text)
        .with_context(|| format!("Failed to parse summary JSON: {}", summary_path.display()))?;

    let label = summary_data
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    if let Some(expected) = expected_label {
        if label != expected {
            return Err(anyhow!(
                "Summary label mismatch: expected '{}', found '{}'",
                expected,
                label
            ));
        }
    }

    let epochs = summary_data
        .get("epochs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Summary missing 'epochs' array"))?;

    let mut roots: Vec<String> = Vec::with_capacity(epochs.len());
    for entry in epochs {
        let epoch = entry
            .get("epoch")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("Epoch entry missing 'epoch'"))?;
        let merkle = entry
            .get("ledger_merkle")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Epoch {} missing 'ledger_merkle'", epoch))?;
        let signature = entry
            .get("signature")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Epoch {} missing 'signature'", epoch))?;

        let ledger_path = if label == "default" {
            ledger_dir.join(format!("epoch_{:03}.json", epoch))
        } else {
            ledger_dir
                .join(label)
                .join(format!("epoch_{:03}.json", epoch))
        };

        let ledger_text = fs::read_to_string(&ledger_path)
            .with_context(|| format!("Failed to read ledger: {}", ledger_path.display()))?;
        let ledger: serde_json::Value = serde_json::from_str(&ledger_text)
            .with_context(|| format!("Failed to parse ledger JSON: {}", ledger_path.display()))?;
        let ledger_entries = ledger
            .get("entries")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Ledger {} missing 'entries' array", ledger_path.display()))?;

        let mut entries = Vec::with_capacity(ledger_entries.len());
        for item in ledger_entries {
            let node_id = item
                .get("node_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Ledger entry missing 'node_id'"))?;
            let anchor = item
                .get("anchor_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Ledger entry missing 'anchor_hash'"))?;
            entries.push(LedgerEntry {
                epoch,
                node_id: node_id.to_string(),
                anchor_hash: anchor.to_string(),
            });
        }

        let recomputed = compute_ledger_merkle(&entries);
        if recomputed != merkle {
            return Err(anyhow!(
                "Ledger merkle mismatch for epoch {}: summary {}, computed {}",
                epoch,
                merkle,
                recomputed
            ));
        }

        if !verify_signature(&merkle, signature) {
            return Err(anyhow!("Signature verification failed for epoch {}", epoch));
        }

        roots.push(merkle.to_string());
    }

    let declared_summary_sig = summary_data
        .get("summary_signature")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Summary missing 'summary_signature'"))?;

    if !verify_summary_signature(&roots, declared_summary_sig) {
        return Err(anyhow!("Summary signature verification failed"));
    }

    Ok(())
}

fn parse_role(value: &str) -> Option<NodeRole> {
    match value.to_ascii_lowercase().as_str() {
        "core" | "core_validator" | "validator" => Some(NodeRole::CoreValidator),
        "edge" | "edge_participant" | "participant" => Some(NodeRole::EdgeParticipant),
        _ => None,
    }
}

fn remove_dir_contents(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}
