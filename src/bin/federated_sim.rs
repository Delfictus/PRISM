//! CLI helper that produces synthetic federated readiness artifacts.
//!
//! This binary can be invoked manually or by automation to generate the Phase
//! M5 governance evidence (simulation summary + ledger anchors). It supports
//! custom scenarios so engineers can model different node topologies while
//! keeping the artifacts deterministic.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use clap::Parser;
use prism_ai::meta::federated::{FederatedInterface, FederationConfig, NodeProfile, NodeRole};
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
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
        let merkle = report.signature.clone();
        merkle_roots.push(merkle.clone());
        epoch_docs.push(json!({
            "epoch": report.epoch,
            "quorum_reached": report.quorum_reached,
            "aggregated_delta": report.aggregated_delta,
            "ledger_merkle": merkle,
            "signature": merkle,
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

    let summary_signature = compute_summary_signature(&merkle_roots);

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
        let merkle_root = report.signature.clone();
        let ledger_doc = json!({
            "epoch": report.epoch,
            "signature": merkle_root,
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

fn load_interface(
    scenario_path: Option<&Path>,
) -> Result<(FederatedInterface, Option<String>), Box<dyn Error>> {
    if let Some(path) = scenario_path {
        let bytes = fs::read(path)?;
        let config: ScenarioConfig = serde_json::from_slice(&bytes)?;
        let federation_config = FederationConfig {
            quorum: config.quorum.unwrap_or(2),
            max_ledger_drift: config.max_ledger_drift.unwrap_or(2),
            epoch: config.epoch_start.unwrap_or(1),
        };

        if config.nodes.is_empty() {
            return Err("Scenario must define at least one node".into());
        }
        let mut profiles = Vec::with_capacity(config.nodes.len());
        for node in config.nodes {
            let role = parse_role(&node.role).ok_or_else(|| {
                format!(
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

fn fnv1a64(value: &str) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn compute_summary_signature(roots: &[String]) -> String {
    if roots.is_empty() {
        return format!("{:016x}", fnv1a64("summary-empty"));
    }
    let mut level: Vec<String> = roots
        .iter()
        .map(|r| format!("{:016x}", fnv1a64(r)))
        .collect();
    level.sort();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for chunk in level.chunks(2) {
            let combined = if chunk.len() == 2 {
                format!("{}{}", chunk[0], chunk[1])
            } else {
                format!("{}{}", chunk[0], chunk[0])
            };
            next.push(format!("{:016x}", fnv1a64(&combined)));
        }
        level = next;
    }
    level[0].clone()
}

fn parse_role(value: &str) -> Option<NodeRole> {
    match value.to_ascii_lowercase().as_str() {
        "core" | "core_validator" | "validator" => Some(NodeRole::CoreValidator),
        "edge" | "edge_participant" | "participant" => Some(NodeRole::EdgeParticipant),
        _ => None,
    }
}

fn remove_dir_contents(path: &Path) -> Result<(), Box<dyn Error>> {
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
