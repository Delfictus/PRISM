use anyhow::{anyhow, Result};
use prism_ai::meta::{MetaOrchestrator, ReflexiveController};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize)]
struct LatticeReport {
    timestamp: String,
    snapshot: prism_ai::meta::LatticeSnapshot,
    history: Vec<prism_ai::meta::ReflexiveState>,
    distribution: Vec<f64>,
    temperature: f64,
    best_index: usize,
}

fn main() -> Result<()> {
    let mut output = PathBuf::from("PRISM-AI-UNIFIED-VAULT/artifacts/mec/M3/lattice_report.json");
    let mut population = 32usize;

    let args: Vec<String> = env::args().collect();
    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err(anyhow!("--output requires a path"));
                }
                output = PathBuf::from(&args[idx]);
            }
            "--population" => {
                idx += 1;
                if idx >= args.len() {
                    return Err(anyhow!("--population requires a value"));
                }
                population = args[idx].parse()?;
            }
            flag => {
                return Err(anyhow!("unknown flag {flag}"));
            }
        }
        idx += 1;
    }

    let mut orchestrator = MetaOrchestrator::new(0xDEAD_BEEF_DEAD_BEEFu64)?;
    orchestrator.attach_reflexive_controller(ReflexiveController::default());
    let outcome = orchestrator.run_generation(0xC0FF_EE_u64, population)?;

    let snapshot = outcome
        .reflexive_snapshot
        .clone()
        .ok_or_else(|| anyhow!("reflexive snapshot missing"))?;

    if let Some(hash) = &outcome.meta.lattice_hash {
        if hash != &snapshot.hash {
            return Err(anyhow!(
                "lattice hash mismatch between determinism meta and snapshot"
            ));
        }
    }

    let report = LatticeReport {
        timestamp: snapshot.state.timestamp.to_rfc3339(),
        snapshot,
        history: outcome.reflexive_history.clone(),
        distribution: outcome.distribution.clone(),
        temperature: outcome.temperature,
        best_index: outcome.best_index,
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!("Lattice report written to {:?}", output);
    Ok(())
}
