use anyhow::{Context, Result};
use prism_ai::features::{registry, MetaFeatureId, MetaFeatureState};
use prism_ai::meta::{MetaOrchestrator, ReflexiveSnapshot};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;

fn parse_u64(value: &str) -> Result<u64> {
    let normalized = value.trim_start_matches("0x");
    u64::from_str_radix(normalized, 16)
        .or_else(|_| value.parse::<u64>())
        .with_context(|| format!("invalid u64 value: {value}"))
}

fn parse_usize(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid usize value: {value}"))
}

fn meta_generation_ready() -> bool {
    let snapshot = registry().snapshot();
    if let Some(record) = snapshot
        .records
        .iter()
        .find(|r| r.id == MetaFeatureId::MetaGeneration)
    {
        matches!(
            record.state,
            MetaFeatureState::Shadow { .. }
                | MetaFeatureState::Gradual { .. }
                | MetaFeatureState::Enabled { .. }
        )
    } else {
        false
    }
}

fn main() -> Result<()> {
    let mut controller_seed = 0xDEADBEEFDEADBEEF;
    let mut base_seed = 0xC0FFEE;
    let mut population = 32usize;
    let mut output: Option<PathBuf> = None;
    let mut stdout = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--controller-seed" => {
                let value = args.next().context("--controller-seed requires a value")?;
                controller_seed = parse_u64(&value)?;
            }
            "--seed" => {
                let value = args.next().context("--seed requires a value")?;
                base_seed = parse_u64(&value)?;
            }
            "--population" => {
                let value = args.next().context("--population requires a value")?;
                population = parse_usize(&value)?;
            }
            "--output" => {
                let value = args.next().context("--output requires a value")?;
                output = Some(PathBuf::from(value));
            }
            "--stdout" => stdout = true,
            "--help" | "-h" => {
                println!(
                    "Usage: meta_reflexive_snapshot [--controller-seed HEX] [--seed HEX] \\\n    \
                     [--population N] [--output PATH] [--stdout]"
                );
                return Ok(());
            }
            other => {
                return Err(anyhow::anyhow!("unrecognized argument: {other}"));
            }
        }
    }

    if stdout && output.is_some() {
        return Err(anyhow::anyhow!(
            "Use either --stdout or --output, not both."
        ));
    }

    if !meta_generation_ready() {
        eprintln!("⚠️ meta_generation flag is disabled; reflexive snapshot may diverge from production expectations.");
    }
    let orchestrator = MetaOrchestrator::new(controller_seed)?;
    let outcome = orchestrator.run_generation(base_seed, population)?;
    let snapshot: &ReflexiveSnapshot = &outcome.reflexive;

    let payload = json!({
        "snapshot": snapshot,
        "distribution": outcome.distribution,
        "temperature": outcome.temperature,
    });

    if let Some(path) = &output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    }

    if stdout || output.is_none() {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    }

    Ok(())
}
