use anyhow::Result;
use chrono::Utc;
use prism_ai::features::{registry, MetaFeatureId, MetaFeatureState};
use prism_ai::meta::MetaOrchestrator;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    std::env::set_var("TELEMETRY_EXPECTED_STAGES", "");

    let manifest = registry().snapshot();
    println!("Meta registry Merkle root: {}", manifest.merkle_root);

    std::env::set_var("TELEMETRY_EXPECTED_STAGES", "meta_variant_emitted");

    let registry = registry();
    registry
        .update_state(
            MetaFeatureId::MetaGeneration,
            MetaFeatureState::Shadow {
                actor: "bootstrap".into(),
                planned_activation: Some(Utc::now()),
            },
            "bootstrap",
            "enable meta_generation for sample run",
            None,
        )
        .ok();

    let orchestrator = MetaOrchestrator::new(0xDEADBEEFDEADBEEF)?;
    let outcome = orchestrator.run_generation(0xC0FFEE_u64, 32)?;

    let artifact_dir = Path::new("PRISM-AI-UNIFIED-VAULT/artifacts/mec/M1");
    fs::create_dir_all(artifact_dir)?;

    fs::write(
        artifact_dir.join("evolution_plan.json"),
        serde_json::to_string_pretty(&outcome.plan)?,
    )?;
    fs::write(
        artifact_dir.join("selection_report.json"),
        serde_json::to_string_pretty(&outcome)?,
    )?;
    outcome
        .determinism_proof
        .write_to_path(artifact_dir.join("determinism_manifest_meta.json"))?;

    println!(
        "Best genome {} with scalar {:.5} (temperature {:.3})",
        outcome.evaluations[outcome.best_index].genome.hash,
        outcome.evaluations[outcome.best_index].metrics.scalar,
        outcome.temperature
    );
    println!("Artifacts written to {:?}", artifact_dir);
    Ok(())
}
