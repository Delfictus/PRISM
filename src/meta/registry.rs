use crate::governance::determinism::{DeterminismProof, MetaDeterminism};
use crate::meta::orchestrator::{EvolutionPlan, VariantEvaluation};
use crate::meta::telemetry::MetaRuntimeMetrics;
use crate::telemetry::{ComponentId, EventLevel, TelemetryEntry};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Duration;
use std::{env, fs};
use thiserror::Error;

const SELECTION_REPORT_PATH: &str = "PRISM-AI-UNIFIED-VAULT/artifacts/mec/M1/selection_report.json";

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionReport {
    pub timestamp: DateTime<Utc>,
    pub plan: PlanSummary,
    pub determinism: DeterminismSummary,
    pub best: VariantSummary,
    pub distribution: DistributionSummary,
    pub telemetry: TelemetrySummary,
    pub latency_ms: f64,
    pub runtime: MetaRuntimeMetrics,
    pub report_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub generated_at: DateTime<Utc>,
    pub base_seed: u64,
    pub population: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismSummary {
    pub master_seed: u64,
    pub input_hash: String,
    pub output_hash: String,
    pub manifest_hash: String,
    pub free_energy_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantSummary {
    pub genome_hash: String,
    pub metrics: MetricsSummary,
    pub parameters: std::collections::BTreeMap<String, String>,
    pub feature_toggles: std::collections::BTreeMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub energy: f64,
    pub chromatic_loss: f64,
    pub divergence: f64,
    pub scalar: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub entropy: f64,
    pub temperature: f64,
    pub top_candidates: Vec<DistributionCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionCandidate {
    pub genome_hash: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub entry_id: String,
    pub component: String,
    pub level: String,
}

pub fn persist_selection_report(
    plan: &EvolutionPlan,
    evaluations: &[VariantEvaluation],
    distribution: &[f64],
    temperature: f64,
    determinism_proof: &DeterminismProof,
    meta: &MetaDeterminism,
    telemetry_entry: &TelemetryEntry,
    best_index: usize,
    elapsed: Duration,
    runtime_metrics: &MetaRuntimeMetrics,
) -> Result<SelectionReport, RegistryError> {
    let report = build_report(
        plan,
        evaluations,
        distribution,
        temperature,
        determinism_proof,
        meta,
        telemetry_entry,
        best_index,
        elapsed,
        runtime_metrics,
    );
    write_report(&report)?;
    Ok(report)
}

fn build_report(
    plan: &EvolutionPlan,
    evaluations: &[VariantEvaluation],
    distribution: &[f64],
    temperature: f64,
    determinism_proof: &DeterminismProof,
    meta: &MetaDeterminism,
    telemetry_entry: &TelemetryEntry,
    best_index: usize,
    elapsed: Duration,
    runtime_metrics: &MetaRuntimeMetrics,
) -> SelectionReport {
    let best = &evaluations[best_index];
    let plan_summary = PlanSummary {
        generated_at: plan.generated_at,
        base_seed: plan.base_seed,
        population: plan.genomes.len(),
    };
    let determinism = DeterminismSummary {
        master_seed: determinism_proof.master_seed,
        input_hash: determinism_proof.input_hash.clone(),
        output_hash: determinism_proof.output_hash.clone(),
        manifest_hash: meta.meta_merkle_root.clone(),
        free_energy_hash: meta.free_energy_hash.clone(),
    };
    let metrics = MetricsSummary {
        energy: best.metrics.energy,
        chromatic_loss: best.metrics.chromatic_loss,
        divergence: best.metrics.divergence,
        scalar: best.metrics.scalar,
    };
    let parameters = best
        .genome
        .parameters
        .iter()
        .map(|(key, value)| (key.clone(), value.to_string()))
        .collect();
    let variant = VariantSummary {
        genome_hash: best.genome.hash.clone(),
        metrics,
        parameters,
        feature_toggles: best.genome.feature_toggles.clone(),
    };

    let distribution = build_distribution_summary(evaluations, distribution, temperature);
    let mut runtime = runtime_metrics.clone();
    if runtime.latency_ms <= 0.0 {
        runtime.latency_ms = elapsed.as_secs_f64() * 1_000.0;
    }
    let telemetry = TelemetrySummary {
        entry_id: telemetry_entry.id.to_string(),
        component: match &telemetry_entry.component {
            ComponentId::Custom(name) => name.clone(),
            ComponentId::Orchestrator => "meta_orchestrator".into(),
            other => format!("{:?}", other),
        },
        level: level_label(&telemetry_entry.level),
    };

    let mut report = SelectionReport {
        timestamp: Utc::now(),
        plan: plan_summary,
        determinism,
        best: variant,
        distribution,
        telemetry,
        latency_ms: runtime.latency_ms,
        runtime,
        report_hash: String::new(),
    };
    report.report_hash = compute_report_hash(&report);
    report
}

fn build_distribution_summary(
    evaluations: &[VariantEvaluation],
    distribution: &[f64],
    temperature: f64,
) -> DistributionSummary {
    let entropy = shannon_entropy(distribution);
    let top_candidates = evaluations
        .iter()
        .zip(distribution.iter())
        .enumerate()
        .map(|(_, (evaluation, weight))| DistributionCandidate {
            genome_hash: evaluation.genome.hash.clone(),
            weight: *weight,
        })
        .collect::<Vec<_>>();

    let mut sorted = top_candidates;
    sorted.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    sorted.truncate(5);

    DistributionSummary {
        entropy,
        temperature,
        top_candidates: sorted,
    }
}

fn compute_report_hash(report: &SelectionReport) -> String {
    let serialized =
        serde_json::to_vec(report).expect("selection report serialization for hashing");
    let mut digest = Sha256::new();
    digest.update(serialized);
    format!("{:x}", digest.finalize())
}

fn write_report(report: &SelectionReport) -> Result<(), RegistryError> {
    let path = report_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(report)?;
    fs::write(path, json)?;
    Ok(())
}

fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>()
}

fn level_label(level: &EventLevel) -> String {
    match level {
        EventLevel::Debug => "debug".into(),
        EventLevel::Info => "info".into(),
        EventLevel::Warning => "warning".into(),
        EventLevel::Error => "error".into(),
    }
}

fn report_path() -> PathBuf {
    if let Ok(custom) = env::var("PRISM_SELECTION_REPORT_PATH") {
        PathBuf::from(custom)
    } else {
        PathBuf::from(SELECTION_REPORT_PATH)
    }
}
