use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct MetaTelemetryWriter {
    path: Arc<PathBuf>,
}

impl MetaTelemetryWriter {
    pub fn new(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                log::warn!(
                    "Failed to create meta telemetry directory {}: {err}",
                    parent.display()
                );
            }
        }
        Self {
            path: Arc::new(path),
        }
    }

    pub fn default() -> Self {
        let custom = std::env::var("PRISM_META_TELEMETRY_PATH").ok();
        let path = custom
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("telemetry/meta_meta.jsonl"));
        Self::new(path)
    }

    pub fn record_orchestrator(
        &self,
        phase: &str,
        determinism: MetaReplayContext,
        metrics: &MetaRuntimeMetrics,
        payload: Value,
    ) {
        let event = MetaTelemetryEvent {
            timestamp_us: current_timestamp_us(),
            phase: phase.to_string(),
            component: MetaComponent {
                id: "meta_orchestrator".to_string(),
                kind: "service".to_string(),
            },
            event: MetaEvent {
                name: "meta_generation_cycle".to_string(),
                level: "info".to_string(),
                payload,
            },
            determinism,
            metrics: metrics.clone(),
            artifacts: collect_default_artifacts(),
        };

        if let Err(err) = self.append(&event) {
            log::error!("Failed to append meta telemetry entry: {err:?}");
        }
    }

    fn append(&self, event: &MetaTelemetryEvent) -> Result<()> {
        let json = serde_json::to_vec(event)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path.as_ref())
            .with_context(|| {
                format!(
                    "opening meta telemetry log {}",
                    self.path.as_ref().display()
                )
            })?;
        file.write_all(&json)?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTelemetryEvent {
    pub timestamp_us: u128,
    pub phase: String,
    pub component: MetaComponent,
    pub event: MetaEvent,
    pub determinism: MetaReplayContext,
    pub metrics: MetaRuntimeMetrics,
    pub artifacts: Vec<MetaArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaComponent {
    pub id: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaEvent {
    pub name: String,
    pub level: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaReplayContext {
    pub manifest_hash: String,
    pub seed: u64,
    pub replay_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaRuntimeMetrics {
    pub latency_ms: f64,
    pub occupancy: f64,
    pub sm_efficiency: f64,
    pub attempts_per_second: f64,
    pub free_energy: f64,
    pub drift_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaArtifact {
    pub uri: String,
    pub content_type: String,
    pub hash: Option<String>,
}

fn collect_default_artifacts() -> Vec<MetaArtifact> {
    let vault_root = std::env::var("PRISM_VAULT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("PRISM-AI-UNIFIED-VAULT"));

    let candidates = [
        ("artifacts/determinism_manifest.json", "application/json"),
        ("reports/determinism_replay.json", "application/json"),
        ("artifacts/advanced_manifest.json", "application/json"),
    ];

    candidates
        .iter()
        .map(|(rel, content_type)| {
            let path = vault_root.join(rel);
            let hash = hash_file(&path);
            MetaArtifact {
                uri: path.display().to_string(),
                content_type: content_type.to_string(),
                hash,
            }
        })
        .collect()
}

fn hash_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(format!("{:x}", hasher.finalize()))
}

fn current_timestamp_us() -> u128 {
    let micros = Utc::now().timestamp_micros();
    if micros >= 0 {
        micros as u128
    } else {
        0
    }
}
