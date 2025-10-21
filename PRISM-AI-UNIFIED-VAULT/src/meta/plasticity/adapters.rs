//! Semantic plasticity adapters for Phase M4.
//!
//! The representation adapter maintains concept prototypes derived from ontology
//! embeddings and records adaptation events that feed explainability artifacts and
//! governance telemetry.

use crate::meta::ontology::ConceptAnchor;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use super::drift::{DriftError, DriftEvaluation, DriftMetrics, DriftStatus, SemanticDriftDetector};

/// Default exponential smoothing factor used when updating concept prototypes.
pub const DEFAULT_ADAPTATION_RATE: f32 = 0.25;
pub const DEFAULT_HISTORY_CAP: usize = 16;

/// Error type returned by [`RepresentationAdapter`].
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("embedding dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("embedding cannot be empty")]
    EmptyEmbedding,
}

impl From<DriftError> for AdapterError {
    fn from(value: DriftError) -> Self {
        match value {
            DriftError::DimensionMismatch { expected, found } => {
                AdapterError::DimensionMismatch { expected, found }
            }
            DriftError::EmptyVector | DriftError::ZeroMagnitude => AdapterError::EmptyEmbedding,
        }
    }
}

/// Operational state of the adapter. Used to contextualize explainability reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMode {
    ColdStart,
    Warmup,
    Stable,
}

impl fmt::Display for AdapterMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            AdapterMode::ColdStart => "cold_start",
            AdapterMode::Warmup => "warmup",
            AdapterMode::Stable => "stable",
        };
        write!(f, "{label}")
    }
}

/// Summary of a single adaptation event.
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    pub concept_id: String,
    pub drift: DriftEvaluation,
    pub timestamp_ms: u128,
    pub notes: String,
}

impl AdaptationEvent {
    fn new(concept_id: &str, drift: DriftEvaluation, notes: impl Into<String>) -> Self {
        let timestamp_ms = current_epoch_ms();
        Self {
            concept_id: concept_id.to_owned(),
            drift,
            timestamp_ms,
            notes: notes.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ConceptState {
    prototype: Vec<f32>,
    anchor_hash: Option<String>,
    observation_count: usize,
    last_updated_ms: u128,
    drift: DriftEvaluation,
}

impl ConceptState {
    fn new(embedding_dim: usize) -> Self {
        Self {
            prototype: vec![0.0; embedding_dim],
            anchor_hash: None,
            observation_count: 0,
            last_updated_ms: current_epoch_ms(),
            drift: DriftEvaluation {
                status: DriftStatus::Stable,
                metrics: DriftMetrics::zero(),
            },
        }
    }

    fn with_prototype(prototype: Vec<f32>) -> Self {
        let last_updated_ms = current_epoch_ms();
        Self {
            prototype,
            anchor_hash: None,
            observation_count: 0,
            last_updated_ms,
            drift: DriftEvaluation {
                status: DriftStatus::Stable,
                metrics: DriftMetrics::zero(),
            },
        }
    }
}

/// Snapshot of the adapter used by explainability reports.
#[derive(Debug, Clone)]
pub struct RepresentationSnapshot {
    pub adapter_id: String,
    pub embedding_dim: usize,
    pub tracked_concepts: usize,
    pub mode: AdapterMode,
    pub recent_events: Vec<AdaptationEvent>,
}

impl RepresentationSnapshot {
    /// Render the snapshot as Markdown content suitable for the Phase M4 explainability artifact.
    pub fn render_markdown(&self) -> String {
        let mut output = String::new();
        output.push_str("# Semantic Plasticity Explainability Report\n\n");
        output.push_str("## Adapter Overview\n");
        output.push_str(&format!("- Adapter ID: `{}`\n", self.adapter_id));
        output.push_str(&format!("- Embedding dimension: {}\n", self.embedding_dim));
        output.push_str(&format!("- Concepts tracked: {}\n", self.tracked_concepts));
        output.push_str(&format!("- Mode: {}\n", self.mode));
        output.push_str("\n## Recent Adaptation Events\n");

        if self.recent_events.is_empty() {
            output.push_str("No adaptation events recorded yet.\n");
            return output;
        }

        for event in &self.recent_events {
            let DriftMetrics {
                cosine_similarity,
                magnitude_ratio,
                delta_l2,
            } = event.drift.metrics;
            output.push_str(&format!(
                "- `{}` @ {} → status: {}, cosine={:.3}, magnitude_ratio={:.3}, ΔL2={:.3}\n",
                event.concept_id,
                event.timestamp_ms,
                event.drift.status,
                cosine_similarity,
                magnitude_ratio,
                delta_l2
            ));
            if !event.notes.is_empty() {
                output.push_str(&format!("  - notes: {}\n", event.notes));
            }
        }

        output
    }
}

/// Manifest representation for governance artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptManifest {
    pub concept_id: String,
    pub anchor_hash: Option<String>,
    pub observation_count: usize,
    pub last_updated_ms: u128,
    pub drift_status: DriftStatus,
    pub cosine_similarity: f32,
    pub magnitude_ratio: f32,
    pub delta_l2: f32,
    pub prototype: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationManifest {
    pub adapter_id: String,
    pub embedding_dim: usize,
    pub mode: AdapterMode,
    pub concepts: Vec<ConceptManifest>,
}

/// Maintains ontology-aware representation prototypes.
#[derive(Debug)]
pub struct RepresentationAdapter {
    adapter_id: String,
    embedding_dim: usize,
    alpha: f32,
    states: BTreeMap<String, ConceptState>,
    history: Vec<AdaptationEvent>,
    max_history: usize,
    adaptation_count: usize,
    drift_detector: SemanticDriftDetector,
}

impl RepresentationAdapter {
    /// Create a new adapter configured for the provided embedding dimension.
    pub fn new(
        adapter_id: impl Into<String>,
        embedding_dim: usize,
        initial_prototypes: BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, AdapterError> {
        if embedding_dim == 0 {
            return Err(AdapterError::EmptyEmbedding);
        }

        let mut states = BTreeMap::new();
        for (concept, embedding) in initial_prototypes {
            if embedding.len() != embedding_dim {
                return Err(AdapterError::DimensionMismatch {
                    expected: embedding_dim,
                    found: embedding.len(),
                });
            }
            debug_assert!(
                !embedding.iter().all(|v| v.is_nan()),
                "initial embedding for {concept} contains NaN values"
            );
            states.insert(concept, ConceptState::with_prototype(embedding));
        }

        Ok(Self {
            adapter_id: adapter_id.into(),
            embedding_dim,
            alpha: DEFAULT_ADAPTATION_RATE,
            states,
            history: Vec::new(),
            max_history: DEFAULT_HISTORY_CAP,
            adaptation_count: 0,
            drift_detector: SemanticDriftDetector::default(),
        })
    }

    /// Register an ontology anchor so that explainability reports can reference canonical hashes.
    pub fn register_anchor(&mut self, anchor: &ConceptAnchor) {
        let hash = anchor.canonical_fingerprint();
        let state = self
            .states
            .entry(anchor.id.clone())
            .or_insert_with(|| ConceptState::new(self.embedding_dim));
        state.anchor_hash = Some(hash);
    }

    /// Set the exponential smoothing factor.
    pub fn with_adaptation_rate(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Override the history size retained for explainability.
    pub fn with_history_cap(mut self, cap: usize) -> Self {
        self.max_history = cap.max(1);
        self
    }

    /// Adapt a concept embedding and record drift metrics.
    pub fn adapt(&mut self, concept_id: &str, embedding: &[f32]) -> Result<AdaptationEvent, AdapterError> {
        if embedding.is_empty() {
            return Err(AdapterError::EmptyEmbedding);
        }
        if embedding.len() != self.embedding_dim {
            return Err(AdapterError::DimensionMismatch {
                expected: self.embedding_dim,
                found: embedding.len(),
            });
        }

        let state = self
            .states
            .entry(concept_id.to_owned())
            .or_insert_with(|| ConceptState::new(self.embedding_dim));

        let drift = if state.observation_count == 0 {
            DriftEvaluation {
                status: DriftStatus::Stable,
                metrics: DriftMetrics::zero(),
            }
        } else {
            self.drift_detector
                .evaluate(state.prototype.as_slice(), embedding)?
        };

        for (idx, value) in embedding.iter().enumerate() {
            state.prototype[idx] = (1.0 - self.alpha) * state.prototype[idx] + self.alpha * value;
        }
        state.observation_count += 1;
        state.last_updated_ms = current_epoch_ms();
        state.drift = drift;

        self.adaptation_count += 1;

        let notes = match drift.status {
            DriftStatus::Stable => "baseline alignment maintained",
            DriftStatus::Warning => "representation drift approaching threshold",
            DriftStatus::Drifted => "representation drift exceeds tolerance",
        };
        let event = AdaptationEvent::new(concept_id, drift, notes);
        self.history.push(event.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Ok(event)
    }

    /// Fetch the current prototype vector for a concept.
    pub fn prototype(&self, concept_id: &str) -> Option<&[f32]> {
        self.states.get(concept_id).map(|state| state.prototype.as_slice())
    }

    /// Produce a snapshot used for explainability.
    pub fn snapshot(&self) -> RepresentationSnapshot {
        RepresentationSnapshot {
            adapter_id: self.adapter_id.clone(),
            embedding_dim: self.embedding_dim,
            tracked_concepts: self.states.len(),
            mode: self.mode(),
            recent_events: self.history.clone(),
        }
    }

    /// Build a manifest for governance artifacts.
    pub fn manifest(&self) -> RepresentationManifest {
        let concepts = self
            .states
            .iter()
            .map(|(concept_id, state)| {
                let DriftMetrics {
                    cosine_similarity,
                    magnitude_ratio,
                    delta_l2,
                } = state.drift.metrics;
                ConceptManifest {
                    concept_id: concept_id.clone(),
                    anchor_hash: state.anchor_hash.clone(),
                    observation_count: state.observation_count,
                    last_updated_ms: state.last_updated_ms,
                    drift_status: state.drift.status,
                    cosine_similarity,
                    magnitude_ratio,
                    delta_l2,
                    prototype: state.prototype.clone(),
                }
            })
            .collect();

        RepresentationManifest {
            adapter_id: self.adapter_id.clone(),
            embedding_dim: self.embedding_dim,
            mode: self.mode(),
            concepts,
        }
    }

    /// Write the manifest to disk.
    pub fn write_manifest<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let manifest = self.manifest();
        let data = serde_json::to_string_pretty(&manifest)
            .expect("representation manifest should be serializable");
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data.as_bytes())
    }

    fn mode(&self) -> AdapterMode {
        match self.adaptation_count {
            0 | 1 => AdapterMode::ColdStart,
            2..=8 => AdapterMode::Warmup,
            _ => AdapterMode::Stable,
        }
    }
}

fn current_epoch_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}
