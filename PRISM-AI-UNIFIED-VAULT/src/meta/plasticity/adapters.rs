//! Semantic plasticity adapters for Phase M4.
//!
//! This module maintains ontology-aware representation prototypes, tracks drift
//! metrics, and produces governance artifacts (manifest + explainability report)
//! required by the Phase M4 charter.

use crate::meta::ontology::ConceptAnchor;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

use super::drift::{DriftError, DriftEvaluation, DriftMetrics, DriftStatus, SemanticDriftDetector};

/// Default exponential smoothing factor used when updating concept prototypes.
pub const DEFAULT_ADAPTATION_RATE: f32 = 0.25;
/// Maximum number of adaptation events retained for explainability.
pub const DEFAULT_HISTORY_CAP: usize = 16;

/// Error type returned by [`RepresentationAdapter`].
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("embedding dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("embedding cannot be empty")]
    EmptyEmbedding,

    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
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

/// Optional metadata passed alongside an observation.
#[derive(Default, Debug, Clone, Copy)]
pub struct AdaptationMetadata<'a> {
    pub timestamp_ms: Option<u128>,
    pub notes: Option<&'a str>,
    pub source: Option<&'a str>,
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
    fn new(
        concept_id: &str,
        drift: DriftEvaluation,
        timestamp_ms: u128,
        notes: impl Into<String>,
    ) -> Self {
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
        Self {
            prototype,
            anchor_hash: None,
            observation_count: 0,
            last_updated_ms: current_epoch_ms(),
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
    pub concepts: Vec<ConceptManifest>,
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

        output.push_str("\n## Concept Metrics\n");
        if self.concepts.is_empty() {
            output.push_str("No concepts tracked yet.\n");
        } else {
            output.push_str(
                "| Concept | Status | Cosine | Magnitude Ratio | ΔL2 | Observations | Anchor |\n",
            );
            output.push_str(
                "|---------|--------|--------|-----------------|-----|-------------|--------|\n",
            );
            for concept in &self.concepts {
                output.push_str(&format!(
                    "| `{}` | {} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
                    concept.concept_id,
                    concept.drift_status,
                    concept.cosine_similarity,
                    concept.magnitude_ratio,
                    concept.delta_l2,
                    concept.observation_count,
                    concept.anchor_hash.as_deref().unwrap_or("-")
                ));
            }
        }

        output.push_str("\n## Recent Adaptation Events\n");
        if self.recent_events.is_empty() {
            output.push_str("No adaptation events recorded yet.\n");
        } else {
            for event in &self.recent_events {
                output.push_str(&format!(
                    "- `{}` @ {} → status: {}, cosine={:.3}, magnitude_ratio={:.3}, ΔL2={:.3}\n",
                    event.concept_id,
                    event.timestamp_ms,
                    event.drift.status,
                    event.drift.metrics.cosine_similarity,
                    event.drift.metrics.magnitude_ratio,
                    event.drift.metrics.delta_l2
                ));
                if !event.notes.is_empty() {
                    output.push_str(&format!("  - notes: {}\n", event.notes));
                }
            }
        }

        output
    }
}

/// Manifest record per concept.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Manifest written to governance artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

    /// Instantiate the adapter directly from a dataset (adapter id inferred when present).
    pub fn from_dataset(dataset: &RepresentationDataset) -> Result<Self, AdapterError> {
        let adapter_id = dataset
            .adapter_id
            .clone()
            .unwrap_or_else(|| "semantic_plasticity".to_string());
        Self::from_dataset_with_id(adapter_id, dataset)
    }

    /// Instantiate the adapter from a dataset using an explicit adapter id.
    pub fn from_dataset_with_id(
        adapter_id: impl Into<String>,
        dataset: &RepresentationDataset,
    ) -> Result<Self, AdapterError> {
        let prototypes = dataset.initial_prototypes()?;
        let mut adapter = RepresentationAdapter::new(adapter_id, dataset.dimension, prototypes)?;
        adapter.ingest_dataset(dataset)?;
        Ok(adapter)
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

    /// Ingest a dataset of ontology concepts and observations.
    pub fn ingest_dataset(&mut self, dataset: &RepresentationDataset) -> Result<(), AdapterError> {
        if dataset.dimension != self.embedding_dim {
            return Err(AdapterError::DimensionMismatch {
                expected: self.embedding_dim,
                found: dataset.dimension,
            });
        }

        for concept in &dataset.concepts {
            if concept.embedding.len() != self.embedding_dim {
                return Err(AdapterError::DimensionMismatch {
                    expected: self.embedding_dim,
                    found: concept.embedding.len(),
                });
            }

            let anchor = ConceptAnchor {
                id: concept.id.clone(),
                description: concept.description.clone(),
                attributes: concept.attributes.clone(),
                related: concept
                    .related
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<String>>(),
            };
            self.register_anchor(&anchor);

            self.states
                .entry(concept.id.clone())
                .and_modify(|state| {
                    if state.observation_count == 0 {
                        state.prototype = concept.embedding.clone();
                    }
                })
                .or_insert_with(|| ConceptState::with_prototype(concept.embedding.clone()));
        }

        let mut observations = dataset.observations.clone();
        observations.sort_by_key(|obs| obs.timestamp_ms.unwrap_or(0));
        for observation in observations {
            let metadata = AdaptationMetadata {
                timestamp_ms: observation.timestamp_ms,
                notes: observation.notes.as_deref(),
                source: observation.source.as_deref(),
            };
            self.adapt_with_metadata(&observation.concept_id, &observation.embedding, metadata)?;
        }

        Ok(())
    }

    /// Adapt a concept embedding and record drift metrics.
    pub fn adapt(
        &mut self,
        concept_id: &str,
        embedding: &[f32],
    ) -> Result<AdaptationEvent, AdapterError> {
        self.adapt_with_metadata(concept_id, embedding, AdaptationMetadata::default())
    }

    /// Adapt a concept embedding using explicit metadata.
    pub fn adapt_with_metadata(
        &mut self,
        concept_id: &str,
        embedding: &[f32],
        metadata: AdaptationMetadata<'_>,
    ) -> Result<AdaptationEvent, AdapterError> {
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

        let timestamp_ms = metadata.timestamp_ms.unwrap_or_else(current_epoch_ms);

        for (idx, value) in embedding.iter().enumerate() {
            state.prototype[idx] = (1.0 - self.alpha) * state.prototype[idx] + self.alpha * value;
        }
        state.observation_count += 1;
        state.last_updated_ms = timestamp_ms;
        state.drift = drift;

        self.adaptation_count += 1;

        let mut base_notes = match drift.status {
            DriftStatus::Stable => "baseline alignment maintained".to_string(),
            DriftStatus::Warning => "representation drift approaching threshold".to_string(),
            DriftStatus::Drifted => "representation drift exceeds tolerance".to_string(),
        };
        if let Some(extra) = metadata.notes {
            if !extra.is_empty() {
                if !base_notes.is_empty() {
                    base_notes.push_str(" | ");
                }
                base_notes.push_str(extra);
            }
        }
        if let Some(source) = metadata.source {
            let suffix = format!("source={}", source);
            if !base_notes.is_empty() {
                base_notes.push_str(" | ");
            }
            base_notes.push_str(&suffix);
        }

        let event = AdaptationEvent::new(concept_id, drift, timestamp_ms, base_notes);
        self.history.push(event.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Ok(event)
    }

    /// Fetch the current prototype vector for a concept.
    pub fn prototype(&self, concept_id: &str) -> Option<&[f32]> {
        self.states
            .get(concept_id)
            .map(|state| state.prototype.as_slice())
    }

    /// Produce a snapshot used for explainability.
    pub fn snapshot(&self) -> RepresentationSnapshot {
        let manifest = self.manifest();
        RepresentationSnapshot {
            adapter_id: manifest.adapter_id.clone(),
            embedding_dim: manifest.embedding_dim,
            tracked_concepts: manifest.concepts.len(),
            mode: manifest.mode,
            concepts: manifest.concepts.clone(),
            recent_events: self.history.clone(),
        }
    }

    /// Build a manifest for governance artifacts.
    pub fn manifest(&self) -> RepresentationManifest {
        let mut concepts: Vec<ConceptManifest> = self
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
        concepts.sort_by(|a, b| a.concept_id.cmp(&b.concept_id));

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

/// Errors surfaced while loading representation datasets.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid dataset: {0}")]
    Invalid(String),
}

/// Serializable dataset describing ontology concepts and observations.
#[derive(Debug, Clone, Deserialize)]
pub struct RepresentationDataset {
    #[serde(default)]
    pub adapter_id: Option<String>,
    pub dimension: usize,
    pub concepts: Vec<DatasetConcept>,
    #[serde(default)]
    pub observations: Vec<ConceptObservation>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConcept {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub related: Vec<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConceptObservation {
    pub concept_id: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub timestamp_ms: Option<u128>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl RepresentationDataset {
    /// Load dataset from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let data = fs::read_to_string(path)?;
        let dataset: Self = serde_json::from_str(&data)?;
        dataset.validate()?;
        Ok(dataset)
    }

    fn validate(&self) -> Result<(), DatasetError> {
        if self.dimension == 0 {
            return Err(DatasetError::Invalid(
                "embedding dimension must be greater than zero".into(),
            ));
        }
        if self.concepts.is_empty() {
            return Err(DatasetError::Invalid("dataset contains no concepts".into()));
        }
        for concept in &self.concepts {
            if concept.embedding.len() != self.dimension {
                return Err(DatasetError::Invalid(format!(
                    "concept {} embedding dimension mismatch (expected {}, found {})",
                    concept.id,
                    self.dimension,
                    concept.embedding.len()
                )));
            }
        }
        for observation in &self.observations {
            if observation.embedding.len() != self.dimension {
                return Err(DatasetError::Invalid(format!(
                    "observation for {} dimension mismatch",
                    observation.concept_id
                )));
            }
        }
        Ok(())
    }

    fn initial_prototypes(&self) -> Result<BTreeMap<String, Vec<f32>>, DatasetError> {
        let mut prototypes = BTreeMap::new();
        for concept in &self.concepts {
            prototypes.insert(concept.id.clone(), concept.embedding.clone());
        }
        Ok(prototypes)
    }
}

fn current_epoch_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}
