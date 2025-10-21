//! Semantic plasticity adapters for Phase M4.
//!
//! The representation adapter maintains concept prototypes derived from ontology
//! embeddings and records adaptation events leveraged by explainability reports.

use std::collections::BTreeMap;
use std::fmt;

use super::drift::{DriftError, DriftEvaluation, DriftMetrics, DriftStatus, SemanticDriftDetector};

/// Default exponential smoothing factor used when updating concept prototypes.
const DEFAULT_ADAPTATION_RATE: f32 = 0.25;
const DEFAULT_HISTORY_CAP: usize = 16;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Maintains ontology-aware representation prototypes.
#[derive(Debug)]
pub struct RepresentationAdapter {
    adapter_id: String,
    embedding_dim: usize,
    alpha: f32,
    prototypes: BTreeMap<String, Vec<f32>>,
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

        for (concept, embedding) in &initial_prototypes {
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
        }

        Ok(Self {
            adapter_id: adapter_id.into(),
            embedding_dim,
            alpha: DEFAULT_ADAPTATION_RATE,
            prototypes: initial_prototypes,
            history: Vec::new(),
            max_history: DEFAULT_HISTORY_CAP,
            adaptation_count: 0,
            drift_detector: SemanticDriftDetector::default(),
        })
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

        let prototype = self
            .prototypes
            .entry(concept_id.to_owned())
            .or_insert_with(|| vec![0.0; self.embedding_dim]);

        let drift = if self.adaptation_count == 0 && prototype.iter().all(|v| *v == 0.0) {
            // Cold start, treat as perfect alignment.
            DriftEvaluation {
                status: DriftStatus::Stable,
                metrics: DriftMetrics {
                    cosine_similarity: 1.0,
                    magnitude_ratio: 1.0,
                    delta_l2: 0.0,
                },
            }
        } else {
            self.drift_detector.evaluate(prototype.as_slice(), embedding)?
        };

        // Update prototype with exponential smoothing.
        for (idx, value) in embedding.iter().enumerate() {
            prototype[idx] = (1.0 - self.alpha) * prototype[idx] + self.alpha * value;
        }

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
        self.prototypes.get(concept_id).map(|vec| vec.as_slice())
    }

    /// Produce a snapshot used for explainability.
    pub fn snapshot(&self) -> RepresentationSnapshot {
        RepresentationSnapshot {
            adapter_id: self.adapter_id.clone(),
            embedding_dim: self.embedding_dim,
            tracked_concepts: self.prototypes.len(),
            mode: self.mode(),
            recent_events: self.history.clone(),
        }
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
