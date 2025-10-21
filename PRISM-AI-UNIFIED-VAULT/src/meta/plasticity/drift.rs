//! Semantic drift detection utilities used by Phase M4.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error type emitted when evaluating semantic drift.
#[derive(Debug, thiserror::Error)]
pub enum DriftError {
    #[error("embedding dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("embedding cannot be empty")]
    EmptyVector,

    #[error("embedding magnitude is zero; cannot compute cosine similarity")]
    ZeroMagnitude,
}

/// Drift severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DriftStatus {
    Stable,
    Warning,
    Drifted,
}

impl DriftStatus {
    /// Represent the status as a short string suitable for tables.
    pub fn as_str(&self) -> &'static str {
        match self {
            DriftStatus::Stable => "stable",
            DriftStatus::Warning => "warning",
            DriftStatus::Drifted => "drifted",
        }
    }
}

impl fmt::Display for DriftStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metrics captured for explainability.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DriftMetrics {
    pub cosine_similarity: f32,
    pub magnitude_ratio: f32,
    pub delta_l2: f32,
}

impl DriftMetrics {
    pub fn zero() -> Self {
        Self {
            cosine_similarity: 1.0,
            magnitude_ratio: 1.0,
            delta_l2: 0.0,
        }
    }
}

/// Drift evaluation outcome.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DriftEvaluation {
    pub status: DriftStatus,
    pub metrics: DriftMetrics,
}

/// Semantic drift detector using cosine similarity and magnitude ratio thresholds.
#[derive(Debug, Clone, Copy)]
pub struct SemanticDriftDetector {
    warning_cosine: f32,
    drift_cosine: f32,
    warning_magnitude_ratio: f32,
    drift_magnitude_ratio: f32,
}

impl Default for SemanticDriftDetector {
    fn default() -> Self {
        Self {
            warning_cosine: 0.92,
            drift_cosine: 0.85,
            warning_magnitude_ratio: 0.85,
            drift_magnitude_ratio: 0.70,
        }
    }
}

impl SemanticDriftDetector {
    pub fn new(
        warning_cosine: f32,
        drift_cosine: f32,
        warning_magnitude_ratio: f32,
        drift_magnitude_ratio: f32,
    ) -> Self {
        Self {
            warning_cosine,
            drift_cosine,
            warning_magnitude_ratio,
            drift_magnitude_ratio,
        }
    }

    /// Evaluate drift between baseline and candidate embeddings.
    pub fn evaluate(&self, baseline: &[f32], candidate: &[f32]) -> Result<DriftEvaluation, DriftError> {
        if baseline.len() != candidate.len() {
            return Err(DriftError::DimensionMismatch {
                expected: baseline.len(),
                found: candidate.len(),
            });
        }
        if baseline.is_empty() || candidate.is_empty() {
            return Err(DriftError::EmptyVector);
        }

        let cosine = cosine_similarity(baseline, candidate)?;
        let baseline_norm = l2_norm(baseline)?;
        let candidate_norm = l2_norm(candidate)?;
        let magnitude_ratio = candidate_norm / baseline_norm;
        let delta_l2 = l2_distance(baseline, candidate);

        let status = if cosine < self.drift_cosine || magnitude_ratio < self.drift_magnitude_ratio {
            DriftStatus::Drifted
        } else if cosine < self.warning_cosine || magnitude_ratio < self.warning_magnitude_ratio {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        };

        Ok(DriftEvaluation {
            status,
            metrics: DriftMetrics {
                cosine_similarity: cosine,
                magnitude_ratio,
                delta_l2,
            },
        })
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum()
}

fn l2_norm(vector: &[f32]) -> Result<f32, DriftError> {
    let squared_sum: f32 = vector.iter().map(|v| v * v).sum();
    if squared_sum == 0.0 {
        return Err(DriftError::ZeroMagnitude);
    }
    Ok(squared_sum.sqrt())
}

fn l2_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            let diff = l - r;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32, DriftError> {
    let norm_left = l2_norm(left)?;
    let norm_right = l2_norm(right)?;
    Ok(dot_product(left, right) / (norm_left * norm_right))
}
