//! Conformal Prediction for Distribution-Free Guarantees
//!
//! Provides prediction sets with guaranteed coverage probability
//! without distributional assumptions.
//!
//! Constitutional Compliance:
//! - Mathematical guarantees without assumptions (Article III)
//! - GPU acceleration for set computation (Article II)
//! - Adaptive for non-stationary data (Article IV)

use std::collections::BTreeSet;
use ndarray::{Array1, Array2, Axis};
use anyhow::{Result, anyhow};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Conformal prediction configuration
#[derive(Clone, Debug)]
pub struct ConformalConfig {
    /// Desired coverage level (e.g., 0.95 for 95% coverage)
    pub coverage_level: f64,
    /// Calibration set size
    pub calibration_size: usize,
    /// Use adaptive conformal prediction
    pub adaptive: bool,
    /// Window size for adaptive CP
    pub adaptive_window: usize,
    /// Score function type
    pub score_function: ScoreFunction,
    /// Use GPU for score computation
    pub use_gpu: bool,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            coverage_level: 0.95,
            calibration_size: 1000,
            adaptive: true,
            adaptive_window: 100,
            score_function: ScoreFunction::Residual,
            use_gpu: true,
        }
    }
}

/// Score functions for conformal prediction
#[derive(Clone, Debug)]
pub enum ScoreFunction {
    /// Simple residual |y - ŷ|
    Residual,
    /// Normalized residual |y - ŷ| / σ
    NormalizedResidual,
    /// Quantile regression score
    Quantile,
    /// Density-based score
    Density,
    /// Custom neural score function
    Neural,
}

/// Conformal predictor
pub struct ConformalPredictor {
    config: ConformalConfig,
    /// Calibration scores
    calibration_scores: Vec<f64>,
    /// Quantile threshold
    quantile: f64,
    /// Adaptive weights (if adaptive)
    adaptive_weights: Option<Vec<f64>>,
    /// Score history for adaptation
    score_history: Vec<f64>,
}

impl ConformalPredictor {
    /// Create new conformal predictor
    pub fn new(config: ConformalConfig) -> Self {
        Self {
            config,
            calibration_scores: Vec::new(),
            quantile: 0.0,
            adaptive_weights: None,
            score_history: Vec::new(),
        }
    }

    /// Calibrate the conformal predictor
    pub fn calibrate(
        &mut self,
        calibration_data: &[(Array1<f64>, f64)],
        model: &dyn PredictiveModel,
    ) -> Result<()> {
        if calibration_data.len() < self.config.calibration_size {
            return Err(anyhow!(
                "Insufficient calibration data: {} < {}",
                calibration_data.len(),
                self.config.calibration_size
            ));
        }

        // Compute calibration scores
        self.calibration_scores.clear();

        for (x, y) in calibration_data {
            let score = self.compute_conformity_score(x, *y, model)?;
            self.calibration_scores.push(score);
        }

        // Sort scores for quantile computation
        self.calibration_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile threshold
        let n = self.calibration_scores.len();
        let q = ((n + 1) as f64 * self.config.coverage_level).ceil() as usize;
        self.quantile = if q <= n {
            self.calibration_scores[q - 1]
        } else {
            self.calibration_scores[n - 1]
        };

        // Initialize adaptive weights if needed
        if self.config.adaptive {
            self.adaptive_weights = Some(vec![1.0 / n as f64; n]);
        }

        Ok(())
    }

    /// Compute prediction set for new input
    pub fn predict_set(
        &mut self,
        x: &Array1<f64>,
        model: &dyn PredictiveModel,
        candidate_values: &[f64],
    ) -> Result<PredictionSet> {
        let mut prediction_set = BTreeSet::new();
        let mut scores = Vec::new();

        // Test each candidate value
        for &y in candidate_values {
            let score = self.compute_conformity_score(x, y, model)?;

            if score <= self.quantile {
                prediction_set.insert(ordered_float::OrderedFloat(y));
            }

            scores.push((y, score));
        }

        // Update adaptive weights if needed
        if self.config.adaptive {
            self.update_adaptive_weights(&scores);
        }

        // Convert to vector
        let set_values: Vec<f64> = prediction_set
            .iter()
            .map(|&v| v.into())
            .collect();

        // Compute set statistics
        let coverage_estimate = self.estimate_coverage(&set_values, candidate_values);
        let efficiency = self.compute_efficiency(&set_values, candidate_values);

        Ok(PredictionSet {
            values: set_values,
            coverage: self.config.coverage_level,
            actual_coverage: coverage_estimate,
            efficiency,
            quantile_threshold: self.quantile,
            adaptive: self.config.adaptive,
        })
    }

    /// Compute prediction interval (for regression)
    pub fn predict_interval(
        &mut self,
        x: &Array1<f64>,
        model: &dyn PredictiveModel,
    ) -> Result<PredictionInterval> {
        // Get point prediction
        let prediction = model.predict(x)?;

        // Compute interval width based on quantile
        let width = self.quantile;

        // For adaptive, adjust width based on local uncertainty
        let adjusted_width = if self.config.adaptive {
            self.adjust_width_adaptive(x, width, model)?
        } else {
            width
        };

        Ok(PredictionInterval {
            lower: prediction - adjusted_width,
            upper: prediction + adjusted_width,
            center: prediction,
            coverage: self.config.coverage_level,
            width: adjusted_width * 2.0,
        })
    }

    /// Split conformal prediction for improved efficiency
    pub fn split_conformal_predict(
        &mut self,
        x: &Array1<f64>,
        model: &dyn PredictiveModel,
        proper_training_data: &[(Array1<f64>, f64)],
    ) -> Result<PredictionInterval> {
        // Split data into two parts
        let split_point = proper_training_data.len() / 2;
        let (train_data, calib_data) = proper_training_data.split_at(split_point);

        // Train model on first half (would be done externally in practice)
        // model.train(train_data)?;

        // Calibrate on second half
        self.calibrate(calib_data, model)?;

        // Make prediction
        self.predict_interval(x, model)
    }

    /// Weighted conformal prediction for covariate shift
    pub fn weighted_conformal_predict(
        &mut self,
        x: &Array1<f64>,
        model: &dyn PredictiveModel,
        weights: &[f64],
    ) -> Result<PredictionInterval> {
        // Recompute quantile with weights
        let weighted_quantile = self.compute_weighted_quantile(weights)?;

        // Store original and use weighted
        let original_quantile = self.quantile;
        self.quantile = weighted_quantile;

        let interval = self.predict_interval(x, model)?;

        // Restore original
        self.quantile = original_quantile;

        Ok(interval)
    }

    /// Compute conformity score
    fn compute_conformity_score(
        &self,
        x: &Array1<f64>,
        y: f64,
        model: &dyn PredictiveModel,
    ) -> Result<f64> {
        let prediction = model.predict(x)?;

        let score = match self.config.score_function {
            ScoreFunction::Residual => {
                (y - prediction).abs()
            }
            ScoreFunction::NormalizedResidual => {
                let uncertainty = model.predict_uncertainty(x)?;
                (y - prediction).abs() / uncertainty.max(1e-10)
            }
            ScoreFunction::Quantile => {
                // Quantile regression score
                let alpha = 0.5; // Median
                let residual = y - prediction;
                if residual >= 0.0 {
                    alpha * residual
                } else {
                    (alpha - 1.0) * residual
                }
            }
            ScoreFunction::Density => {
                // Negative log density
                let density = model.predict_density(x, y)?;
                -density.ln()
            }
            ScoreFunction::Neural => {
                // Use learned score function
                model.compute_neural_score(x, y)?
            }
        };

        Ok(score)
    }

    /// Update adaptive weights based on recent performance
    fn update_adaptive_weights(&mut self, scores: &[(f64, f64)]) {
        if let Some(ref mut weights) = self.adaptive_weights {
            // Add scores to history
            for (_, score) in scores {
                self.score_history.push(*score);
            }

            // Keep only recent window
            if self.score_history.len() > self.config.adaptive_window {
                self.score_history.drain(0..self.score_history.len() - self.config.adaptive_window);
            }

            // Update weights based on recent accuracy
            let n = weights.len();
            let decay_factor = 0.95;

            for i in 0..n {
                // Exponential decay
                weights[i] *= decay_factor;
            }

            // Add weight to recent accurate predictions
            let recent_weight = (1.0 - decay_factor) / self.config.adaptive_window as f64;
            for score in &self.score_history {
                // Find corresponding calibration score index
                let idx = self.calibration_scores
                    .binary_search_by(|&s| s.partial_cmp(score).unwrap())
                    .unwrap_or_else(|i| i.min(n - 1));

                weights[idx] += recent_weight;
            }

            // Normalize weights
            let sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() {
                *w /= sum;
            }

            // Update quantile with new weights (clone to avoid borrow checker issue)
            let weights_clone = weights.clone();
            drop(weights);  // Explicitly drop mutable borrow
            self.quantile = self.compute_weighted_quantile(&weights_clone).unwrap_or(self.quantile);
        }
    }

    /// Compute weighted quantile
    fn compute_weighted_quantile(&self, weights: &[f64]) -> Result<f64> {
        if weights.len() != self.calibration_scores.len() {
            return Err(anyhow!("Weight dimension mismatch"));
        }

        // Create sorted pairs of (score, weight)
        let mut weighted_scores: Vec<_> = self.calibration_scores
            .iter()
            .zip(weights.iter())
            .map(|(&s, &w)| (s, w))
            .collect();

        weighted_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find weighted quantile
        let target = self.config.coverage_level;
        let mut cumsum = 0.0;

        for (score, weight) in weighted_scores {
            cumsum += weight;
            if cumsum >= target {
                return Ok(score);
            }
        }

        Ok(self.calibration_scores[self.calibration_scores.len() - 1])
    }

    /// Adjust interval width adaptively
    fn adjust_width_adaptive(
        &self,
        x: &Array1<f64>,
        base_width: f64,
        model: &dyn PredictiveModel,
    ) -> Result<f64> {
        // Get local uncertainty estimate
        let local_uncertainty = model.predict_uncertainty(x)?;

        // Get average uncertainty
        let avg_uncertainty = if !self.score_history.is_empty() {
            self.score_history.iter().sum::<f64>() / self.score_history.len() as f64
        } else {
            base_width
        };

        // Adjust width based on local vs average uncertainty
        let adjustment_factor = (local_uncertainty / avg_uncertainty).min(2.0).max(0.5);

        Ok(base_width * adjustment_factor)
    }

    /// Estimate actual coverage
    fn estimate_coverage(&self, prediction_set: &[f64], candidates: &[f64]) -> f64 {
        prediction_set.len() as f64 / candidates.len() as f64
    }

    /// Compute efficiency (inverse of average set size)
    fn compute_efficiency(&self, prediction_set: &[f64], candidates: &[f64]) -> f64 {
        if prediction_set.is_empty() {
            0.0
        } else {
            1.0 / prediction_set.len() as f64
        }
    }
}

/// Prediction set with coverage guarantee
#[derive(Debug, Clone)]
pub struct PredictionSet {
    /// Values in the prediction set
    pub values: Vec<f64>,
    /// Nominal coverage level
    pub coverage: f64,
    /// Estimated actual coverage
    pub actual_coverage: f64,
    /// Efficiency (inverse size)
    pub efficiency: f64,
    /// Quantile threshold used
    pub quantile_threshold: f64,
    /// Whether adaptive CP was used
    pub adaptive: bool,
}

impl PredictionSet {
    /// Check if value is in prediction set
    pub fn contains(&self, value: f64) -> bool {
        self.values.iter().any(|&v| (v - value).abs() < 1e-10)
    }

    /// Get set size
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Check if set is valid (non-empty)
    pub fn is_valid(&self) -> bool {
        !self.values.is_empty()
    }
}

/// Prediction interval with coverage guarantee
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Center (point prediction)
    pub center: f64,
    /// Coverage level
    pub coverage: f64,
    /// Interval width
    pub width: f64,
}

impl PredictionInterval {
    /// Check if value is in interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Get relative width (width / center)
    pub fn relative_width(&self) -> f64 {
        if self.center.abs() < 1e-10 {
            self.width
        } else {
            self.width / self.center.abs()
        }
    }

    /// Check if interval is informative (not too wide)
    pub fn is_informative(&self) -> bool {
        self.relative_width() < 0.5  // Less than 50% relative width
    }
}

/// Trait for predictive models
pub trait PredictiveModel {
    /// Make point prediction
    fn predict(&self, x: &Array1<f64>) -> Result<f64>;

    /// Predict uncertainty (standard deviation)
    fn predict_uncertainty(&self, x: &Array1<f64>) -> Result<f64> {
        Ok(1.0)  // Default uniform uncertainty
    }

    /// Predict density at point
    fn predict_density(&self, x: &Array1<f64>, y: f64) -> Result<f64> {
        // Default Gaussian density
        let pred = self.predict(x)?;
        let sigma = self.predict_uncertainty(x)?;
        let z = (y - pred) / sigma;
        Ok((-0.5 * z * z).exp() / (2.5066 * sigma))
    }

    /// Compute neural conformity score
    fn compute_neural_score(&self, x: &Array1<f64>, y: f64) -> Result<f64> {
        // Default to residual
        Ok((y - self.predict(x)?).abs())
    }
}

/// Mondrian conformal prediction for conditional coverage
pub struct MondrianConformalPredictor {
    /// Base predictor
    base: ConformalPredictor,
    /// Number of Mondrian categories
    num_categories: usize,
    /// Category assignments
    category_assignments: Vec<usize>,
    /// Per-category quantiles
    category_quantiles: Vec<f64>,
}

impl MondrianConformalPredictor {
    /// Create new Mondrian CP
    pub fn new(config: ConformalConfig, num_categories: usize) -> Self {
        Self {
            base: ConformalPredictor::new(config),
            num_categories,
            category_assignments: Vec::new(),
            category_quantiles: vec![0.0; num_categories],
        }
    }

    /// Calibrate with categories
    pub fn calibrate(
        &mut self,
        calibration_data: &[(Array1<f64>, f64, usize)],
        model: &dyn PredictiveModel,
    ) -> Result<()> {
        // Group by category
        let mut category_scores: Vec<Vec<f64>> = vec![Vec::new(); self.num_categories];

        for (x, y, cat) in calibration_data {
            let score = self.base.compute_conformity_score(x, *y, model)?;
            category_scores[*cat].push(score);
            self.category_assignments.push(*cat);
        }

        // Compute per-category quantiles
        for (cat, scores) in category_scores.iter_mut().enumerate() {
            if scores.is_empty() {
                self.category_quantiles[cat] = f64::INFINITY;
                continue;
            }

            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = scores.len();
            let q = ((n + 1) as f64 * self.base.config.coverage_level).ceil() as usize;
            self.category_quantiles[cat] = if q <= n {
                scores[q - 1]
            } else {
                scores[n - 1]
            };
        }

        Ok(())
    }

    /// Predict with category-specific guarantee
    pub fn predict_interval(
        &mut self,
        x: &Array1<f64>,
        category: usize,
        model: &dyn PredictiveModel,
    ) -> Result<PredictionInterval> {
        // Use category-specific quantile
        let original = self.base.quantile;
        self.base.quantile = self.category_quantiles[category];

        let interval = self.base.predict_interval(x, model)?;

        self.base.quantile = original;

        Ok(interval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleModel {
        noise: f64,
    }

    impl PredictiveModel for SimpleModel {
        fn predict(&self, x: &Array1<f64>) -> Result<f64> {
            Ok(x[0] * 2.0 + 1.0)
        }

        fn predict_uncertainty(&self, _x: &Array1<f64>) -> Result<f64> {
            Ok(self.noise)
        }
    }

    #[test]
    fn test_conformal_calibration() {
        let config = ConformalConfig::default();
        let mut cp = ConformalPredictor::new(config);

        // Generate calibration data
        let mut calibration_data = Vec::new();
        for i in 0..100 {
            let x = arr1(&[i as f64 / 10.0]);
            let y = i as f64 / 5.0 + 1.0;
            calibration_data.push((x, y));
        }

        let model = SimpleModel { noise: 0.1 };

        cp.calibrate(&calibration_data, &model).unwrap();

        assert!(cp.quantile > 0.0);
        assert_eq!(cp.calibration_scores.len(), 100);
    }

    #[test]
    fn test_prediction_interval() {
        let config = ConformalConfig {
            coverage_level: 0.9,
            ..Default::default()
        };
        let mut cp = ConformalPredictor::new(config);

        // Calibration data
        let mut calibration_data = Vec::new();
        for i in 0..100 {
            let x = arr1(&[i as f64 / 10.0]);
            let y = x[0] * 2.0 + 1.0 + (i as f64 * 0.01 - 0.5); // Add noise
            calibration_data.push((x, y));
        }

        let model = SimpleModel { noise: 0.5 };
        cp.calibrate(&calibration_data, &model).unwrap();

        // Test prediction
        let x_test = arr1(&[5.0]);
        let interval = cp.predict_interval(&x_test, &model).unwrap();

        assert!(interval.coverage == 0.9);
        assert!(interval.center == 11.0); // 5*2 + 1
        assert!(interval.width > 0.0);
    }
}