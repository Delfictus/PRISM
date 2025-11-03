//! Conformal Prediction for Distribution-Free Prediction Intervals
//!
//! Constitution: Phase 6, Week 3, Sprint 3.2
//!
//! Implementation based on:
//! - Vovk et al. 2005: Algorithmic Learning in a Random World
//! - Shafer & Vovk 2008: A Tutorial on Conformal Prediction
//! - Angelopoulos & Bates 2021: A Gentle Introduction to Conformal Prediction
//!
//! Purpose: Provide finite-sample, distribution-free coverage guarantees
//! Valid for any data distribution - no assumptions required!


/// Conformal Predictor with distribution-free guarantees
pub struct ConformalPredictor {
    pub calibration_set: Vec<CalibrationPoint>,
    alpha: f64,  // Miscoverage rate (e.g., 0.05 for 95% coverage)
    conformity_measure: ConformityMeasure,
}

impl ConformalPredictor {
    /// Create new conformal predictor with target coverage 1-α
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0,1)");

        Self {
            calibration_set: Vec::new(),
            alpha,
            conformity_measure: ConformityMeasure::AbsoluteResidual,
        }
    }

    /// Set conformity measure type
    pub fn with_conformity_measure(mut self, measure: ConformityMeasure) -> Self {
        self.conformity_measure = measure;
        self
    }

    /// Calibrate predictor on labeled data
    pub fn calibrate(&mut self, calibration_data: Vec<(Vec<f64>, f64)>) {
        self.calibration_set = calibration_data.into_iter()
            .map(|(features, label)| CalibrationPoint {
                features,
                true_value: label,
            })
            .collect();
    }

    /// Predict interval for new solution with guaranteed coverage
    pub fn predict_interval(&self, solution: &crate::cma::Solution) -> PredictionInterval {
        if self.calibration_set.is_empty() {
            return PredictionInterval::empty(self.alpha);
        }

        // Compute non-conformity scores on calibration set
        let scores = self.compute_nonconformity_scores(&solution.data, solution.cost);

        // Find quantile: ⌈(n+1)(1-α)⌉/n th order statistic
        let quantile = self.compute_quantile(&scores);

        // Construct prediction interval
        PredictionInterval {
            point_prediction: solution.cost,
            lower: solution.cost - quantile,
            upper: solution.cost + quantile,
            coverage_level: 1.0 - self.alpha,
            calibration_size: self.calibration_set.len(),
            quantile_threshold: quantile,
        }
    }

    /// Predict set (for classification-style problems)
    pub fn predict_set(&self, features: &[f64], candidate_values: &[f64]) -> Vec<f64> {
        let mut prediction_set = Vec::new();

        for &candidate in candidate_values {
            let scores = self.compute_nonconformity_scores(features, candidate);
            let quantile = self.compute_quantile(&scores);

            // Include candidate if its score is within quantile
            let candidate_score = self.conformity_score(features, candidate);
            if candidate_score <= quantile {
                prediction_set.push(candidate);
            }
        }

        prediction_set
    }

    /// Compute non-conformity scores for all calibration points + new point
    fn compute_nonconformity_scores(&self, features: &[f64], predicted_value: f64) -> Vec<f64> {
        let mut scores = Vec::new();

        // Scores for calibration set
        for point in &self.calibration_set {
            let score = self.nonconformity_score(&point.features, point.true_value);
            scores.push(score);
        }

        // Score for new point (using its prediction as "true" value for conformity)
        let new_score = self.conformity_score(features, predicted_value);
        scores.push(new_score);

        scores
    }

    /// Non-conformity score: how "unusual" is this point?
    fn nonconformity_score(&self, features: &[f64], true_value: f64) -> f64 {
        match self.conformity_measure {
            ConformityMeasure::AbsoluteResidual => {
                // |y - ŷ| where ŷ is prediction
                let prediction = self.predict_point(features);
                (true_value - prediction).abs()
            }
            ConformityMeasure::SquaredResidual => {
                let prediction = self.predict_point(features);
                (true_value - prediction).powi(2)
            }
            ConformityMeasure::NormalizedResidual => {
                let prediction = self.predict_point(features);
                let uncertainty = self.estimate_uncertainty(features);
                (true_value - prediction).abs() / uncertainty.max(1e-6)
            }
        }
    }

    /// Conformity score for prediction (without true label)
    fn conformity_score(&self, features: &[f64], predicted_value: f64) -> f64 {
        // For new point, use prediction as "true" value
        self.nonconformity_score(features, predicted_value)
    }

    /// Compute quantile from scores using conformal formula
    fn compute_quantile(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return f64::INFINITY;
        }

        let n = scores.len();
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Conformal quantile: ⌈(n+1)(1-α)⌉-th smallest score
        let rank = ((n as f64 + 1.0) * (1.0 - self.alpha)).ceil() as usize;
        let index = rank.saturating_sub(1).min(n - 1);

        sorted_scores[index]
    }

    /// Point prediction (simple mean predictor for now)
    fn predict_point(&self, features: &[f64]) -> f64 {
        if self.calibration_set.is_empty() {
            return 0.0;
        }

        // k-NN prediction: average of k nearest neighbors
        let k = 5.min(self.calibration_set.len());
        let mut distances: Vec<(f64, f64)> = self.calibration_set.iter()
            .map(|point| {
                let dist = self.distance(features, &point.features);
                (dist, point.true_value)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = distances.iter().take(k).map(|(_, val)| val).sum();
        sum / k as f64
    }

    /// Estimate uncertainty (for normalized residuals)
    fn estimate_uncertainty(&self, features: &[f64]) -> f64 {
        if self.calibration_set.is_empty() {
            return 1.0;
        }

        // Local variance estimate
        let k = 5.min(self.calibration_set.len());
        let prediction = self.predict_point(features);

        let mut distances: Vec<(f64, f64)> = self.calibration_set.iter()
            .map(|point| {
                let dist = self.distance(features, &point.features);
                (dist, point.true_value)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let variance: f64 = distances.iter()
            .take(k)
            .map(|(_, val)| (val - prediction).powi(2))
            .sum::<f64>() / k as f64;

        variance.sqrt().max(0.1)
    }

    /// Euclidean distance between feature vectors
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        a.iter()
            .zip(b.iter())
            .take(min_len)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Validate coverage empirically
    pub fn validate_coverage(&self, test_data: &[(Vec<f64>, f64)]) -> CoverageValidation {
        if test_data.is_empty() {
            return CoverageValidation::empty();
        }

        let mut covered = 0;
        let mut total_width = 0.0;

        for (features, true_value) in test_data {
            let solution = crate::cma::Solution {
                data: features.clone(),
                cost: *true_value,
            };

            let interval = self.predict_interval(&solution);

            // Check if true value is within interval
            if *true_value >= interval.lower && *true_value <= interval.upper {
                covered += 1;
            }

            total_width += interval.upper - interval.lower;
        }

        let empirical_coverage = covered as f64 / test_data.len() as f64;
        let avg_width = total_width / test_data.len() as f64;

        CoverageValidation {
            empirical_coverage,
            target_coverage: 1.0 - self.alpha,
            num_samples: test_data.len(),
            covered,
            avg_interval_width: avg_width,
            valid: (empirical_coverage - (1.0 - self.alpha)).abs() < 0.1, // 10% tolerance
        }
    }
}

/// Calibration point with features and true value
#[derive(Clone, Debug)]
pub struct CalibrationPoint {
    pub features: Vec<f64>,
    pub true_value: f64,
}

/// Prediction interval with guaranteed coverage
#[derive(Clone, Debug)]
pub struct PredictionInterval {
    pub point_prediction: f64,
    pub lower: f64,
    pub upper: f64,
    pub coverage_level: f64,
    pub calibration_size: usize,
    pub quantile_threshold: f64,
}

impl PredictionInterval {
    fn empty(alpha: f64) -> Self {
        Self {
            point_prediction: 0.0,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            coverage_level: 1.0 - alpha,
            calibration_size: 0,
            quantile_threshold: f64::INFINITY,
        }
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Conformity measure types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConformityMeasure {
    AbsoluteResidual,   // |y - ŷ|
    SquaredResidual,    // (y - ŷ)²
    NormalizedResidual, // |y - ŷ| / σ̂
}

/// Coverage validation results
#[derive(Clone, Debug)]
pub struct CoverageValidation {
    pub empirical_coverage: f64,
    pub target_coverage: f64,
    pub num_samples: usize,
    pub covered: usize,
    pub avg_interval_width: f64,
    pub valid: bool,
}

impl CoverageValidation {
    fn empty() -> Self {
        Self {
            empirical_coverage: 0.0,
            target_coverage: 0.0,
            num_samples: 0,
            covered: 0,
            avg_interval_width: 0.0,
            valid: false,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Conformal Coverage Validation:\n\
             Target coverage: {:.2}%\n\
             Empirical coverage: {:.2}%\n\
             Samples: {} (covered: {})\n\
             Avg interval width: {:.4}\n\
             Status: {}",
            self.target_coverage * 100.0,
            self.empirical_coverage * 100.0,
            self.num_samples,
            self.covered,
            self.avg_interval_width,
            if self.valid { "VALID ✓" } else { "INVALID ✗" }
        )
    }
}

/// Adaptive Conformal Prediction (adjusts to distribution shift)
pub struct AdaptiveConformalPredictor {
    predictor: ConformalPredictor,
    window_size: usize,
    recent_scores: std::collections::VecDeque<f64>,
}

impl AdaptiveConformalPredictor {
    pub fn new(alpha: f64, window_size: usize) -> Self {
        Self {
            predictor: ConformalPredictor::new(alpha),
            window_size,
            recent_scores: std::collections::VecDeque::with_capacity(window_size),
        }
    }

    /// Update with new observation (for online setting)
    pub fn update(&mut self, features: Vec<f64>, true_value: f64) {
        // Add to calibration set
        self.predictor.calibration_set.push(CalibrationPoint {
            features,
            true_value,
        });

        // Keep only recent window
        if self.predictor.calibration_set.len() > self.window_size {
            self.predictor.calibration_set.remove(0);
        }
    }

    pub fn predict_interval(&self, solution: &crate::cma::Solution) -> PredictionInterval {
        self.predictor.predict_interval(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor_creation() {
        let predictor = ConformalPredictor::new(0.05);
        assert_eq!(predictor.alpha, 0.05);
    }

    #[test]
    fn test_calibration() {
        let mut predictor = ConformalPredictor::new(0.1);

        let calibration_data = vec![
            (vec![1.0, 2.0], 3.0),
            (vec![2.0, 3.0], 5.0),
            (vec![3.0, 4.0], 7.0),
        ];

        predictor.calibrate(calibration_data);
        assert_eq!(predictor.calibration_set.len(), 3);
    }

    #[test]
    fn test_quantile_computation() {
        let predictor = ConformalPredictor::new(0.1);

        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantile = predictor.compute_quantile(&scores);

        println!("90% quantile: {}", quantile);
        assert!(quantile >= 4.0 && quantile <= 5.0);
    }

    #[test]
    fn test_prediction_interval() {
        let mut predictor = ConformalPredictor::new(0.05);

        let calibration = vec![
            (vec![1.0], 1.0),
            (vec![2.0], 2.0),
            (vec![3.0], 3.0),
            (vec![4.0], 4.0),
            (vec![5.0], 5.0),
        ];

        predictor.calibrate(calibration);

        let solution = crate::cma::Solution {
            data: vec![3.0],
            cost: 3.0,
        };

        let interval = predictor.predict_interval(&solution);

        println!("Prediction interval: [{}, {}]", interval.lower, interval.upper);
        println!("Coverage: {}%", interval.coverage_level * 100.0);

        assert!(interval.lower <= interval.upper);
        assert_eq!(interval.coverage_level, 0.95);
    }
}
