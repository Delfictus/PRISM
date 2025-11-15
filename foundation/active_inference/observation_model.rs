// Observation Model: p(o | x)
// Constitution: Phase 2, Task 2.1 - Generative Model Architecture
//
// Implements wavefront sensing for adaptive optics:
// o(u,v) = |∫ A(x,y)·exp(i·φ(x,y))·exp(-2πi(ux + vy)) dx dy|²
//
// Linearized for small phases:
// o ≈ o₀ + J·x + ε_obs
//
// Where:
// - J: Jacobian (sensitivity matrix, measurement×state)
// - ε_obs ~ N(0, Σ_obs): photon shot noise
// - σ_photon² = 1/√N_photons

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use super::hierarchical_model::{constants, GaussianBelief};

/// Observation model for wavefront sensing
///
/// Maps hidden states (window phases) to observations (intensity measurements)
#[derive(Debug, Clone)]
pub struct ObservationModel {
    /// Number of measurement points
    pub n_measurements: usize,
    /// Number of state dimensions (window phases)
    pub n_states: usize,
    /// Jacobian matrix J: ∂o/∂x (linearization around small phases)
    pub jacobian: Array2<f64>,
    /// Observation noise covariance (diagonal: photon shot noise)
    pub noise_covariance: Array1<f64>,
    /// Observation noise precision (inverse covariance)
    pub noise_precision: Array1<f64>,
    /// Reference intensity (unperturbed wavefront)
    pub reference_intensity: Array1<f64>,
}

impl ObservationModel {
    /// Create new observation model
    ///
    /// # Arguments
    /// * `n_measurements` - Number of detector pixels
    /// * `n_states` - Number of window phases (900)
    /// * `star_magnitude` - Apparent magnitude of target star
    /// * `integration_time` - Detector integration time (seconds)
    pub fn new(
        n_measurements: usize,
        n_states: usize,
        star_magnitude: f64,
        integration_time: f64,
    ) -> Self {
        // Compute photon shot noise from stellar magnitude
        let photon_count = Self::photons_per_measurement(star_magnitude, integration_time);
        let shot_noise = 1.0 / photon_count.sqrt();

        // Initialize noise covariance (isotropic photon noise)
        let noise_variance = shot_noise * shot_noise;
        let noise_covariance = Array1::from_elem(n_measurements, noise_variance);
        let noise_precision = Array1::from_elem(n_measurements, 1.0 / noise_variance);

        // Jacobian: sensitivity of measurements to phase changes
        // For now, use geometric projection matrix
        // Each measurement samples multiple windows
        let jacobian = Self::compute_jacobian(n_measurements, n_states);

        // Reference intensity (flat wavefront)
        let reference_intensity = Array1::ones(n_measurements);

        Self {
            n_measurements,
            n_states,
            jacobian,
            noise_covariance,
            noise_precision,
            reference_intensity,
        }
    }

    /// Photon count per measurement
    ///
    /// Pogson's formula: m = -2.5 log₁₀(Φ/Φ₀)
    /// → Φ = Φ₀ · 10^(-m/2.5)
    ///
    /// N_photons = Φ · A · Δt · η_QE
    ///
    /// Where:
    /// - Φ: photon flux (photons/s/m²)
    /// - A: collecting area per window (1 m²)
    /// - Δt: integration time (s)
    /// - η_QE: quantum efficiency (~0.9 for CCDs)
    fn photons_per_measurement(magnitude: f64, integration_time: f64) -> f64 {
        // Vega (magnitude 0) flux at 550 nm: ~1000 photons/s/cm²
        let vega_flux = 1e7; // photons/s/m²
        let flux = vega_flux * 10.0_f64.powf(-magnitude / 2.5);

        let area = constants::WINDOW_SIZE * constants::WINDOW_SIZE; // 1 m²
        let quantum_efficiency = 0.9;

        flux * area * integration_time * quantum_efficiency
    }

    /// Compute Jacobian matrix: ∂o/∂x
    ///
    /// Linearization: o ≈ o₀ + J·(x - x₀)
    ///
    /// For phase-only aberrations:
    /// ∂I/∂φ_i ≈ 2·√I₀·Re[exp(i·φ_i)]  (for small φ)
    fn compute_jacobian(n_measurements: usize, n_states: usize) -> Array2<f64> {
        let mut jacobian = Array2::zeros((n_measurements, n_states));

        // Geometric mapping: each measurement samples nearby windows
        // Simple model: each pixel sees 3×3 window neighborhood
        let windows_per_pixel = (n_states as f64 / n_measurements as f64).sqrt();

        for i in 0..n_measurements {
            // Center window for this measurement
            let center_window = (i as f64 * windows_per_pixel) as usize % n_states;

            // Sensitivity to center window
            jacobian[[i, center_window]] = 1.0;

            // Sensitivity to neighbors (reduced)
            if center_window > 0 {
                jacobian[[i, center_window - 1]] = 0.3;
            }
            if center_window + 1 < n_states {
                jacobian[[i, center_window + 1]] = 0.3;
            }
        }

        jacobian
    }

    /// Predict observations from state: o = g(x) + ε
    ///
    /// Linearized model: o = o₀ + J·x + ε
    pub fn predict(&self, state: &Array1<f64>) -> Array1<f64> {
        assert_eq!(state.len(), self.n_states);

        // Linear prediction: o = J·x (assuming x₀ = 0)
        self.jacobian.dot(state) + &self.reference_intensity
    }

    /// Observation likelihood: p(o | x) = N(o | g(x), Σ_obs)
    ///
    /// Log-likelihood: ln p(o|x) = -0.5·(o - ĝ)ᵀ·Π_obs·(o - ĝ) - 0.5·ln|2πΣ|
    pub fn log_likelihood(&self, observation: &Array1<f64>, state: &Array1<f64>) -> f64 {
        let predicted = self.predict(state);
        let residual = observation - &predicted;

        // Mahalanobis distance (using diagonal precision)
        let mahalanobis = (&residual * &residual * &self.noise_precision).sum();

        // Log-determinant term
        let log_det = self.noise_covariance.iter().map(|v| v.ln()).sum::<f64>();
        let n = self.n_measurements as f64;

        -0.5 * (mahalanobis + n * (2.0 * PI).ln() + log_det)
    }

    /// Compute prediction error: ε = Π·(o - ĝ(x))
    ///
    /// Precision-weighted error (for variational updates)
    pub fn prediction_error(&self, observation: &Array1<f64>, state: &Array1<f64>) -> Array1<f64> {
        let predicted = self.predict(state);
        let residual = observation - &predicted;

        &residual * &self.noise_precision
    }

    /// Compute expected observation under belief q(x) = N(μ, Σ)
    ///
    /// E_q[o] = ∫ g(x)·q(x) dx ≈ g(μ)  (for linear g)
    pub fn expected_observation(&self, belief: &GaussianBelief) -> Array1<f64> {
        self.predict(&belief.mean)
    }

    /// Compute observation uncertainty under belief
    ///
    /// Var[o] = J·Σ_x·Jᵀ + Σ_obs
    pub fn observation_variance(&self, belief: &GaussianBelief) -> Array1<f64> {
        // For diagonal state covariance: diag(J·Σ·Jᵀ) = Σ_i J[·,i]² · Σ[i,i]
        let mut variance = Array1::zeros(self.n_measurements);

        for i in 0..self.n_measurements {
            let mut state_contribution = 0.0;
            for j in 0..self.n_states {
                state_contribution += self.jacobian[[i, j]].powi(2) * belief.variance[j];
            }
            variance[i] = state_contribution + self.noise_covariance[i];
        }

        variance
    }

    /// Surprise: -ln p(o | x)
    ///
    /// Measures unexpectedness of observation
    pub fn surprise(&self, observation: &Array1<f64>, state: &Array1<f64>) -> f64 {
        -self.log_likelihood(observation, state)
    }

    /// Update Jacobian from empirical sensitivity analysis
    ///
    /// Useful for online calibration as atmospheric conditions change
    pub fn calibrate_jacobian(&mut self, states: &[Array1<f64>], observations: &[Array1<f64>]) {
        assert_eq!(states.len(), observations.len());
        assert!(!states.is_empty());

        // Least-squares fit: J* = argmin_J Σ_t ||o_t - J·x_t||²
        // Solution: J = (Σ_t o_t·x_tᵀ)·(Σ_t x_t·x_tᵀ)^(-1)

        // For simplicity, use simple correlation-based update
        // Full implementation would use proper least-squares
        for (state, obs) in states.iter().zip(observations.iter()) {
            for i in 0..self.n_measurements {
                for j in 0..self.n_states {
                    // Correlate observation[i] with state[j]
                    self.jacobian[[i, j]] += 0.01 * obs[i] * state[j];
                }
            }
        }

        // Normalize
        for i in 0..self.n_measurements {
            let row_norm = (0..self.n_states)
                .map(|j| self.jacobian[[i, j]].powi(2))
                .sum::<f64>()
                .sqrt();

            if row_norm > 1e-10 {
                for j in 0..self.n_states {
                    self.jacobian[[i, j]] /= row_norm;
                }
            }
        }
    }
}

/// Active measurement selection
///
/// For DARPA Narcissus: which 100 of 900 windows to measure?
#[derive(Debug, Clone)]
pub struct MeasurementPattern {
    /// Which windows to actively measure (indices)
    pub active_windows: Vec<usize>,
    /// Total number of windows available
    pub n_windows: usize,
}

impl MeasurementPattern {
    /// Create new measurement pattern
    pub fn new(active_windows: Vec<usize>, n_windows: usize) -> Self {
        assert!(active_windows.iter().all(|&i| i < n_windows));
        Self {
            active_windows,
            n_windows,
        }
    }

    /// Create uniform sampling pattern (every Nth window)
    pub fn uniform(n_active: usize, n_windows: usize) -> Self {
        let stride = n_windows / n_active;
        let active_windows: Vec<usize> = (0..n_active).map(|i| i * stride).collect();

        Self::new(active_windows, n_windows)
    }

    /// Create random sampling pattern
    pub fn random(n_active: usize, n_windows: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut indices: Vec<usize> = (0..n_windows).collect();
        indices.shuffle(&mut rng);

        let active_windows = indices.into_iter().take(n_active).collect();

        Self::new(active_windows, n_windows)
    }

    /// Create adaptive pattern based on uncertainty
    ///
    /// Measure windows with highest phase uncertainty
    pub fn adaptive(n_active: usize, belief: &GaussianBelief) -> Self {
        let n_windows = belief.variance.len();

        // Sort windows by variance (descending)
        let mut indices: Vec<usize> = (0..n_windows).collect();
        indices.sort_by(|&i, &j| belief.variance[j].partial_cmp(&belief.variance[i]).unwrap());

        // Take top-N most uncertain windows
        let active_windows = indices.into_iter().take(n_active).collect();

        Self::new(active_windows, n_windows)
    }

    /// Extract submatrix for active measurements
    pub fn extract_jacobian(&self, full_jacobian: &Array2<f64>) -> Array2<f64> {
        let n_measurements = full_jacobian.nrows();
        let mut submatrix = Array2::zeros((n_measurements, self.active_windows.len()));

        for (j, &window_idx) in self.active_windows.iter().enumerate() {
            for i in 0..n_measurements {
                submatrix[[i, j]] = full_jacobian[[i, window_idx]];
            }
        }

        submatrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photon_count_scaling() {
        // Magnitude 8 star, 10ms integration
        let n1 = ObservationModel::photons_per_measurement(8.0, 0.01);

        // Magnitude 13 star (100x fainter)
        let n2 = ObservationModel::photons_per_measurement(13.0, 0.01);

        // Should be ~100x fewer photons (5 magnitudes = 100x)
        assert!((n1 / n2 - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_observation_prediction() {
        let model = ObservationModel::new(100, 900, 8.0, 0.01);
        let state = Array1::zeros(900);

        let prediction = model.predict(&state);
        assert_eq!(prediction.len(), 100);

        // Should predict reference intensity for zero phase
        assert!((prediction[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_likelihood_perfect_match() {
        let model = ObservationModel::new(100, 900, 8.0, 0.01);
        let state = Array1::zeros(900);
        let obs = model.predict(&state);

        let log_like = model.log_likelihood(&obs, &state);

        // Perfect match should have high (finite) log-likelihood
        assert!(log_like.is_finite());
        assert!(log_like > -1000.0);
    }

    #[test]
    fn test_surprise_increases_with_error() {
        let model = ObservationModel::new(100, 900, 8.0, 0.01);
        let state = Array1::zeros(900);
        let obs_good = model.predict(&state);
        let obs_bad = &obs_good + 1.0; // Large deviation

        let surprise_good = model.surprise(&obs_good, &state);
        let surprise_bad = model.surprise(&obs_bad, &state);

        assert!(surprise_bad > surprise_good);
    }

    #[test]
    fn test_measurement_pattern_uniform() {
        let pattern = MeasurementPattern::uniform(100, 900);
        assert_eq!(pattern.active_windows.len(), 100);

        // Should be evenly spaced
        let stride = pattern.active_windows[1] - pattern.active_windows[0];
        assert_eq!(stride, 9);
    }

    #[test]
    fn test_measurement_pattern_adaptive() {
        let mut variance = Array1::ones(900);
        variance[42] = 10.0; // High uncertainty window
        variance[100] = 8.0;

        let belief = GaussianBelief::new(Array1::zeros(900), variance);
        let pattern = MeasurementPattern::adaptive(5, &belief);

        // Should include high-uncertainty windows
        assert!(pattern.active_windows.contains(&42));
        assert!(pattern.active_windows.contains(&100));
    }

    #[test]
    fn test_observation_variance_includes_noise() {
        let model = ObservationModel::new(100, 900, 8.0, 0.01);
        let belief = GaussianBelief::isotropic(900, 0.0, 0.1);

        let obs_var = model.observation_variance(&belief);

        // Variance should be at least as large as measurement noise
        for &v in obs_var.iter() {
            assert!(v >= model.noise_covariance[0]);
        }
    }
}
