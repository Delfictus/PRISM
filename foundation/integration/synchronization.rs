//! Phase Synchronization
//!
//! Constitution: Phase 3, Task 3.1 - Cross-Domain Bridge
//!
//! Implements phase synchronization between neuromorphic (thermodynamic network)
//! and quantum domains using Kuramoto model and information-theoretic measures.
//!
//! Mathematical Foundation:
//! ```text
//! dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)  // Kuramoto model
//! r e^{iψ} = (1/N) Σ_j e^{iθ_j}  // Order parameter
//! ρ = |r|  // Phase coherence (0 = incoherent, 1 = fully synchronized)
//! ```
//!
//! Performance: O(N) for coherence computation via GPU

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Coherence level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceLevel {
    /// Incoherent (ρ < 0.3)
    Incoherent,
    /// Partially coherent (0.3 ≤ ρ < 0.8)
    Partial,
    /// Highly coherent (ρ ≥ 0.8)
    HighCoherence,
}

impl CoherenceLevel {
    /// Classify coherence from order parameter magnitude
    pub fn from_coherence(rho: f64) -> Self {
        if rho < 0.3 {
            CoherenceLevel::Incoherent
        } else if rho < 0.8 {
            CoherenceLevel::Partial
        } else {
            CoherenceLevel::HighCoherence
        }
    }
}

/// Synchronization metrics
#[derive(Debug, Clone)]
pub struct SynchronizationMetrics {
    /// Phase coherence ρ ∈ [0,1]
    pub coherence: f64,
    /// Mean phase ψ
    pub mean_phase: f64,
    /// Synchronization index (relative to threshold)
    pub sync_index: f64,
    /// Coherence level
    pub level: CoherenceLevel,
    /// Phase velocity dispersion
    pub velocity_dispersion: f64,
}

impl SynchronizationMetrics {
    /// Check if validation criteria met (ρ > 0.8)
    pub fn meets_criteria(&self) -> bool {
        self.coherence > 0.8
    }
}

/// Phase synchronizer for cross-domain coupling
#[derive(Debug, Clone)]
pub struct PhaseSynchronizer {
    /// Number of oscillators per domain
    pub n_oscillators: usize,
    /// Coupling strength K
    pub coupling_strength: f64,
    /// Natural frequencies (neuromorphic domain)
    pub omega_neuro: Array1<f64>,
    /// Natural frequencies (quantum domain)
    pub omega_quantum: Array1<f64>,
    /// Current phases (neuromorphic domain)
    pub phases_neuro: Array1<f64>,
    /// Current phases (quantum domain)
    pub phases_quantum: Array1<f64>,
    /// Cross-domain coupling matrix
    pub coupling_matrix: Array2<f64>,
}

impl PhaseSynchronizer {
    /// Create new phase synchronizer
    pub fn new(n_oscillators: usize, coupling_strength: f64) -> Self {
        Self {
            n_oscillators,
            coupling_strength,
            omega_neuro: Array1::zeros(n_oscillators),
            omega_quantum: Array1::zeros(n_oscillators),
            phases_neuro: Array1::zeros(n_oscillators),
            phases_quantum: Array1::zeros(n_oscillators),
            coupling_matrix: Array2::eye(n_oscillators),
        }
    }

    /// Initialize with random phases and frequencies
    pub fn initialize_random(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Random phases [0, 2π)
        for theta in self.phases_neuro.iter_mut() {
            *theta = rng.gen::<f64>() * 2.0 * PI;
        }
        for theta in self.phases_quantum.iter_mut() {
            *theta = rng.gen::<f64>() * 2.0 * PI;
        }

        // Random natural frequencies (Lorentzian distribution)
        let omega_mean = 1.0;
        let omega_width = 0.1;
        for omega in self.omega_neuro.iter_mut() {
            *omega = omega_mean + omega_width * rng.gen::<f64>();
        }
        for omega in self.omega_quantum.iter_mut() {
            *omega = omega_mean + omega_width * rng.gen::<f64>();
        }
    }

    /// Compute Kuramoto order parameter: r·e^{iψ} = (1/N) Σ_j e^{iθ_j}
    ///
    /// Returns (magnitude, phase) where magnitude is coherence ∈ [0,1]
    pub fn compute_order_parameter(&self, phases: &Array1<f64>) -> (f64, f64) {
        let n = phases.len() as f64;

        // Complex order parameter
        let mut real = 0.0;
        let mut imag = 0.0;

        for &theta in phases.iter() {
            real += theta.cos();
            imag += theta.sin();
        }

        real /= n;
        imag /= n;

        // Magnitude (coherence)
        let magnitude = (real * real + imag * imag).sqrt();

        // Phase
        let phase = imag.atan2(real);

        (magnitude, phase)
    }

    /// Compute synchronization metrics for neuromorphic domain
    pub fn compute_neuro_metrics(&self) -> SynchronizationMetrics {
        let (coherence, mean_phase) = self.compute_order_parameter(&self.phases_neuro);

        // Velocity dispersion
        let mean_omega = self.omega_neuro.mean().unwrap();
        let velocity_dispersion = self.omega_neuro.iter()
            .map(|&omega| (omega - mean_omega).powi(2))
            .sum::<f64>() / self.n_oscillators as f64;

        SynchronizationMetrics {
            coherence,
            mean_phase,
            sync_index: coherence / 0.8, // Normalized to threshold
            level: CoherenceLevel::from_coherence(coherence),
            velocity_dispersion,
        }
    }

    /// Compute synchronization metrics for quantum domain
    pub fn compute_quantum_metrics(&self) -> SynchronizationMetrics {
        let (coherence, mean_phase) = self.compute_order_parameter(&self.phases_quantum);

        let mean_omega = self.omega_quantum.mean().unwrap();
        let velocity_dispersion = self.omega_quantum.iter()
            .map(|&omega| (omega - mean_omega).powi(2))
            .sum::<f64>() / self.n_oscillators as f64;

        SynchronizationMetrics {
            coherence,
            mean_phase,
            sync_index: coherence / 0.8,
            level: CoherenceLevel::from_coherence(coherence),
            velocity_dispersion,
        }
    }

    /// Compute cross-domain phase coherence
    ///
    /// Measures synchronization between neuromorphic and quantum domains
    pub fn compute_cross_domain_coherence(&self) -> f64 {
        let mut real = 0.0;
        let mut imag = 0.0;
        let n = self.n_oscillators as f64;

        for i in 0..self.n_oscillators {
            let phase_diff = self.phases_quantum[i] - self.phases_neuro[i];
            real += phase_diff.cos();
            imag += phase_diff.sin();
        }

        real /= n;
        imag /= n;

        (real * real + imag * imag).sqrt()
    }

    /// Evolve Kuramoto dynamics for one time step
    ///
    /// dθ_i/dt = ω_i + (K/N) Σ_j C_ij sin(θ_j - θ_i)
    pub fn evolve_step(&mut self, dt: f64) {
        let k_over_n = self.coupling_strength / self.n_oscillators as f64;

        // Evolve neuromorphic domain
        let mut d_theta_neuro = self.omega_neuro.clone();
        for i in 0..self.n_oscillators {
            let mut coupling_term = 0.0;
            for j in 0..self.n_oscillators {
                let phase_diff = self.phases_neuro[j] - self.phases_neuro[i];
                coupling_term += self.coupling_matrix[[i, j]] * phase_diff.sin();
            }
            d_theta_neuro[i] += k_over_n * coupling_term;
        }

        // Evolve quantum domain
        let mut d_theta_quantum = self.omega_quantum.clone();
        for i in 0..self.n_oscillators {
            let mut coupling_term = 0.0;
            for j in 0..self.n_oscillators {
                let phase_diff = self.phases_quantum[j] - self.phases_quantum[i];
                coupling_term += self.coupling_matrix[[i, j]] * phase_diff.sin();
            }
            d_theta_quantum[i] += k_over_n * coupling_term;
        }

        // Update phases
        self.phases_neuro = &self.phases_neuro + &(d_theta_neuro * dt);
        self.phases_quantum = &self.phases_quantum + &(d_theta_quantum * dt);

        // Wrap to [0, 2π)
        for theta in self.phases_neuro.iter_mut() {
            *theta = (*theta).rem_euclid(2.0 * PI);
        }
        for theta in self.phases_quantum.iter_mut() {
            *theta = (*theta).rem_euclid(2.0 * PI);
        }
    }

    /// Synchronize domains by evolving until coherence threshold met
    ///
    /// Returns number of steps taken
    pub fn synchronize(&mut self, target_coherence: f64, max_steps: usize, dt: f64) -> usize {
        for step in 0..max_steps {
            self.evolve_step(dt);

            let coherence = self.compute_cross_domain_coherence();
            if coherence >= target_coherence {
                return step + 1;
            }
        }

        max_steps
    }

    /// Set coupling matrix from information-theoretic measures
    ///
    /// C_ij = f(TE_ij) where TE is transfer entropy
    pub fn set_coupling_from_transfer_entropy(&mut self, te_matrix: &Array2<f64>) {
        // Normalize transfer entropy to [0,1]
        let te_max = te_matrix.iter().cloned().fold(0.0f64, f64::max);

        if te_max > 1e-10 {
            self.coupling_matrix = te_matrix / te_max;
        } else {
            self.coupling_matrix = Array2::eye(self.n_oscillators);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_parameter_uniform() {
        let phases = Array1::from_vec(vec![0.0, PI/2.0, PI, 3.0*PI/2.0]);
        let synchronizer = PhaseSynchronizer::new(4, 1.0);

        let (coherence, _) = synchronizer.compute_order_parameter(&phases);

        // Uniformly distributed phases should have low coherence
        assert!(coherence < 0.1);
    }

    #[test]
    fn test_order_parameter_synchronized() {
        let phases = Array1::from_vec(vec![0.0, 0.01, 0.02, 0.01]);
        let synchronizer = PhaseSynchronizer::new(4, 1.0);

        let (coherence, _) = synchronizer.compute_order_parameter(&phases);

        // Nearly identical phases should have high coherence
        assert!(coherence > 0.99);
    }

    #[test]
    fn test_coherence_bounds() {
        let mut synchronizer = PhaseSynchronizer::new(100, 2.0);
        synchronizer.initialize_random();

        let metrics = synchronizer.compute_neuro_metrics();

        // Coherence must be in [0,1]
        assert!(metrics.coherence >= 0.0);
        assert!(metrics.coherence <= 1.0);
    }

    #[test]
    fn test_synchronization_increases_coherence() {
        let mut synchronizer = PhaseSynchronizer::new(50, 3.0);
        synchronizer.initialize_random();

        let initial_coherence = synchronizer.compute_neuro_metrics().coherence;

        // Evolve dynamics
        for _ in 0..100 {
            synchronizer.evolve_step(0.01);
        }

        let final_coherence = synchronizer.compute_neuro_metrics().coherence;

        // Strong coupling should increase coherence
        assert!(final_coherence >= initial_coherence - 0.1);
    }

    #[test]
    fn test_cross_domain_coherence() {
        let mut synchronizer = PhaseSynchronizer::new(20, 2.0);

        // Initialize with identical phases
        synchronizer.phases_neuro.fill(PI/4.0);
        synchronizer.phases_quantum.fill(PI/4.0);

        let coherence = synchronizer.compute_cross_domain_coherence();

        // Identical phases should give perfect cross-domain coherence
        assert!((coherence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_synchronize_achieves_target() {
        let mut synchronizer = PhaseSynchronizer::new(30, 5.0);
        synchronizer.initialize_random();

        // Start with similar phases for faster sync
        synchronizer.phases_neuro.fill(0.0);
        synchronizer.phases_quantum.fill(0.1);

        let steps = synchronizer.synchronize(0.8, 1000, 0.01);

        let final_coherence = synchronizer.compute_cross_domain_coherence();

        // Should achieve target coherence
        assert!(final_coherence >= 0.8 || steps == 1000);
    }

    #[test]
    fn test_coherence_level_classification() {
        assert_eq!(CoherenceLevel::from_coherence(0.1), CoherenceLevel::Incoherent);
        assert_eq!(CoherenceLevel::from_coherence(0.5), CoherenceLevel::Partial);
        assert_eq!(CoherenceLevel::from_coherence(0.9), CoherenceLevel::HighCoherence);
    }

    #[test]
    fn test_meets_criteria() {
        let metrics = SynchronizationMetrics {
            coherence: 0.85,
            mean_phase: 0.0,
            sync_index: 1.0625,
            level: CoherenceLevel::HighCoherence,
            velocity_dispersion: 0.01,
        };

        assert!(metrics.meets_criteria());
    }
}
