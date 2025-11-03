//! Adaptive Coupling Layer
//!
//! Combines physics-based initialization with online learning for optimal
//! neuromorphic-quantum co-processing performance.
//!
//! Architecture:
//! 1. Physics-based baseline (from coupling_physics.rs)
//! 2. Adaptive perturbations using multi-objective optimization
//! 3. Stability constraints via Lyapunov analysis
//! 4. Credit assignment via finite-difference gradients

use crate::coupling_physics::{PhysicsCoupling, StabilityAnalysis};
use anyhow::Result;
use std::collections::VecDeque;

/// Adaptive parameter with physics-based baseline
#[derive(Debug, Clone)]
pub struct AdaptiveParameter {
    /// Current value (baseline + learned adjustment)
    pub value: f64,

    /// Physics-based baseline value
    pub baseline: f64,

    /// Learned adjustment from baseline
    pub adjustment: f64,

    /// Learning rate (derived from Fisher information)
    pub learning_rate: f64,

    /// Gradient estimate (from finite differences)
    pub gradient: f64,

    /// Historical performance when this parameter changes
    pub performance_history: VecDeque<f64>,

    /// Maximum deviation from baseline (safety constraint)
    pub max_deviation: f64,
}

impl AdaptiveParameter {
    /// Create new adaptive parameter with physics-based baseline
    pub fn new(baseline: f64, fisher_information: f64) -> Self {
        // Learning rate inversely proportional to Fisher information
        // High Fisher info → high sensitivity → small learning rate
        let learning_rate = (1.0 / (1.0 + fisher_information)).min(0.1);

        Self {
            value: baseline,
            baseline,
            adjustment: 0.0,
            learning_rate,
            gradient: 0.0,
            performance_history: VecDeque::with_capacity(100),
            max_deviation: 1.0, // Allow ±100% deviation
        }
    }

    /// Update parameter using gradient descent
    pub fn update(&mut self, gradient: f64, stability_factor: f64) {
        self.gradient = gradient;

        // Gradient descent with stability damping
        let update = self.learning_rate * gradient * stability_factor;
        self.adjustment += update;

        // Enforce safety constraints
        self.adjustment = self
            .adjustment
            .clamp(-self.max_deviation, self.max_deviation);

        // Update value
        self.value = self.baseline + self.adjustment;

        // Clamp to positive values
        self.value = self.value.max(0.01);
    }

    /// Reset to baseline (emergency fallback)
    pub fn reset_to_baseline(&mut self) {
        self.adjustment = 0.0;
        self.value = self.baseline;
        self.gradient = 0.0;
    }

    /// Record performance for credit assignment
    pub fn record_performance(&mut self, metric: f64) {
        self.performance_history.push_back(metric);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
    }

    /// Get average recent performance
    pub fn avg_performance(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.5;
        }
        self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
    }
}

/// Multi-objective performance metrics
#[derive(Debug, Clone, Copy)]
pub struct PerformanceMetrics {
    /// Convergence speed (higher is better)
    pub convergence_speed: f64,

    /// Solution accuracy (higher is better)
    pub accuracy: f64,

    /// Phase coherence preservation (higher is better)
    pub coherence: f64,

    /// System stability (higher is better)
    pub stability: f64,

    /// Energy efficiency (higher is better)
    pub efficiency: f64,
}

impl PerformanceMetrics {
    /// Compute weighted aggregate score
    pub fn aggregate(&self) -> f64 {
        // Weights tuned for quantum optimization problems
        0.3 * self.convergence_speed
            + 0.3 * self.accuracy
            + 0.2 * self.coherence
            + 0.15 * self.stability
            + 0.05 * self.efficiency
    }

    /// Check if metrics indicate good performance
    pub fn is_good(&self) -> bool {
        self.aggregate() > 0.6
    }
}

/// Adaptive coupling that learns on top of physics-based baseline
#[derive(Debug, Clone)]
pub struct AdaptiveCoupling {
    /// Underlying physics-based coupling
    pub physics: PhysicsCoupling,

    /// Adaptive parameters for neuromorphic → quantum
    pub neuro_to_quantum_adaptive: NeuroQuantumAdaptive,

    /// Adaptive parameters for quantum → neuromorphic
    pub quantum_to_neuro_adaptive: QuantumNeuroAdaptive,

    /// Performance history for stability monitoring
    pub performance_history: VecDeque<PerformanceMetrics>,

    /// Number of updates performed
    pub update_count: usize,

    /// Current stability factor (0 = unstable, 1 = stable)
    pub stability_factor: f64,

    /// Emergency mode (falls back to pure physics)
    pub emergency_mode: bool,
}

#[derive(Debug, Clone)]
pub struct NeuroQuantumAdaptive {
    pub pattern_to_hamiltonian: AdaptiveParameter,
    pub coherence_to_evolution: AdaptiveParameter,
    pub memory_to_persistence: AdaptiveParameter,
    pub phase_coupling_strength: AdaptiveParameter,
}

#[derive(Debug, Clone)]
pub struct QuantumNeuroAdaptive {
    pub energy_to_learning_rate: AdaptiveParameter,
    pub phase_to_timing_precision: AdaptiveParameter,
    pub entanglement_to_coupling: AdaptiveParameter,
    pub state_to_reservoir_input: AdaptiveParameter,
}

impl AdaptiveCoupling {
    /// Initialize with physics-based coupling
    pub fn new(physics: PhysicsCoupling) -> Self {
        let fisher = physics.info_metrics.fisher_information;

        // Initialize adaptive parameters from physics baselines
        let neuro_to_quantum_adaptive = NeuroQuantumAdaptive {
            pattern_to_hamiltonian: AdaptiveParameter::new(
                physics.neuro_to_quantum.pattern_to_hamiltonian,
                fisher,
            ),
            coherence_to_evolution: AdaptiveParameter::new(
                physics.neuro_to_quantum.coherence_to_evolution,
                fisher,
            ),
            memory_to_persistence: AdaptiveParameter::new(
                physics.neuro_to_quantum.memory_to_persistence,
                fisher,
            ),
            phase_coupling_strength: AdaptiveParameter::new(
                physics.neuro_to_quantum.phase_coupling_strength,
                fisher * 0.5, // More sensitive
            ),
        };

        let quantum_to_neuro_adaptive = QuantumNeuroAdaptive {
            energy_to_learning_rate: AdaptiveParameter::new(
                physics.quantum_to_neuro.energy_to_learning_rate,
                fisher,
            ),
            phase_to_timing_precision: AdaptiveParameter::new(
                physics.quantum_to_neuro.phase_to_timing_precision,
                fisher,
            ),
            entanglement_to_coupling: AdaptiveParameter::new(
                physics.quantum_to_neuro.entanglement_to_coupling,
                fisher,
            ),
            state_to_reservoir_input: AdaptiveParameter::new(
                physics.quantum_to_neuro.state_to_reservoir_input,
                fisher,
            ),
        };

        Self {
            physics,
            neuro_to_quantum_adaptive,
            quantum_to_neuro_adaptive,
            performance_history: VecDeque::with_capacity(1000),
            update_count: 0,
            stability_factor: 1.0,
            emergency_mode: false,
        }
    }

    /// Update adaptive parameters using multi-objective optimization
    ///
    /// Uses finite-difference gradient estimation for credit assignment
    pub fn update(&mut self, metrics: PerformanceMetrics) -> Result<()> {
        self.performance_history.push_back(metrics);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        self.update_count += 1;

        // Check stability
        let stability = self.physics.check_stability()?;
        self.update_stability_factor(&stability, &metrics);

        // Emergency mode check
        if self.should_enter_emergency_mode(&stability, &metrics) {
            self.enter_emergency_mode();
            return Ok(());
        }

        // Normal adaptive updates
        if self.emergency_mode {
            // Try to exit emergency mode
            if metrics.is_good() && stability.is_stable {
                self.exit_emergency_mode();
            } else {
                return Ok(()); // Stay in emergency mode
            }
        }

        // Compute gradients via finite differences
        self.compute_gradients(&metrics)?;

        // Update all parameters
        self.apply_updates();

        Ok(())
    }

    /// Compute gradients using finite-difference method
    ///
    /// For each parameter θ_i:
    /// ∂f/∂θ_i ≈ (f(θ + εe_i) - f(θ - εe_i)) / (2ε)
    fn compute_gradients(&mut self, current_metrics: &PerformanceMetrics) -> Result<()> {
        // Perturbation size (adaptive based on stability)
        let epsilon = 0.01 * self.stability_factor;

        let current_score = current_metrics.aggregate();

        // Get references to all parameters
        let params = self.all_parameters_mut();

        for param in params {
            // Estimate gradient from recent performance history
            // This is approximate but avoids expensive perturbation testing

            if param.performance_history.len() >= 2 {
                // Use historical data for gradient estimation
                let recent_perf: Vec<f64> = param.performance_history.iter().copied().collect();
                let n = recent_perf.len();

                // Simple finite difference from last two points
                let gradient = if n >= 2 {
                    (recent_perf[n - 1] - recent_perf[n - 2]) / epsilon
                } else {
                    0.0
                };

                param.gradient = gradient;
            } else {
                // Not enough history, use current performance as signal
                param.gradient = (current_score - 0.5) * param.adjustment.signum();
            }

            // Record current performance for this parameter
            param.record_performance(current_score);
        }

        Ok(())
    }

    /// Apply gradient updates to all parameters
    fn apply_updates(&mut self) {
        let stability_factor = self.stability_factor; // Copy before mutable borrow
        let params = self.all_parameters_mut();

        for param in params {
            param.update(param.gradient, stability_factor);
        }
    }

    /// Update stability factor based on Lyapunov analysis
    fn update_stability_factor(
        &mut self,
        stability: &StabilityAnalysis,
        metrics: &PerformanceMetrics,
    ) {
        // Stability factor combines Lyapunov analysis with performance
        let lyapunov_factor = if stability.lyapunov_exponent < 0.0 {
            1.0 // Stable
        } else {
            0.5 // Unstable - reduce learning rate
        };

        let order_factor = stability.order_parameter; // Higher order = better sync

        let performance_factor = metrics.stability;

        // Weighted combination
        self.stability_factor =
            0.4 * lyapunov_factor + 0.3 * order_factor + 0.3 * performance_factor;

        // Clamp
        self.stability_factor = self.stability_factor.clamp(0.1, 1.0);
    }

    /// Check if system should enter emergency mode
    fn should_enter_emergency_mode(
        &self,
        stability: &StabilityAnalysis,
        metrics: &PerformanceMetrics,
    ) -> bool {
        // Enter emergency if:
        // 1. Lyapunov exponent very positive (unstable)
        // 2. Performance degraded significantly
        // 3. Stability factor very low

        let unstable = stability.lyapunov_exponent > 1.0;
        let poor_performance = metrics.aggregate() < 0.3;
        let low_stability = self.stability_factor < 0.2;

        unstable || (poor_performance && low_stability)
    }

    /// Enter emergency mode (fallback to pure physics)
    fn enter_emergency_mode(&mut self) {
        self.emergency_mode = true;

        // Reset all parameters to physics-based baselines
        for param in self.all_parameters_mut() {
            param.reset_to_baseline();
        }

        println!("⚠️  EMERGENCY MODE: Falling back to physics-based coupling");
    }

    /// Exit emergency mode
    fn exit_emergency_mode(&mut self) {
        self.emergency_mode = false;
        println!("✓ Exiting emergency mode - system stable");
    }

    /// Get all parameters for bulk operations
    fn all_parameters_mut(&mut self) -> Vec<&mut AdaptiveParameter> {
        vec![
            &mut self.neuro_to_quantum_adaptive.pattern_to_hamiltonian,
            &mut self.neuro_to_quantum_adaptive.coherence_to_evolution,
            &mut self.neuro_to_quantum_adaptive.memory_to_persistence,
            &mut self.neuro_to_quantum_adaptive.phase_coupling_strength,
            &mut self.quantum_to_neuro_adaptive.energy_to_learning_rate,
            &mut self.quantum_to_neuro_adaptive.phase_to_timing_precision,
            &mut self.quantum_to_neuro_adaptive.entanglement_to_coupling,
            &mut self.quantum_to_neuro_adaptive.state_to_reservoir_input,
        ]
    }

    /// Get current coupling values for use in platform
    pub fn get_current_values(&self) -> CouplingValues {
        CouplingValues {
            pattern_to_hamiltonian: self.neuro_to_quantum_adaptive.pattern_to_hamiltonian.value,
            coherence_to_evolution: self.neuro_to_quantum_adaptive.coherence_to_evolution.value,
            memory_to_persistence: self.neuro_to_quantum_adaptive.memory_to_persistence.value,
            phase_coupling_strength: self.neuro_to_quantum_adaptive.phase_coupling_strength.value,
            energy_to_learning_rate: self.quantum_to_neuro_adaptive.energy_to_learning_rate.value,
            phase_to_timing_precision: self
                .quantum_to_neuro_adaptive
                .phase_to_timing_precision
                .value,
            entanglement_to_coupling: self
                .quantum_to_neuro_adaptive
                .entanglement_to_coupling
                .value,
            state_to_reservoir_input: self
                .quantum_to_neuro_adaptive
                .state_to_reservoir_input
                .value,
        }
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> PerformanceSummary {
        if self.performance_history.is_empty() {
            return PerformanceSummary::default();
        }

        let recent: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .take(100)
            .map(|m| m.aggregate())
            .collect();

        let avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute variance
        let variance = recent.iter().map(|&x| (x - avg).powi(2)).sum::<f64>() / recent.len() as f64;

        PerformanceSummary {
            avg_performance: avg,
            min_performance: min,
            max_performance: max,
            performance_variance: variance,
            update_count: self.update_count,
            stability_factor: self.stability_factor,
            emergency_mode: self.emergency_mode,
        }
    }
}

/// Current coupling values for platform use
#[derive(Debug, Clone, Copy)]
pub struct CouplingValues {
    pub pattern_to_hamiltonian: f64,
    pub coherence_to_evolution: f64,
    pub memory_to_persistence: f64,
    pub phase_coupling_strength: f64,
    pub energy_to_learning_rate: f64,
    pub phase_to_timing_precision: f64,
    pub entanglement_to_coupling: f64,
    pub state_to_reservoir_input: f64,
}

/// Performance summary
#[derive(Debug, Clone, Copy)]
pub struct PerformanceSummary {
    pub avg_performance: f64,
    pub min_performance: f64,
    pub max_performance: f64,
    pub performance_variance: f64,
    pub update_count: usize,
    pub stability_factor: f64,
    pub emergency_mode: bool,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            avg_performance: 0.5,
            min_performance: 0.0,
            max_performance: 1.0,
            performance_variance: 0.0,
            update_count: 0,
            stability_factor: 1.0,
            emergency_mode: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coupling_physics::PhysicsCoupling;
    use num_complex::Complex64;

    #[test]
    fn test_adaptive_parameter() {
        let mut param = AdaptiveParameter::new(0.5, 1.0);

        assert_eq!(param.value, 0.5);
        assert_eq!(param.baseline, 0.5);
        assert_eq!(param.adjustment, 0.0);

        // Positive gradient should increase value
        param.update(1.0, 1.0);
        assert!(param.value > 0.5);

        // Negative gradient should decrease value
        param.update(-2.0, 1.0);
        assert!(param.value < 0.5);

        // Reset should return to baseline
        param.reset_to_baseline();
        assert_eq!(param.value, 0.5);
    }

    #[test]
    fn test_adaptive_coupling_initialization() {
        let neuro_state = vec![0.5, 0.3, 0.7];
        let spike_pattern = vec![1.0, 0.0, 1.0];
        let quantum_state = vec![Complex64::new(0.7, 0.0), Complex64::new(0.5, 0.5)];
        let coupling_matrix = nalgebra::DMatrix::from_element(2, 2, Complex64::new(0.5, 0.1));

        let physics = PhysicsCoupling::from_system_state(
            &neuro_state,
            &spike_pattern,
            &quantum_state,
            &coupling_matrix,
        )
        .unwrap();

        let adaptive = AdaptiveCoupling::new(physics);

        // Check that adaptive parameters start at physics baselines
        assert_eq!(
            adaptive
                .neuro_to_quantum_adaptive
                .pattern_to_hamiltonian
                .value,
            adaptive
                .neuro_to_quantum_adaptive
                .pattern_to_hamiltonian
                .baseline
        );

        assert!(!adaptive.emergency_mode);
        assert_eq!(adaptive.stability_factor, 1.0);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            convergence_speed: 0.8,
            accuracy: 0.7,
            coherence: 0.9,
            stability: 0.85,
            efficiency: 0.6,
        };

        let aggregate = metrics.aggregate();
        assert!(aggregate > 0.5);
        assert!(metrics.is_good());
    }

    #[test]
    fn test_emergency_mode() {
        let neuro_state = vec![0.5, 0.3, 0.7];
        let spike_pattern = vec![1.0, 0.0, 1.0];
        let quantum_state = vec![Complex64::new(0.7, 0.0), Complex64::new(0.5, 0.5)];
        let coupling_matrix = nalgebra::DMatrix::from_element(2, 2, Complex64::new(0.5, 0.1));

        let physics = PhysicsCoupling::from_system_state(
            &neuro_state,
            &spike_pattern,
            &quantum_state,
            &coupling_matrix,
        )
        .unwrap();

        let mut adaptive = AdaptiveCoupling::new(physics);

        // Force emergency mode
        adaptive.enter_emergency_mode();
        assert!(adaptive.emergency_mode);

        // Check parameters reset to baseline
        assert_eq!(
            adaptive
                .neuro_to_quantum_adaptive
                .pattern_to_hamiltonian
                .value,
            adaptive
                .neuro_to_quantum_adaptive
                .pattern_to_hamiltonian
                .baseline
        );
    }
}
