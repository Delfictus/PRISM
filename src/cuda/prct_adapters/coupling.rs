//! Physics Coupling Adapter
//!
//! Implements PhysicsCouplingPort for the PRCT algorithm.
//! Bridges neuromorphic and quantum layers via Kuramoto synchronization.

use crate::cuda::prct_gpu::PRCTGpuManager;
use prct_core::errors::Result;
use prct_core::ports::PhysicsCouplingPort;
use shared_types::*;
use std::sync::Arc;

/// Physics coupling adapter implementing Kuramoto synchronization
pub struct PhysicsCouplingAdapter {
    coupling_strength: f64,
    dt: f64,
    gpu_manager: Option<Arc<PRCTGpuManager>>,
}

impl PhysicsCouplingAdapter {
    pub fn new(coupling_strength: f64) -> Result<Self> {
        // Try to initialize GPU, fall back to CPU if unavailable
        let gpu_manager = PRCTGpuManager::new().ok().map(Arc::new);

        if gpu_manager.is_some() {
            log::info!("[COUPLING] GPU acceleration enabled");
        } else {
            log::warn!("[COUPLING] GPU unavailable, using CPU fallback");
        }

        Ok(Self {
            coupling_strength,
            dt: 0.01,
            gpu_manager,
        })
    }
}

impl PhysicsCouplingPort for PhysicsCouplingAdapter {
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength> {
        // Compute bidirectional coupling strength based on system coherences
        let neuro_to_quantum = neuro_state.coherence * self.coupling_strength;
        let quantum_to_neuro = quantum_state.phase_coherence * self.coupling_strength;

        // Overall coherence is the geometric mean
        let bidirectional_coherence = (neuro_to_quantum * quantum_to_neuro).sqrt();

        Ok(CouplingStrength {
            neuro_to_quantum,
            quantum_to_neuro,
            bidirectional_coherence,
            timestamp_ns: 0,
        })
    }

    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState> {
        let n = neuro_phases.len().min(quantum_phases.len());

        // Initialize natural frequencies from quantum phases
        let natural_frequencies: Vec<f64> = quantum_phases
            .iter()
            .take(n)
            .map(|&phase| phase / (2.0 * std::f64::consts::PI))
            .collect();

        // Build coupling matrix (all-to-all for simplicity)
        let coupling_matrix = vec![1.0; n * n];

        // Initialize phases from neuromorphic layer
        let mut phases = neuro_phases[..n].to_vec();

        // Perform multiple Kuramoto steps
        let num_steps = 100;
        let step_dt = dt / num_steps as f64;

        // Try GPU acceleration
        if let Some(gpu) = &self.gpu_manager {
            for _ in 0..num_steps {
                match gpu.kuramoto_step_gpu(
                    &phases,
                    &natural_frequencies,
                    &coupling_matrix,
                    self.coupling_strength,
                    step_dt,
                ) {
                    Ok(new_phases) => {
                        phases = new_phases;
                    }
                    Err(e) => {
                        log::warn!(
                            "[COUPLING] GPU Kuramoto step failed: {}, falling back to CPU",
                            e
                        );
                        phases = self.kuramoto_step_cpu(
                            &phases,
                            &natural_frequencies,
                            &coupling_matrix,
                            step_dt,
                            n,
                        );
                    }
                }
            }
        } else {
            // CPU fallback
            for _ in 0..num_steps {
                phases = self.kuramoto_step_cpu(
                    &phases,
                    &natural_frequencies,
                    &coupling_matrix,
                    step_dt,
                    n,
                );
            }
        }

        // Compute order parameter - try GPU first
        let order_parameter = if let Some(gpu) = &self.gpu_manager {
            match gpu.kuramoto_order_parameter_gpu(&phases) {
                Ok(order) => {
                    log::debug!("[COUPLING] GPU order parameter: {:.6}", order);
                    order
                }
                Err(e) => {
                    log::warn!("[COUPLING] GPU order parameter failed: {}, using CPU", e);
                    self.compute_order_parameter(&phases)
                }
            }
        } else {
            self.compute_order_parameter(&phases)
        };

        // Compute mean phase
        let mean_phase = phases.iter().sum::<f64>() / n as f64;

        Ok(KuramotoState {
            phases,
            natural_frequencies,
            coupling_matrix,
            order_parameter,
            mean_phase,
        })
    }

    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        lag: f64,
    ) -> Result<TransferEntropy> {
        let min_len = source.len().min(target.len());
        if min_len < 2 {
            return Ok(TransferEntropy {
                entropy_bits: 0.0,
                confidence: 0.0,
                lag_ms: lag,
            });
        }

        // Compute means
        let source_mean = source.iter().take(min_len).sum::<f64>() / min_len as f64;
        let target_mean = target.iter().take(min_len).sum::<f64>() / min_len as f64;

        // Try GPU acceleration for correlation computation
        let (covariance, source_var, target_var) = if let Some(gpu) = &self.gpu_manager {
            match gpu.compute_correlation_gpu(source, target, source_mean, target_mean) {
                Ok((cov, var_s, var_t)) => {
                    log::debug!("[COUPLING] GPU correlation succeeded");
                    (cov, var_s, var_t)
                }
                Err(e) => {
                    log::warn!("[COUPLING] GPU correlation failed: {}, using CPU", e);
                    self.compute_correlation_cpu(source, target, source_mean, target_mean, min_len)
                }
            }
        } else {
            self.compute_correlation_cpu(source, target, source_mean, target_mean, min_len)
        };

        let correlation = if source_var > 1e-10 && target_var > 1e-10 {
            covariance / (source_var * target_var).sqrt()
        } else {
            0.0
        };

        // Convert correlation to bits (simplified)
        let entropy_bits = -correlation.abs() * correlation.abs().max(1e-10).log2();
        let confidence = correlation.abs();

        Ok(TransferEntropy {
            entropy_bits,
            confidence,
            lag_ms: lag,
        })
    }

    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling> {
        // Extract phases from neuron states (normalize to [0, 2π])
        let neuro_phases: Vec<f64> = neuro_state
            .neuron_states
            .iter()
            .map(|&state| state * 2.0 * std::f64::consts::PI)
            .collect();

        // Extract phases from quantum amplitudes
        let quantum_phases: Vec<f64> = quantum_state
            .amplitudes
            .iter()
            .map(|(re, im)| im.atan2(*re))
            .collect();

        // Compute transfer entropies
        let neuro_to_quantum_entropy = self.calculate_transfer_entropy(
            &neuro_state.neuron_states,
            &quantum_phases,
            1.0, // 1 ms lag
        )?;

        let quantum_to_neuro_entropy =
            self.calculate_transfer_entropy(&quantum_phases, &neuro_state.neuron_states, 1.0)?;

        // Update Kuramoto synchronization
        let kuramoto_state = self.update_kuramoto_sync(
            &neuro_phases,
            &quantum_phases,
            0.1, // 0.1 second evolution
        )?;

        // Compute overall coupling quality
        let coupling_quality = (neuro_to_quantum_entropy.confidence
            + quantum_to_neuro_entropy.confidence
            + kuramoto_state.order_parameter)
            / 3.0;

        Ok(BidirectionalCoupling {
            neuro_to_quantum_entropy,
            quantum_to_neuro_entropy,
            kuramoto_state,
            coupling_quality,
        })
    }
}

impl PhysicsCouplingAdapter {
    /// CPU fallback for Kuramoto evolution step
    fn kuramoto_step_cpu(
        &self,
        phases: &[f64],
        natural_frequencies: &[f64],
        coupling_matrix: &[f64],
        dt: f64,
        n: usize,
    ) -> Vec<f64> {
        let mut new_phases = vec![0.0; n];

        for i in 0..n {
            // Natural frequency term
            let mut phase_derivative = natural_frequencies[i];

            // Coupling term
            let mut coupling_sum = 0.0;
            for j in 0..n {
                if i != j {
                    let coupling_ij = coupling_matrix[i * n + j];
                    let phase_diff = phases[j] - phases[i];
                    coupling_sum += coupling_ij * phase_diff.sin();
                }
            }

            phase_derivative += (self.coupling_strength / n as f64) * coupling_sum;

            // Update phase
            let new_phase = phases[i] + phase_derivative * dt;
            new_phases[i] = new_phase.rem_euclid(2.0 * std::f64::consts::PI);
        }

        new_phases
    }

    /// CPU fallback for correlation computation
    fn compute_correlation_cpu(
        &self,
        source: &[f64],
        target: &[f64],
        source_mean: f64,
        target_mean: f64,
        min_len: usize,
    ) -> (f64, f64, f64) {
        let covariance: f64 = source
            .iter()
            .zip(target.iter())
            .take(min_len)
            .map(|(s, t)| (s - source_mean) * (t - target_mean))
            .sum::<f64>()
            / min_len as f64;

        let source_var: f64 = source
            .iter()
            .take(min_len)
            .map(|s| (s - source_mean).powi(2))
            .sum::<f64>()
            / min_len as f64;

        let target_var: f64 = target
            .iter()
            .take(min_len)
            .map(|t| (t - target_mean).powi(2))
            .sum::<f64>()
            / min_len as f64;

        (covariance, source_var, target_var)
    }

    fn compute_order_parameter(&self, phases: &[f64]) -> f64 {
        if phases.is_empty() {
            return 0.0;
        }

        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        // Order parameter: r = |<e^(iθ)>|
        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }
}
