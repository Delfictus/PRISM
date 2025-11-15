//! Quantum Adapter
//!
//! Implements QuantumPort for the PRCT algorithm.
//! Constructs Hamiltonians from graphs and evolves quantum states.

use crate::cuda::prct_gpu::PRCTGpuManager;
use prct_core::errors::Result;
use prct_core::ports::QuantumPort;
use shared_types::*;
use std::sync::Arc;

/// Quantum processing adapter
pub struct QuantumAdapter {
    coupling_strength: f64,
    evolution_time: f64,
    gpu_manager: Option<Arc<PRCTGpuManager>>,
}

impl QuantumAdapter {
    pub fn new() -> Result<Self> {
        // Try to initialize GPU, fall back to CPU if unavailable
        let gpu_manager = PRCTGpuManager::new().ok().map(Arc::new);

        if gpu_manager.is_some() {
            log::info!("[QUANTUM] GPU acceleration enabled");
        } else {
            log::warn!("[QUANTUM] GPU unavailable, using CPU fallback");
        }

        Ok(Self {
            coupling_strength: 1.0,
            evolution_time: 1.0,
            gpu_manager,
        })
    }
}

impl QuantumPort for QuantumAdapter {
    fn build_hamiltonian(
        &self,
        graph: &Graph,
        params: &EvolutionParams,
    ) -> Result<HamiltonianState> {
        let n = graph.num_vertices;

        // Build Hamiltonian matrix from graph structure
        // H = -J * sum_{<i,j>} |i><j| + |j><i|
        let mut matrix_elements = vec![(0.0, 0.0); n * n];

        // Add coupling terms for edges
        for i in 0..n {
            for j in 0..n {
                if i != j && graph.adjacency[i * n + j] {
                    // Off-diagonal coupling
                    let idx = i * n + j;
                    matrix_elements[idx] = (-params.strength, 0.0);
                }
            }
        }

        // Add diagonal terms (on-site energies based on degree)
        for i in 0..n {
            let degree = (0..n)
                .filter(|&j| j != i && graph.adjacency[i * n + j])
                .count();

            let idx = i * n + i;
            matrix_elements[idx] = (degree as f64 * 0.5, 0.0);
        }

        // Compute eigenvalues (simplified - just extract diagonal for now)
        let eigenvalues: Vec<f64> = (0..n).map(|i| matrix_elements[i * n + i].0).collect();

        // Find ground state energy
        let ground_state_energy = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);

        Ok(HamiltonianState {
            matrix_elements,
            eigenvalues,
            ground_state_energy,
            dimension: n,
        })
    }

    fn evolve_state(
        &self,
        hamiltonian: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        let n = hamiltonian.dimension;

        // Try GPU acceleration
        let amplitudes = if let Some(gpu) = &self.gpu_manager {
            // Separate real and imaginary parts for GPU
            let mut amps_re: Vec<f64> =
                initial_state.amplitudes.iter().map(|(re, _)| *re).collect();
            let mut amps_im: Vec<f64> =
                initial_state.amplitudes.iter().map(|(_, im)| *im).collect();

            match gpu.quantum_evolve_gpu(
                &mut amps_re,
                &mut amps_im,
                &hamiltonian.eigenvalues,
                evolution_time,
            ) {
                Ok(_) => {
                    log::debug!("[QUANTUM] GPU evolution succeeded");
                    // Recombine into tuples
                    amps_re.into_iter().zip(amps_im.into_iter()).collect()
                }
                Err(e) => {
                    log::warn!("[QUANTUM] GPU evolution failed: {}, falling back to CPU", e);
                    self.evolve_state_cpu(
                        &initial_state.amplitudes,
                        &hamiltonian.eigenvalues,
                        evolution_time,
                    )
                }
            }
        } else {
            self.evolve_state_cpu(
                &initial_state.amplitudes,
                &hamiltonian.eigenvalues,
                evolution_time,
            )
        };

        // Compute phase coherence
        let phases: Vec<f64> = amplitudes.iter().map(|(re, im)| im.atan2(*re)).collect();
        let phase_coherence = self.compute_phase_coherence(&phases);

        // Compute energy
        let energy = amplitudes
            .iter()
            .enumerate()
            .map(|(i, (re, im))| {
                let prob = re * re + im * im;
                prob * hamiltonian.eigenvalues.get(i).unwrap_or(&0.0)
            })
            .sum();

        // Compute entanglement
        let entanglement = self.compute_entanglement(&amplitudes);

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement,
            timestamp_ns: 0,
        })
    }

    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField> {
        let n = state.amplitudes.len();

        // Extract phases - try GPU first
        let phases = if let Some(gpu) = &self.gpu_manager {
            let amps_re: Vec<f64> = state.amplitudes.iter().map(|(re, _)| *re).collect();
            let amps_im: Vec<f64> = state.amplitudes.iter().map(|(_, im)| *im).collect();

            match gpu.extract_phases_gpu(&amps_re, &amps_im) {
                Ok(p) => {
                    log::debug!("[QUANTUM] GPU phase extraction succeeded");
                    p
                }
                Err(e) => {
                    log::warn!("[QUANTUM] GPU phase extraction failed: {}, using CPU", e);
                    state
                        .amplitudes
                        .iter()
                        .map(|(re, im)| im.atan2(*re))
                        .collect()
                }
            }
        } else {
            state
                .amplitudes
                .iter()
                .map(|(re, im)| im.atan2(*re))
                .collect()
        };

        // Compute coherence matrix - try GPU first
        let coherence_matrix = if let Some(gpu) = &self.gpu_manager {
            match gpu.compute_coherence_matrix_gpu(&phases) {
                Ok(matrix) => {
                    log::debug!("[QUANTUM] GPU coherence matrix succeeded");
                    matrix
                }
                Err(e) => {
                    log::warn!("[QUANTUM] GPU coherence matrix failed: {}, using CPU", e);
                    self.compute_coherence_matrix_cpu(&phases, n)
                }
            }
        } else {
            self.compute_coherence_matrix_cpu(&phases, n)
        };

        // Compute global order parameter
        let order_parameter = state.phase_coherence;

        // Estimate resonance frequency
        let resonance_frequency = 1.0 / (2.0 * std::f64::consts::PI);

        Ok(PhaseField {
            phases,
            coherence_matrix,
            order_parameter,
            resonance_frequency,
        })
    }

    fn compute_ground_state(&self, hamiltonian: &HamiltonianState) -> Result<QuantumState> {
        let n = hamiltonian.dimension;

        // Initialize in equal superposition
        let amplitude_val = 1.0 / (n as f64).sqrt();
        let mut amplitudes = vec![(amplitude_val, 0.0); n];

        // Apply imaginary time evolution to find ground state
        // This is a simplified version - real implementation would use
        // power iteration or Lanczos method
        let imaginary_time = 10.0;

        for i in 0..n {
            let energy = hamiltonian.eigenvalues.get(i).unwrap_or(&0.0);
            let decay_factor = (-energy * imaginary_time).exp();
            amplitudes[i] = (decay_factor / (n as f64).sqrt(), 0.0);
        }

        // Normalize
        let norm: f64 = amplitudes
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>()
            .sqrt();

        if norm > 1e-10 {
            for (re, im) in &mut amplitudes {
                *re /= norm;
                *im /= norm;
            }
        }

        // Compute phase coherence
        let phases: Vec<f64> = amplitudes.iter().map(|(re, im)| im.atan2(*re)).collect();
        let phase_coherence = self.compute_phase_coherence(&phases);

        // Compute ground state energy
        let energy = hamiltonian.ground_state_energy;

        // Compute entanglement
        let entanglement = self.compute_entanglement(&amplitudes);

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement,
            timestamp_ns: 0,
        })
    }
}

impl QuantumAdapter {
    /// CPU fallback for quantum evolution
    fn evolve_state_cpu(
        &self,
        amplitudes: &[(f64, f64)],
        eigenvalues: &[f64],
        time: f64,
    ) -> Vec<(f64, f64)> {
        let n = amplitudes.len();
        let mut result = amplitudes.to_vec();

        for i in 0..n.min(eigenvalues.len()) {
            let energy = eigenvalues[i];
            let phase = -energy * time;

            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            let (re, im) = result[i];
            result[i] = (
                re * cos_phase - im * sin_phase,
                re * sin_phase + im * cos_phase,
            );
        }

        // Normalize
        let norm: f64 = result
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>()
            .sqrt();

        if norm > 1e-10 {
            for (re, im) in &mut result {
                *re /= norm;
                *im /= norm;
            }
        }

        result
    }

    /// CPU fallback for coherence matrix
    fn compute_coherence_matrix_cpu(&self, phases: &[f64], n: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let phase_diff = (phases[j] - phases[i]).abs();
                let normalized_diff = if phase_diff > std::f64::consts::PI {
                    2.0 * std::f64::consts::PI - phase_diff
                } else {
                    phase_diff
                };
                matrix[i * n + j] = 1.0 - (normalized_diff / std::f64::consts::PI);
            }
        }

        matrix
    }

    fn compute_phase_coherence(&self, phases: &[f64]) -> f64 {
        if phases.is_empty() {
            return 0.0;
        }

        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        // Order parameter: r = |<e^(iθ)>|
        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }

    fn compute_entanglement(&self, amplitudes: &[(f64, f64)]) -> f64 {
        // Simplified von Neumann entropy: S = -sum(p_i * log(p_i))
        // where p_i = |ψ_i|^2
        let mut entropy = 0.0;

        for (re, im) in amplitudes {
            let probability = re * re + im * im;
            if probability > 1e-10 {
                entropy -= probability * probability.ln();
            }
        }

        entropy
    }
}
