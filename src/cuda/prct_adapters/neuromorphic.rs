//! Neuromorphic Adapter
//!
//! Implements NeuromorphicPort for the PRCT algorithm.
//! Converts graphs to spike patterns and processes them through reservoir computing.

use crate::cuda::prct_gpu::PRCTGpuManager;
use prct_core::errors::Result;
use prct_core::ports::{NeuromorphicEncodingParams, NeuromorphicPort};
use shared_types::*;
use std::sync::Arc;

/// Neuromorphic processing adapter
pub struct NeuromorphicAdapter {
    base_frequency: f64,
    time_window: f64,
    gpu_manager: Option<Arc<PRCTGpuManager>>,
}

impl NeuromorphicAdapter {
    pub fn new() -> Result<Self> {
        // Try to initialize GPU, fall back to CPU if unavailable
        let gpu_manager = PRCTGpuManager::new().ok().map(Arc::new);

        if gpu_manager.is_some() {
            log::info!("[NEUROMORPHIC] GPU acceleration enabled");
        } else {
            log::warn!("[NEUROMORPHIC] GPU unavailable, using CPU fallback");
        }

        Ok(Self {
            base_frequency: 20.0, // Hz
            time_window: 100.0,   // ms
            gpu_manager,
        })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        let n = graph.num_vertices;

        // Convert graph structure to spike trains
        // Each vertex generates spikes based on its degree
        let mut spikes = Vec::new();

        for i in 0..n {
            // Count neighbors (degree)
            let degree = (0..n)
                .filter(|&j| j != i && graph.adjacency[i * n + j])
                .count();

            // Generate spikes proportional to degree
            let num_spikes = (degree as f64 / n as f64 * 10.0).ceil() as usize;

            for spike_idx in 0..num_spikes {
                let time_ms = (spike_idx as f64 / params.base_frequency) * 1000.0;
                if time_ms < params.time_window {
                    spikes.push(Spike {
                        neuron_id: i,
                        time_ms,
                        amplitude: (degree as f64 / n as f64) * 100.0, // mV
                    });
                }
            }
        }

        Ok(SpikePattern {
            spikes,
            duration_ms: params.time_window,
            num_neurons: n,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        let n = spikes.num_neurons;

        // Try GPU acceleration first
        let (mut neuron_states, spike_pattern) = if let Some(gpu) = &self.gpu_manager {
            // Extract spike data for GPU
            let spike_ids: Vec<i32> = spikes.spikes.iter().map(|s| s.neuron_id as i32).collect();
            let spike_amps: Vec<f64> = spikes.spikes.iter().map(|s| s.amplitude).collect();

            match gpu.process_spikes_gpu(&spike_ids, &spike_amps, n) {
                Ok((states, counts)) => {
                    log::debug!("[NEUROMORPHIC] GPU spike processing succeeded");
                    // Convert i32 counts to u8
                    let counts_u8 = counts.iter().map(|&c| c.min(255) as u8).collect();
                    (states, counts_u8)
                }
                Err(e) => {
                    log::warn!(
                        "[NEUROMORPHIC] GPU processing failed: {}, falling back to CPU",
                        e
                    );
                    self.process_spikes_cpu(spikes, n)?
                }
            }
        } else {
            // CPU fallback
            self.process_spikes_cpu(spikes, n)?
        };

        // Normalize states on GPU if available
        if let Some(gpu) = &self.gpu_manager {
            if let Err(e) = gpu.normalize_states_gpu(&mut neuron_states) {
                log::warn!("[NEUROMORPHIC] GPU normalization failed: {}, using CPU", e);
                self.normalize_cpu(&mut neuron_states);
            }
        } else {
            self.normalize_cpu(&mut neuron_states);
        }

        // Compute coherence on GPU if available
        let coherence = if let Some(gpu) = &self.gpu_manager {
            match gpu.compute_coherence_gpu(&neuron_states) {
                Ok(coh) => {
                    log::debug!("[NEUROMORPHIC] GPU coherence: {:.6}", coh);
                    coh
                }
                Err(e) => {
                    log::warn!("[NEUROMORPHIC] GPU coherence failed: {}, using CPU", e);
                    self.compute_coherence(&neuron_states)
                }
            }
        } else {
            self.compute_coherence(&neuron_states)
        };

        // Compute pattern strength
        let pattern_strength = neuron_states.iter().map(|&x| x * x).sum::<f64>() / n as f64;

        Ok(NeuroState {
            neuron_states,
            spike_pattern,
            coherence,
            pattern_strength,
            timestamp_ns: 0,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        // Simplified: no patterns detected yet
        Ok(Vec::new())
    }
}

impl NeuromorphicAdapter {
    /// CPU fallback for spike processing
    fn process_spikes_cpu(&self, spikes: &SpikePattern, n: usize) -> Result<(Vec<f64>, Vec<u8>)> {
        let mut neuron_states = vec![0.0; n];
        let mut spike_pattern = vec![0u8; n];

        for spike in &spikes.spikes {
            if spike.neuron_id < n {
                neuron_states[spike.neuron_id] += spike.amplitude;
                if spike_pattern[spike.neuron_id] < 255 {
                    spike_pattern[spike.neuron_id] += 1;
                }
            }
        }

        Ok((neuron_states, spike_pattern))
    }

    /// CPU fallback for normalization
    fn normalize_cpu(&self, states: &mut [f64]) {
        let max_state = states.iter().cloned().fold(0.0f64, f64::max);
        if max_state > 0.0 {
            for state in states.iter_mut() {
                *state /= max_state;
            }
        }
    }

    /// CPU fallback for coherence computation
    fn compute_coherence(&self, states: &[f64]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }

        // Compute as order parameter: r = |<e^(iθ)>|
        let n = states.len() as f64;
        let phases: Vec<f64> = states
            .iter()
            .map(|&s| s * 2.0 * std::f64::consts::PI)
            .collect();

        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }
}
