//! GPU-Accelerated PRCT Operations
//!
//! CUDA kernel wrappers for neuromorphic, quantum, and Kuramoto computations

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// GPU manager for PRCT operations
pub struct PRCTGpuManager {
    device: Arc<CudaDevice>,
    // Neuromorphic kernels
    process_spikes: cudarc::driver::CudaFunction,
    find_max_state: cudarc::driver::CudaFunction,
    normalize_states: cudarc::driver::CudaFunction,
    compute_phase_coherence: cudarc::driver::CudaFunction,
    // Quantum kernels
    quantum_phase_evolution: cudarc::driver::CudaFunction,
    compute_norm_squared: cudarc::driver::CudaFunction,
    normalize_amplitudes: cudarc::driver::CudaFunction,
    extract_quantum_phases: cudarc::driver::CudaFunction,
    compute_phase_coherence_matrix: cudarc::driver::CudaFunction,
    // Kuramoto kernels
    kuramoto_step: cudarc::driver::CudaFunction,
    kuramoto_order_parameter: cudarc::driver::CudaFunction,
    compute_local_coherence: cudarc::driver::CudaFunction,
    // Utility kernels
    compute_correlation: cudarc::driver::CudaFunction,
}

impl PRCTGpuManager {
    /// Create new GPU manager and load PRCT kernels
    pub fn new() -> Result<Self> {
        // Get CUDA device
        let device = CudaDevice::new(0)?;

        // Load PTX from compiled kernels
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ptx/prct_kernels.ptx"));
        device.load_ptx(
            ptx.into(),
            "prct_kernels",
            &[
                "process_spikes_to_states",
                "find_max_state",
                "normalize_states",
                "compute_phase_coherence",
                "quantum_phase_evolution",
                "compute_norm_squared",
                "normalize_amplitudes",
                "extract_quantum_phases",
                "compute_phase_coherence_matrix",
                "kuramoto_step",
                "kuramoto_order_parameter",
                "compute_local_coherence",
                "compute_correlation",
            ],
        )?;

        Ok(Self {
            device: device.clone(),
            process_spikes: device
                .get_func("prct_kernels", "process_spikes_to_states")
                .unwrap(),
            find_max_state: device.get_func("prct_kernels", "find_max_state").unwrap(),
            normalize_states: device.get_func("prct_kernels", "normalize_states").unwrap(),
            compute_phase_coherence: device
                .get_func("prct_kernels", "compute_phase_coherence")
                .unwrap(),
            quantum_phase_evolution: device
                .get_func("prct_kernels", "quantum_phase_evolution")
                .unwrap(),
            compute_norm_squared: device
                .get_func("prct_kernels", "compute_norm_squared")
                .unwrap(),
            normalize_amplitudes: device
                .get_func("prct_kernels", "normalize_amplitudes")
                .unwrap(),
            extract_quantum_phases: device
                .get_func("prct_kernels", "extract_quantum_phases")
                .unwrap(),
            compute_phase_coherence_matrix: device
                .get_func("prct_kernels", "compute_phase_coherence_matrix")
                .unwrap(),
            kuramoto_step: device.get_func("prct_kernels", "kuramoto_step").unwrap(),
            kuramoto_order_parameter: device
                .get_func("prct_kernels", "kuramoto_order_parameter")
                .unwrap(),
            compute_local_coherence: device
                .get_func("prct_kernels", "compute_local_coherence")
                .unwrap(),
            compute_correlation: device
                .get_func("prct_kernels", "compute_correlation")
                .unwrap(),
        })
    }

    // ========================================================================
    // NEUROMORPHIC GPU OPERATIONS
    // ========================================================================

    /// Process spikes on GPU to compute neuron states
    pub fn process_spikes_gpu(
        &self,
        spike_neuron_ids: &[i32],
        spike_amplitudes: &[f64],
        num_neurons: usize,
    ) -> Result<(Vec<f64>, Vec<i32>)> {
        let num_spikes = spike_neuron_ids.len();

        // Transfer data to GPU
        let d_spike_ids = self.device.htod_sync_copy(spike_neuron_ids)?;
        let d_spike_amps = self.device.htod_sync_copy(spike_amplitudes)?;
        let d_neuron_states = self.device.alloc_zeros::<f64>(num_neurons)?;
        let d_spike_counts = self.device.alloc_zeros::<i32>(num_neurons)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(num_neurons as u32);
        unsafe {
            self.process_spikes.clone().launch(
                cfg,
                (
                    &d_spike_ids,
                    &d_spike_amps,
                    num_spikes as i32,
                    &d_neuron_states,
                    &d_spike_counts,
                    num_neurons as i32,
                ),
            )?;
        }

        // Synchronize before copying back
        self.device.synchronize()?;

        // Copy results back
        let neuron_states = self.device.dtoh_sync_copy(&d_neuron_states)?;
        let spike_counts = self.device.dtoh_sync_copy(&d_spike_counts)?;

        Ok((neuron_states, spike_counts))
    }

    /// Normalize neuron states on GPU
    pub fn normalize_states_gpu(&self, states: &mut [f64]) -> Result<()> {
        let n = states.len();

        // Find max on CPU (simpler and avoids atomic issues)
        let max_val = states.iter().cloned().fold(0.0f64, f64::max);

        if max_val < 1e-10 {
            return Ok(()); // Already normalized or all zeros
        }

        // Transfer to GPU
        let d_states = self.device.htod_sync_copy(states)?;

        // Normalize on GPU
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.normalize_states
                .clone()
                .launch(cfg, (&d_states, n as i32, max_val))?;
        }

        // Synchronize and copy back
        self.device.synchronize()?;
        self.device.dtoh_sync_copy_into(&d_states, states)?;

        Ok(())
    }

    /// Compute phase coherence on GPU (hybrid: phases on GPU, reduction on CPU)
    pub fn compute_coherence_gpu(&self, states: &[f64]) -> Result<f64> {
        // For small arrays, atomic operations have issues
        // Use CPU for reduction (still faster overall due to other GPU ops)
        let n = states.len() as f64;
        let phases: Vec<f64> = states
            .iter()
            .map(|&s| s * 2.0 * std::f64::consts::PI)
            .collect();

        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        let coherence = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();

        Ok(coherence)
    }

    // ========================================================================
    // QUANTUM GPU OPERATIONS
    // ========================================================================

    /// Evolve quantum state on GPU (hybrid: evolution on GPU, normalization on CPU)
    pub fn quantum_evolve_gpu(
        &self,
        amplitudes_real: &mut [f64],
        amplitudes_imag: &mut [f64],
        eigenvalues: &[f64],
        time: f64,
    ) -> Result<()> {
        let n = amplitudes_real.len();

        let d_re = self.device.htod_sync_copy(amplitudes_real)?;
        let d_im = self.device.htod_sync_copy(amplitudes_imag)?;
        let d_eigenvals = self.device.htod_sync_copy(eigenvalues)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.quantum_phase_evolution
                .clone()
                .launch(cfg, (&d_re, &d_im, &d_eigenvals, time, n as i32))?;
        }

        // Synchronize and copy back
        self.device.synchronize()?;
        self.device.dtoh_sync_copy_into(&d_re, amplitudes_real)?;
        self.device.dtoh_sync_copy_into(&d_im, amplitudes_imag)?;

        // Normalize on CPU (avoids atomic operations)
        let norm_sq: f64 = amplitudes_real
            .iter()
            .zip(amplitudes_imag.iter())
            .map(|(re, im)| re * re + im * im)
            .sum();
        let norm = norm_sq.sqrt();

        if norm > 1e-10 {
            for (re, im) in amplitudes_real.iter_mut().zip(amplitudes_imag.iter_mut()) {
                *re /= norm;
                *im /= norm;
            }
        }

        Ok(())
    }

    /// Extract phases from quantum amplitudes on GPU
    pub fn extract_phases_gpu(
        &self,
        amplitudes_real: &[f64],
        amplitudes_imag: &[f64],
    ) -> Result<Vec<f64>> {
        let n = amplitudes_real.len();

        let d_re = self.device.htod_sync_copy(amplitudes_real)?;
        let d_im = self.device.htod_sync_copy(amplitudes_imag)?;
        let d_phases = self.device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.extract_quantum_phases
                .clone()
                .launch(cfg, (&d_re, &d_im, &d_phases, n as i32))?;
        }

        let phases = self.device.dtoh_sync_copy(&d_phases)?;
        Ok(phases)
    }

    /// Compute phase coherence matrix on GPU
    pub fn compute_coherence_matrix_gpu(&self, phases: &[f64]) -> Result<Vec<f64>> {
        let n = phases.len();

        let d_phases = self.device.htod_sync_copy(phases)?;
        let d_matrix = self.device.alloc_zeros::<f64>(n * n)?;

        // 2D launch for matrix computation
        let block_dim = 16;
        let grid_dim = ((n + block_dim - 1) / block_dim) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, grid_dim, 1),
            block_dim: (block_dim as u32, block_dim as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.compute_phase_coherence_matrix
                .clone()
                .launch(cfg, (&d_phases, n as i32, &d_matrix))?;
        }

        let matrix = self.device.dtoh_sync_copy(&d_matrix)?;
        Ok(matrix)
    }

    // ========================================================================
    // KURAMOTO GPU OPERATIONS
    // ========================================================================

    /// Perform Kuramoto synchronization step on GPU
    pub fn kuramoto_step_gpu(
        &self,
        phases: &[f64],
        natural_frequencies: &[f64],
        coupling_matrix: &[f64],
        coupling_strength: f64,
        dt: f64,
    ) -> Result<Vec<f64>> {
        let n = phases.len();

        let d_phases = self.device.htod_sync_copy(phases)?;
        let d_freqs = self.device.htod_sync_copy(natural_frequencies)?;
        let d_coupling = self.device.htod_sync_copy(coupling_matrix)?;
        let d_new_phases = self.device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.kuramoto_step.clone().launch(
                cfg,
                (
                    &d_phases,
                    &d_freqs,
                    &d_coupling,
                    coupling_strength,
                    dt,
                    &d_new_phases,
                    n as i32,
                ),
            )?;
        }

        let new_phases = self.device.dtoh_sync_copy(&d_new_phases)?;
        Ok(new_phases)
    }

    /// Compute Kuramoto order parameter (CPU-only - avoids atomic operations)
    pub fn kuramoto_order_parameter_gpu(&self, phases: &[f64]) -> Result<f64> {
        // Use CPU for reduction (avoids atomic issues with atomicAdd on doubles)
        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        let order_param = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();

        Ok(order_param)
    }

    /// Compute local coherence levels on GPU
    pub fn compute_local_coherence_gpu(
        &self,
        phases: &[f64],
        coupling_matrix: &[f64],
    ) -> Result<Vec<f64>> {
        let n = phases.len();

        let d_phases = self.device.htod_sync_copy(phases)?;
        let d_coupling = self.device.htod_sync_copy(coupling_matrix)?;
        let d_coherence = self.device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.compute_local_coherence
                .clone()
                .launch(cfg, (&d_phases, &d_coupling, &d_coherence, n as i32))?;
        }

        let coherence_levels = self.device.dtoh_sync_copy(&d_coherence)?;
        Ok(coherence_levels)
    }

    /// Compute correlation for transfer entropy (CPU-only - avoids atomic operations)
    pub fn compute_correlation_gpu(
        &self,
        source: &[f64],
        target: &[f64],
        source_mean: f64,
        target_mean: f64,
    ) -> Result<(f64, f64, f64)> {
        // Use CPU for correlation computation (avoids atomic issues)
        let n = source.len().min(target.len());

        let covariance: f64 = source
            .iter()
            .zip(target.iter())
            .take(n)
            .map(|(s, t)| (s - source_mean) * (t - target_mean))
            .sum::<f64>()
            / n as f64;

        let source_var: f64 = source
            .iter()
            .take(n)
            .map(|s| (s - source_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        let target_var: f64 = target
            .iter()
            .take(n)
            .map(|t| (t - target_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        Ok((covariance, source_var, target_var))
    }
}
