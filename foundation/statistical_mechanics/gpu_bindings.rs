//! GPU FFI bindings for thermodynamic network CUDA kernels
//!
//! Constitution: Phase 1, Task 1.3 - GPU Acceleration
//!
//! These bindings provide Rust access to CUDA kernels for:
//! - Langevin dynamics evolution
//! - Entropy calculation
//! - Energy calculation
//! - Phase coherence calculation

use anyhow::{Result, Context};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::*;

/// GPU-accelerated thermodynamic network
#[cfg(feature = "cuda")]
pub struct GpuThermodynamicNetwork {
    device: Arc<CudaContext>,
    module: Arc<CudaModule>,

    // Device memory
    d_phases: CudaSlice<f64>,
    d_velocities: CudaSlice<f64>,
    d_natural_frequencies: CudaSlice<f64>,
    d_coupling_matrix: CudaSlice<f64>,
    d_new_phases: CudaSlice<f64>,
    d_new_velocities: CudaSlice<f64>,
    d_forces: CudaSlice<f64>,
    d_rng_states: CudaSlice<u64>,

    // Reduction buffers
    d_entropy: CudaSlice<f64>,
    d_energy: CudaSlice<f64>,
    d_coherence_real: CudaSlice<f64>,
    d_coherence_imag: CudaSlice<f64>,

    n_oscillators: usize,
}


#[cfg(feature = "cuda")]
impl GpuThermodynamicNetwork {
    /// Create new GPU-accelerated network
    pub fn new(
        n_oscillators: usize,
        phases: &[f64],
        velocities: &[f64],
        natural_frequencies: &[f64],
        coupling_matrix: &[f64],
        seed: u64,
    ) -> Result<Self> {
        // Initialize CUDA device
        let device = CudaContext::new(0)?;

        // Load PTX module
        let ptx_path = "target/ptx/thermodynamic_evolution.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to load PTX from {}. Run: cargo build --release", ptx_path))?;

        let module = device.load_module(ptx.into())?;

        // Get stream for uploads
        let stream = device.default_stream();

        // Upload initial state to GPU
        let d_phases = stream.memcpy_stod(phases)?;
        let d_velocities = stream.memcpy_stod(velocities)?;
        let d_natural_frequencies = stream.memcpy_stod(natural_frequencies)?;
        let d_coupling_matrix = stream.memcpy_stod(coupling_matrix)?;

        // Allocate working buffers
        let d_new_phases = stream.alloc_zeros::<f64>(n_oscillators)?;
        let d_new_velocities = stream.alloc_zeros::<f64>(n_oscillators)?;
        let d_forces = stream.alloc_zeros::<f64>(n_oscillators)?;

        // Initialize RNG states
        let rng_states = Self::init_rng_states_cpu(n_oscillators, seed);
        let d_rng_states = stream.memcpy_stod(&rng_states)?;

        // Allocate reduction buffers
        let d_entropy = stream.alloc_zeros::<f64>(1)?;
        let d_energy = stream.alloc_zeros::<f64>(1)?;
        let d_coherence_real = stream.alloc_zeros::<f64>(1)?;
        let d_coherence_imag = stream.alloc_zeros::<f64>(1)?;

        Ok(Self {
            device,
            module,
            d_phases,
            d_velocities,
            d_natural_frequencies,
            d_coupling_matrix,
            d_new_phases,
            d_new_velocities,
            d_forces,
            d_rng_states,
            d_entropy,
            d_energy,
            d_coherence_real,
            d_coherence_imag,
            n_oscillators,
        })
    }

    /// Initialize RNG states on CPU (simple LCG)
    fn init_rng_states_cpu(n: usize, seed: u64) -> Vec<u64> {
        let mut states = vec![seed; n];
        for i in 0..n {
            // Linear congruential generator
            states[i] = states[i].wrapping_mul(1664525).wrapping_add(1013904223 + i as u64);
        }
        states
    }

    /// Execute one Langevin dynamics step on GPU
    pub fn step_gpu(
        &mut self,
        dt: f64,
        damping: f64,
        temperature: f64,
        coupling_strength: f64,
    ) -> Result<()> {
        let kernel = self.module.load_function("langevin_step_kernel")?;
        let stream = self.device.default_stream();

        let n_i32 = self.n_oscillators as i32;

        let mut launcher = stream.launch_builder(&kernel);
        launcher.arg(&self.d_phases);
        launcher.arg(&self.d_velocities);
        launcher.arg(&self.d_natural_frequencies);
        launcher.arg(&self.d_coupling_matrix);
        launcher.arg(&self.d_new_phases);
        launcher.arg(&self.d_new_velocities);
        launcher.arg(&self.d_forces);
        launcher.arg(&n_i32);
        launcher.arg(&dt);
        launcher.arg(&damping);
        launcher.arg(&temperature);
        launcher.arg(&coupling_strength);
        launcher.arg(&self.d_rng_states);

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            launcher.launch(cfg)?;
        }

        // Don't synchronize here - let the next operation do it implicitly
        // stream.synchronize()?;

        // Swap buffers
        std::mem::swap(&mut self.d_phases, &mut self.d_new_phases);
        std::mem::swap(&mut self.d_velocities, &mut self.d_new_velocities);

        Ok(())
    }

    /// Calculate entropy on GPU
    pub fn calculate_entropy_gpu(&mut self, temperature: f64) -> Result<f64> {
        let stream = self.device.default_stream();

        // Zero output buffer
        stream.memset_zeros(&mut self.d_entropy)?;

        let kernel = self.module.load_function("calculate_entropy_kernel")?;

        let n_i32 = self.n_oscillators as i32;

        let mut launcher = stream.launch_builder(&kernel);
        launcher.arg(&self.d_velocities);
        launcher.arg(&n_i32);
        launcher.arg(&temperature);
        launcher.arg(&self.d_entropy);

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            launcher.launch(cfg)?;
        }
        stream.synchronize()?;

        // Copy result back to host
        let mut entropy_host = vec![0.0f64];
        stream.memcpy_dtoh(&self.d_entropy, &mut entropy_host)?;
        Ok(entropy_host[0])
    }

    /// Calculate energy on GPU
    pub fn calculate_energy_gpu(&mut self, coupling_strength: f64) -> Result<f64> {
        let stream = self.device.default_stream();
        stream.memset_zeros(&mut self.d_energy)?;

        let kernel = self.module.load_function("calculate_energy_kernel")?;

        let n_i32 = self.n_oscillators as i32;

        let mut launcher = stream.launch_builder(&kernel);
        launcher.arg(&self.d_phases);
        launcher.arg(&self.d_velocities);
        launcher.arg(&self.d_coupling_matrix);
        launcher.arg(&n_i32);
        launcher.arg(&coupling_strength);
        launcher.arg(&self.d_energy);

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            launcher.launch(cfg)?;
        }
        stream.synchronize()?;

        let mut energy_host = vec![0.0f64];
        stream.memcpy_dtoh(&self.d_energy, &mut energy_host)?;
        Ok(energy_host[0])
    }

    /// Calculate phase coherence on GPU
    pub fn calculate_coherence_gpu(&mut self) -> Result<f64> {
        let stream = self.device.default_stream();
        stream.memset_zeros(&mut self.d_coherence_real)?;
        stream.memset_zeros(&mut self.d_coherence_imag)?;

        let kernel = self.module.load_function("calculate_coherence_kernel")?;

        let n_i32 = self.n_oscillators as i32;

        let mut launcher = stream.launch_builder(&kernel);
        launcher.arg(&self.d_phases);
        launcher.arg(&n_i32);
        launcher.arg(&self.d_coherence_real);
        launcher.arg(&self.d_coherence_imag);

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = 2 * threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            launcher.launch(cfg)?;
        }
        stream.synchronize()?;

        let mut real_host = vec![0.0f64];
        let mut imag_host = vec![0.0f64];
        stream.memcpy_dtoh(&self.d_coherence_real, &mut real_host)?;
        stream.memcpy_dtoh(&self.d_coherence_imag, &mut imag_host)?;

        let magnitude = (real_host[0] * real_host[0] + imag_host[0] * imag_host[0]).sqrt();
        Ok(magnitude / self.n_oscillators as f64)
    }

    /// Get current phases from GPU
    pub fn get_phases(&self) -> Result<Vec<f64>> {
        let stream = self.device.default_stream();
        let mut phases = vec![0.0f64; self.n_oscillators];
        stream.memcpy_dtoh(&self.d_phases, &mut phases)?;
        Ok(phases)
    }

    /// Get current velocities from GPU
    pub fn get_velocities(&self) -> Result<Vec<f64>> {
        let stream = self.device.default_stream();
        let mut velocities = vec![0.0f64; self.n_oscillators];
        stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        Ok(velocities)
    }

    /// Update coupling matrix on GPU
    pub fn update_coupling_matrix(&mut self, coupling_matrix: &[f64]) -> Result<()> {
        let stream = self.device.default_stream();
        let new_coupling = stream.memcpy_stod(coupling_matrix)?;
        stream.memcpy_dtod(&new_coupling, &mut self.d_coupling_matrix)?;
        stream.synchronize()?;
        Ok(())
    }
}

