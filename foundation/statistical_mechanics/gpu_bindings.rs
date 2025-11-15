//! GPU FFI bindings for thermodynamic network CUDA kernels
//!
//! Constitution: Phase 1, Task 1.3 - GPU Acceleration
//!
//! These bindings provide Rust access to CUDA kernels for:
//! - Langevin dynamics evolution
//! - Entropy calculation
//! - Energy calculation
//! - Phase coherence calculation

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::*;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

/// GPU-accelerated thermodynamic network
#[cfg(feature = "cuda")]
pub struct GpuThermodynamicNetwork {
    device: Arc<CudaDevice>,
    kernels: HashMap<&'static str, CudaFunction>,

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

/// Parameters for Langevin dynamics kernel
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
struct LangevinParams {
    n: i32,
    dt: f64,
    damping: f64,
    temperature: f64,
    coupling_strength: f64,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for LangevinParams {}

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
        let device = CudaDevice::new(0)?;

        // Load PTX module
        let ptx_path = "target/ptx/thermodynamic_evolution.ptx";
        let ptx = std::fs::read_to_string(ptx_path).with_context(|| {
            format!(
                "Failed to load PTX from {}. Run: cargo build --release",
                ptx_path
            )
        })?;

        // Load PTX and get all kernel functions
        let kernel_names = [
            "langevin_step_kernel",
            "calculate_entropy_kernel",
            "calculate_energy_kernel",
            "calculate_coherence_kernel",
        ];
        device.load_ptx(ptx.into(), "thermodynamic_module", &kernel_names)?;

        let mut kernels = HashMap::new();
        for &name in &kernel_names {
            let func = device
                .get_func("thermodynamic_module", name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Kernel function '{}' not found in module 'thermodynamic_module'",
                        name
                    )
                })?;
            kernels.insert(name, func);
        }

        // Upload initial state to GPU
        let d_phases = device.htod_sync_copy(phases)?;
        let d_velocities = device.htod_sync_copy(velocities)?;
        let d_natural_frequencies = device.htod_sync_copy(natural_frequencies)?;
        let d_coupling_matrix = device.htod_sync_copy(coupling_matrix)?;

        // Allocate working buffers
        let d_new_phases = device.alloc_zeros::<f64>(n_oscillators)?;
        let d_new_velocities = device.alloc_zeros::<f64>(n_oscillators)?;
        let d_forces = device.alloc_zeros::<f64>(n_oscillators)?;

        // Initialize RNG states
        let rng_states = Self::init_rng_states_cpu(n_oscillators, seed);
        let d_rng_states = device.htod_sync_copy(&rng_states)?;

        // Allocate reduction buffers
        let d_entropy = device.alloc_zeros::<f64>(1)?;
        let d_energy = device.alloc_zeros::<f64>(1)?;
        let d_coherence_real = device.alloc_zeros::<f64>(1)?;
        let d_coherence_imag = device.alloc_zeros::<f64>(1)?;

        Ok(Self {
            device,
            kernels,
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
            states[i] = states[i]
                .wrapping_mul(1664525)
                .wrapping_add(1013904223 + i as u64);
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
        let kernel = &self.kernels["langevin_step_kernel"];
        let n_i32 = self.n_oscillators as i32;

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Pack scalar parameters into struct (reduces 13 params → 9 params)
        let params = LangevinParams {
            n: n_i32,
            dt,
            damping,
            temperature,
            coupling_strength,
        };

        unsafe {
            cudarc::driver::CudaFunction::clone(kernel).launch(
                cfg,
                (
                    &self.d_phases,
                    &self.d_velocities,
                    &self.d_natural_frequencies,
                    &self.d_coupling_matrix,
                    &mut self.d_new_phases,
                    &mut self.d_new_velocities,
                    &mut self.d_forces,
                    params,
                    &mut self.d_rng_states,
                ),
            )?;
        }

        // Swap buffers
        std::mem::swap(&mut self.d_phases, &mut self.d_new_phases);
        std::mem::swap(&mut self.d_velocities, &mut self.d_new_velocities);

        Ok(())
    }

    /// Calculate entropy on GPU
    pub fn calculate_entropy_gpu(&mut self, temperature: f64) -> Result<f64> {
        // Zero output buffer
        self.device.memset_zeros(&mut self.d_entropy)?;

        let kernel = &self.kernels["calculate_entropy_kernel"];
        let n_i32 = self.n_oscillators as i32;

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            kernel.clone().launch(
                cfg,
                (&self.d_velocities, n_i32, temperature, &mut self.d_entropy),
            )?;
        }
        self.device.synchronize()?;

        // Copy result back to host
        let entropy_host = self.device.dtoh_sync_copy(&self.d_entropy)?;
        Ok(entropy_host[0])
    }

    /// Calculate energy on GPU
    pub fn calculate_energy_gpu(&mut self, coupling_strength: f64) -> Result<f64> {
        self.device.memset_zeros(&mut self.d_energy)?;

        let kernel = &self.kernels["calculate_energy_kernel"];
        let n_i32 = self.n_oscillators as i32;

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            kernel.clone().launch(
                cfg,
                (
                    &self.d_phases,
                    &self.d_velocities,
                    &self.d_coupling_matrix,
                    n_i32,
                    coupling_strength,
                    &mut self.d_energy,
                ),
            )?;
        }
        self.device.synchronize()?;

        let energy_host = self.device.dtoh_sync_copy(&self.d_energy)?;
        Ok(energy_host[0])
    }

    /// Calculate phase coherence on GPU
    pub fn calculate_coherence_gpu(&mut self) -> Result<f64> {
        self.device.memset_zeros(&mut self.d_coherence_real)?;
        self.device.memset_zeros(&mut self.d_coherence_imag)?;

        let kernel = &self.kernels["calculate_coherence_kernel"];
        let n_i32 = self.n_oscillators as i32;

        let threads = 256;
        let blocks = (self.n_oscillators + threads - 1) / threads;
        let shared_mem = 2 * threads * std::mem::size_of::<f64>();

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            kernel.clone().launch(
                cfg,
                (
                    &self.d_phases,
                    n_i32,
                    &mut self.d_coherence_real,
                    &mut self.d_coherence_imag,
                ),
            )?;
        }
        self.device.synchronize()?;

        let real_host = self.device.dtoh_sync_copy(&self.d_coherence_real)?;
        let imag_host = self.device.dtoh_sync_copy(&self.d_coherence_imag)?;

        let magnitude = (real_host[0] * real_host[0] + imag_host[0] * imag_host[0]).sqrt();
        Ok(magnitude / self.n_oscillators as f64)
    }

    /// Get current phases from GPU
    pub fn get_phases(&self) -> Result<Vec<f64>> {
        let phases = self.device.dtoh_sync_copy(&self.d_phases)?;
        Ok(phases)
    }

    /// Get current velocities from GPU
    pub fn get_velocities(&self) -> Result<Vec<f64>> {
        let velocities = self.device.dtoh_sync_copy(&self.d_velocities)?;
        Ok(velocities)
    }

    /// Update coupling matrix on GPU
    pub fn update_coupling_matrix(&mut self, coupling_matrix: &[f64]) -> Result<()> {
        let new_coupling = self.device.htod_sync_copy(coupling_matrix)?;
        self.device
            .dtod_copy(&new_coupling, &mut self.d_coupling_matrix)?;
        self.device.synchronize()?;
        Ok(())
    }
}
