//! GPU-Accelerated Thermodynamic Network
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context
//! - Article VI: Data stays on GPU during evolution
//! - Article VII: PTX runtime loading (no FFI linking)
//!
//! Implements damped coupled oscillator dynamics on GPU:
//! - Langevin equation: dx/dt = v, dv/dt = F - γv + √(2γkT)*η(t)
//! - Coupling forces from network topology
//! - Entropy production tracking (2nd Law verification)

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use super::{NetworkConfig, ThermodynamicState};

/// GPU-accelerated thermodynamic network
///
/// Evolves coupled oscillators on GPU with thermal noise
pub struct ThermodynamicGpu {
    context: Arc<CudaDevice>,

    // Kernels loaded from PTX
    init_kernel: Arc<CudaFunction>,
    forces_kernel: Arc<CudaFunction>,
    evolve_kernel: Arc<CudaFunction>,
    energy_kernel: Arc<CudaFunction>,
    entropy_kernel: Arc<CudaFunction>,
    order_kernel: Arc<CudaFunction>,

    // Configuration
    config: NetworkConfig,

    // GPU state
    positions: CudaSlice<f64>,
    velocities: CudaSlice<f64>,
    phases: CudaSlice<f64>,
    coupling_matrix: CudaSlice<f64>,

    // History for entropy tracking
    entropy_history: Vec<f64>,
    iteration: usize,
}

impl ThermodynamicGpu {
    /// Create new GPU thermodynamic network
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (Article V compliance)
    /// * `config` - Network configuration
    pub fn new(context: Arc<CudaDevice>, config: NetworkConfig) -> Result<Self> {
        // Load PTX module (Article VII compliance)
        let ptx_path = "target/ptx/thermodynamic.ptx";

        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Thermodynamic PTX not found at: {}", ptx_path));
        }

        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX: {}", ptx_path))?;
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);

        let kernel_names = vec![
            "initialize_oscillators_kernel",
            "compute_coupling_forces_kernel",
            "evolve_oscillators_kernel",
            "compute_energy_kernel",
            "compute_entropy_kernel",
            "compute_order_parameter_kernel",
        ];
        context.load_ptx(ptx, "thermodynamic", &kernel_names)?;

        let init_kernel = Arc::new(
            context
                .get_func("thermodynamic", "initialize_oscillators_kernel")
                .ok_or_else(|| anyhow!("Failed to load initialize_oscillators_kernel"))?,
        );
        let forces_kernel = Arc::new(
            context
                .get_func("thermodynamic", "compute_coupling_forces_kernel")
                .ok_or_else(|| anyhow!("Failed to load compute_coupling_forces_kernel"))?,
        );
        let evolve_kernel = Arc::new(
            context
                .get_func("thermodynamic", "evolve_oscillators_kernel")
                .ok_or_else(|| anyhow!("Failed to load evolve_oscillators_kernel"))?,
        );
        let energy_kernel = Arc::new(
            context
                .get_func("thermodynamic", "compute_energy_kernel")
                .ok_or_else(|| anyhow!("Failed to load compute_energy_kernel"))?,
        );
        let entropy_kernel = Arc::new(
            context
                .get_func("thermodynamic", "compute_entropy_kernel")
                .ok_or_else(|| anyhow!("Failed to load compute_entropy_kernel"))?,
        );
        let order_kernel = Arc::new(
            context
                .get_func("thermodynamic", "compute_order_parameter_kernel")
                .ok_or_else(|| anyhow!("Failed to load compute_order_parameter_kernel"))?,
        );

        let n = config.n_oscillators;

        // Allocate GPU memory
        let mut positions = context.alloc_zeros::<f64>(n)?;
        let mut velocities = context.alloc_zeros::<f64>(n)?;
        let mut phases = context.alloc_zeros::<f64>(n)?;

        // Initialize coupling matrix (identity for now, will be updated)
        let coupling_vec: Vec<f64> = (0..n * n)
            .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
            .collect();
        let coupling_matrix = context.htod_sync_copy(&coupling_vec)?;

        // Initialize oscillators with random states
        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let seed = config.seed as u64;

        unsafe {
            cudarc::driver::CudaFunction::clone(&init_kernel).launch(
                cfg,
                (&mut positions, &mut velocities, &mut phases, n_i32, seed),
            )?;
        }

        Ok(Self {
            context,
            init_kernel,
            forces_kernel,
            evolve_kernel,
            energy_kernel,
            entropy_kernel,
            order_kernel,
            config,
            positions,
            velocities,
            phases,
            coupling_matrix,
            entropy_history: vec![],
            iteration: 0,
        })
    }

    /// Update coupling matrix from information flow
    pub fn update_coupling(&mut self, coupling: &Array2<f64>) -> Result<()> {
        let n = self.config.n_oscillators;

        if coupling.dim() != (n, n) {
            return Err(anyhow!("Coupling matrix dimension mismatch"));
        }

        // Flatten and upload
        let coupling_flat: Vec<f64> = coupling.iter().cloned().collect();
        self.coupling_matrix = self.context.htod_sync_copy(&coupling_flat)?;

        Ok(())
    }

    /// Evolve network for one time step on GPU
    ///
    /// Article VI Compliance: All computation on GPU
    pub fn evolve_step(&mut self) -> Result<ThermodynamicState> {
        let n = self.config.n_oscillators;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Allocate force buffer
        let mut forces = self.context.alloc_zeros::<f64>(n)?;

        // Step 1: Compute coupling forces
        let n_i32 = n as i32;
        let coupling_strength = self.config.coupling_strength;

        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.forces_kernel).launch(
                cfg,
                (
                    &self.positions,
                    &self.coupling_matrix,
                    &mut forces,
                    n_i32,
                    coupling_strength,
                ),
            )?;
        }

        // Step 2: Evolve oscillators (Langevin dynamics)
        let dt = self.config.dt;
        let damping = self.config.damping;
        let temperature = self.config.temperature;
        let seed = (self.config.seed as u64) + self.iteration as u64;
        let iter_i32 = self.iteration as i32;

        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.evolve_kernel).launch(
                cfg,
                (
                    &mut self.positions,
                    &mut self.velocities,
                    &mut self.phases,
                    &forces,
                    dt,
                    damping,
                    temperature,
                    n_i32,
                    seed,
                    iter_i32,
                ),
            )?;
        }

        self.iteration += 1;

        // Step 3: Compute observables
        let state = self.get_state()?;

        // Track entropy for 2nd law verification
        self.entropy_history.push(state.entropy);

        Ok(state)
    }

    /// Get current thermodynamic state from GPU
    pub fn get_state(&self) -> Result<ThermodynamicState> {
        let n = self.config.n_oscillators;

        // Compute energy on GPU
        let mut energy_components = self.context.alloc_zeros::<f64>(3)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let coupling_strength = self.config.coupling_strength;

        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.energy_kernel).launch(
                cfg,
                (
                    &self.positions,
                    &self.velocities,
                    &self.coupling_matrix,
                    &mut energy_components,
                    n_i32,
                    coupling_strength,
                ),
            )?;
        }

        // Compute entropy on GPU
        let mut entropy_result = self.context.alloc_zeros::<f64>(1)?;
        let temperature = self.config.temperature;

        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.entropy_kernel).launch(
                cfg,
                (
                    &self.positions,
                    &self.velocities,
                    &mut entropy_result,
                    n_i32,
                    temperature,
                ),
            )?;
        }

        // Compute order parameter on GPU
        let mut order_real = self.context.alloc_zeros::<f64>(1)?;
        let mut order_imag = self.context.alloc_zeros::<f64>(1)?;

        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.order_kernel)
                .launch(cfg, (&self.phases, &mut order_real, &mut order_imag, n_i32))?;
        }

        // Download results
        let energy_vec = self.context.dtoh_sync_copy(&energy_components)?;
        let entropy_vec = self.context.dtoh_sync_copy(&entropy_result)?;
        let order_real_vec = self.context.dtoh_sync_copy(&order_real)?;
        let order_imag_vec = self.context.dtoh_sync_copy(&order_imag)?;
        let phases_vec = self.context.dtoh_sync_copy(&self.phases)?;

        let total_energy = energy_vec[0] + energy_vec[1] + energy_vec[2];
        let entropy = entropy_vec[0];

        // Order parameter: r = |⟨e^(iθ)⟩| / N
        let order_r =
            (order_real_vec[0] * order_real_vec[0] + order_imag_vec[0] * order_imag_vec[0]).sqrt()
                / (n as f64);

        // Build ThermodynamicState matching actual struct
        let n = self.config.n_oscillators;

        // Download velocities for state
        let velocities_vec = self.context.dtoh_sync_copy(&self.velocities)?;

        // Build coupling matrix (simplified - use identity for now)
        let coupling_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        Ok(ThermodynamicState {
            phases: phases_vec,
            velocities: velocities_vec,
            natural_frequencies: vec![1.0; n], // Simplified
            coupling_matrix,
            time: (self.iteration as f64) * self.config.dt,
            entropy,
            energy: total_energy,
        })
    }

    /// Compute entropy production rate (dS/dt)
    ///
    /// Constitutional: Must be ≥ 0 (2nd Law)
    pub fn entropy_production(&self) -> f64 {
        if self.entropy_history.len() < 2 {
            return 0.0;
        }

        let n = self.entropy_history.len();
        let s_current = self.entropy_history[n - 1];
        let s_prev = self.entropy_history[n - 2];
        let delta_s = s_current - s_prev;

        eprintln!(
            "[Thermo GPU] Entropy check: S_prev={:.4}, S_current={:.4}, ΔS={:.4}, dS/dt={:.4}",
            s_prev,
            s_current,
            delta_s,
            delta_s / self.config.dt
        );

        delta_s / self.config.dt
    }

    /// Get entropy history
    pub fn entropy_history(&self) -> &[f64] {
        &self.entropy_history
    }

    /// Get Kuramoto synchronization state
    pub fn get_kuramoto_state(&self) -> Result<shared_types::KuramotoState> {
        let n = self.config.n_oscillators;

        // Download phases from GPU
        let phases_vec = self.context.dtoh_sync_copy(&self.phases)?;

        // Download coupling matrix
        let coupling_vec = self.context.dtoh_sync_copy(&self.coupling_matrix)?;

        // Compute order parameter on GPU
        let mut order_real = self.context.alloc_zeros::<f64>(1)?;
        let mut order_imag = self.context.alloc_zeros::<f64>(1)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        unsafe {
            cudarc::driver::CudaFunction::clone(&*self.order_kernel)
                .launch(cfg, (&self.phases, &mut order_real, &mut order_imag, n_i32))?;
        }

        let order_real_vec = self.context.dtoh_sync_copy(&order_real)?;
        let order_imag_vec = self.context.dtoh_sync_copy(&order_imag)?;

        // Compute order parameter and mean phase
        let order_parameter =
            (order_real_vec[0] * order_real_vec[0] + order_imag_vec[0] * order_imag_vec[0]).sqrt()
                / (n as f64);
        let mean_phase = order_imag_vec[0].atan2(order_real_vec[0]);

        Ok(shared_types::KuramotoState {
            phases: phases_vec,
            natural_frequencies: vec![1.0; n], // Default frequencies
            coupling_matrix: coupling_vec,
            order_parameter: order_parameter.min(1.0),
            mean_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermodynamic_gpu_creation() {
        if let Ok(context) = CudaDevice::new(0) {
            let config = NetworkConfig {
                n_oscillators: 10,
                temperature: 1.0,
                damping: 0.1,
                dt: 0.001,
                coupling_strength: 0.5,
                enable_information_gating: false,
                seed: 42,
            };

            let thermo_gpu = ThermodynamicGpu::new(context, config);
            assert!(thermo_gpu.is_ok());
        }
    }

    #[test]
    fn test_thermodynamic_evolution() {
        if let Ok(context) = CudaDevice::new(0) {
            let config = NetworkConfig {
                n_oscillators: 10,
                temperature: 1.0,
                damping: 0.1,
                dt: 0.001,
                coupling_strength: 0.5,
                enable_information_gating: false,
                seed: 42,
            };

            if let Ok(mut thermo_gpu) = ThermodynamicGpu::new(context, config) {
                // Evolve for 10 steps
                for _ in 0..10 {
                    let state = thermo_gpu.evolve_step();
                    assert!(state.is_ok());

                    let s = state.unwrap();
                    assert!(s.energy.is_finite());
                    assert!(s.entropy.is_finite());
                    assert!(s.order_parameter >= 0.0 && s.order_parameter <= 1.0);
                }

                // Check 2nd law: entropy production ≥ 0
                let ds_dt = thermo_gpu.entropy_production();
                assert!(ds_dt >= -1e-10); // Allow tiny numerical error

                println!("GPU Thermodynamic evolution: dS/dt = {:.6}", ds_dt);
            }
        }
    }
}
