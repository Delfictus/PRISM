//! GPU Policy Evaluation
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context
//! - Article VI: Data stays on GPU during evaluation
//! - Article VII: PTX runtime loading (no FFI linking)
//!
//! Implements GPU-accelerated policy evaluation for Active Inference:
//! - Parallel trajectory prediction for multiple policies
//! - Hierarchical physics simulation on GPU
//! - Expected free energy computation
//!
//! Target: 231ms CPU → <10ms GPU (23x speedup)

use std::sync::Arc;
use ndarray::{Array1, Array2};
use anyhow::{Result, anyhow, Context};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::policy_selection::Policy;
use super::hierarchical_model::HierarchicalModel;

/// Dimensions for GPU policy evaluation
#[derive(Debug, Clone, Copy)]
pub struct StateDimensions {
    pub satellite: usize,    // 6 (position + velocity)
    pub atmosphere: usize,   // 50 (turbulence modes)
    pub windows: usize,      // 900 (phase array)
    pub observations: usize, // 100 (measurement space)
}

impl Default for StateDimensions {
    fn default() -> Self {
        Self {
            satellite: 6,
            atmosphere: 50,
            windows: 900,
            observations: 100,
        }
    }
}

/// GPU buffers for trajectory storage
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct GpuTrajectoryBuffers {
    // Persistent allocations for trajectories
    satellite_states: CudaSlice<f64>,    // [n_policies × horizon × 6]
    atmosphere_states: CudaSlice<f64>,   // [n_policies × horizon × 50]
    window_states: CudaSlice<f64>,       // [n_policies × horizon × 900]
    variances: CudaSlice<f64>,           // [n_policies × horizon × 900]

    // Observations
    predicted_obs: CudaSlice<f64>,       // [n_policies × horizon × 100]
    obs_variances: CudaSlice<f64>,       // [n_policies × horizon × 100]
}


/// GPU buffers for EFE computation
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct GpuEfeBuffers {
    risk: CudaSlice<f64>,                // [n_policies]
    ambiguity: CudaSlice<f64>,           // [n_policies]
    novelty: CudaSlice<f64>,             // [n_policies]
}


/// GPU buffers for model state and parameters
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct GpuModelBuffers {
    // Initial states (for all policies)
    initial_satellite: CudaSlice<f64>,     // [n_policies × 6]
    initial_atmosphere: CudaSlice<f64>,    // [n_policies × 50]
    initial_windows: CudaSlice<f64>,       // [n_policies × 900]
    initial_variances: CudaSlice<f64>,     // [n_policies × 900]

    // Policy actions
    actions: CudaSlice<f64>,               // [n_policies × horizon × 900]

    // Observation model
    observation_matrix: CudaSlice<f64>,    // [100 × 900]
    observation_noise: CudaSlice<f64>,     // [100]
    preferred_obs: CudaSlice<f64>,         // [100]

    // Prior (for novelty calculation)
    prior_variance: CudaSlice<f64>,        // [900]
}


/// GPU-accelerated policy evaluator
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GpuPolicyEvaluator {
    context: Arc<CudaContext>,

    // Kernels loaded from PTX
    satellite_kernel: Arc<CudaFunction>,
    atmosphere_kernel: Arc<CudaFunction>,
    windows_kernel: Arc<CudaFunction>,
    observation_kernel: Arc<CudaFunction>,
    efe_kernel: Arc<CudaFunction>,
    rng_init_kernel: Arc<CudaFunction>,

    // Persistent GPU buffers
    trajectories: GpuTrajectoryBuffers,
    efe_buffers: GpuEfeBuffers,
    model_buffers: GpuModelBuffers,

    // RNG states
    rng_states_atmosphere: CudaSlice<u8>,  // curandState is opaque
    rng_states_windows: CudaSlice<u8>,

    // Configuration
    n_policies: usize,
    horizon: usize,
    substeps: usize,  // For window evolution
    dims: StateDimensions,

    // Physics parameters
    damping: f64,
    diffusion: f64,
    decorrelation_rate: f64,
    c_n_squared: f64,
}


#[cfg(feature = "cuda")]
impl GpuPolicyEvaluator {
    /// Create new GPU policy evaluator
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (Article V compliance)
    /// * `n_policies` - Number of policies to evaluate in parallel (typically 5)
    /// * `horizon` - Planning horizon (typically 3)
    /// * `substeps` - Substeps for window evolution (typically 10)
    pub fn new(
        context: Arc<CudaContext>,
        n_policies: usize,
        horizon: usize,
        substeps: usize,
    ) -> Result<Self> {
        let dims = StateDimensions::default();

        println!("[GPU-POLICY] Initializing GPU policy evaluator");
        println!("[GPU-POLICY] Config: {} policies, {} horizon, {} substeps",
                 n_policies, horizon, substeps);

        // Load PTX module (Article VII compliance)
        let ptx_path = "target/ptx/policy_evaluation.ptx";

        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Policy evaluation PTX not found at: {}", ptx_path));
        }

        let ptx_data = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_data);
        let module = context.load_module(ptx)?;

        println!("[GPU-POLICY] PTX module loaded successfully");

        // Load kernel functions (names are C++ mangled)
        let satellite_kernel = Arc::new(
            module.load_function("_Z23evolve_satellite_kernelPKdPddi")
                .context("Failed to load satellite kernel")?
        );
        let atmosphere_kernel = Arc::new(
            module.load_function("_Z24evolve_atmosphere_kernelPKdS0_PdS1_P17curandStateXORWOWdddii")
                .context("Failed to load atmosphere kernel")?
        );
        let windows_kernel = Arc::new(
            module.load_function("_Z21evolve_windows_kernelPKdS0_S0_S0_PdS1_P17curandStateXORWOWdddiiiii")
                .context("Failed to load windows kernel")?
        );
        let observation_kernel = Arc::new(
            module.load_function("_Z27predict_observations_kernelPKdS0_S0_S0_PdS1_iiii")
                .context("Failed to load observation kernel")?
        );
        let efe_kernel = Arc::new(
            module.load_function("_Z18compute_efe_kernelPKdS0_S0_S0_S0_PdS1_S1_iiii")
                .context("Failed to load EFE kernel")?
        );
        let rng_init_kernel = Arc::new(
            module.load_function("_Z22init_rng_states_kernelP17curandStateXORWOWyi")
                .context("Failed to load RNG init kernel")?
        );

        println!("[GPU-POLICY] All 6 kernels loaded");

        // Allocate persistent GPU buffers
        let stream = context.default_stream();

        println!("[GPU-POLICY] Allocating GPU memory...");

        let trajectories = GpuTrajectoryBuffers {
            satellite_states: stream.alloc_zeros(n_policies * horizon * dims.satellite)?,
            atmosphere_states: stream.alloc_zeros(n_policies * horizon * dims.atmosphere)?,
            window_states: stream.alloc_zeros(n_policies * horizon * dims.windows)?,
            variances: stream.alloc_zeros(n_policies * horizon * dims.windows)?,
            predicted_obs: stream.alloc_zeros(n_policies * horizon * dims.observations)?,
            obs_variances: stream.alloc_zeros(n_policies * horizon * dims.observations)?,
        };

        let efe_buffers = GpuEfeBuffers {
            risk: stream.alloc_zeros(n_policies)?,
            ambiguity: stream.alloc_zeros(n_policies)?,
            novelty: stream.alloc_zeros(n_policies)?,
        };

        let model_buffers = GpuModelBuffers {
            initial_satellite: stream.alloc_zeros(n_policies * dims.satellite)?,
            initial_atmosphere: stream.alloc_zeros(n_policies * dims.atmosphere)?,
            initial_windows: stream.alloc_zeros(n_policies * dims.windows)?,
            initial_variances: stream.alloc_zeros(n_policies * dims.windows)?,
            actions: stream.alloc_zeros(n_policies * horizon * dims.windows)?,
            observation_matrix: stream.alloc_zeros(dims.observations * dims.windows)?,
            observation_noise: stream.alloc_zeros(dims.observations)?,
            preferred_obs: stream.alloc_zeros(dims.observations)?,
            prior_variance: stream.alloc_zeros(dims.windows)?,
        };

        // Allocate RNG states
        // curandState is 48 bytes per state
        let rng_state_size = 48;
        let rng_states_atmosphere = stream.alloc_zeros(
            n_policies * dims.atmosphere * rng_state_size
        )?;
        let rng_states_windows = stream.alloc_zeros(
            n_policies * dims.windows * rng_state_size
        )?;

        println!("[GPU-POLICY] GPU memory allocated successfully");
        println!("[GPU-POLICY] - Trajectories: ~{} MB",
                 (n_policies * horizon * (dims.satellite + dims.atmosphere + 2*dims.windows) * 8) / 1_000_000);
        println!("[GPU-POLICY] - Observations: ~{} KB",
                 (n_policies * horizon * 2 * dims.observations * 8) / 1_000);

        // Initialize RNG states
        let seed = 12345u64;
        Self::initialize_rng(&context, &rng_init_kernel, &rng_states_atmosphere,
                            n_policies * dims.atmosphere, seed)?;
        Self::initialize_rng(&context, &rng_init_kernel, &rng_states_windows,
                            n_policies * dims.windows, seed + 1)?;

        println!("[GPU-POLICY] RNG states initialized");

        // Physics parameters (from TransitionModel)
        let damping = 10.0;  // Hz (window phase damping)
        let diffusion = 0.1; // Diffusion coefficient
        let decorrelation_rate = 0.1; // 1/10s (atmospheric decorrelation)
        let c_n_squared = 1e-13; // Atmospheric structure constant

        Ok(Self {
            context,
            satellite_kernel,
            atmosphere_kernel,
            windows_kernel,
            observation_kernel,
            efe_kernel,
            rng_init_kernel,
            trajectories,
            efe_buffers,
            model_buffers,
            rng_states_atmosphere,
            rng_states_windows,
            n_policies,
            horizon,
            substeps,
            dims,
            damping,
            diffusion,
            decorrelation_rate,
            c_n_squared,
        })
    }

    /// Initialize RNG states
    fn initialize_rng(
        context: &Arc<CudaContext>,
        rng_kernel: &CudaFunction,
        rng_states: &CudaSlice<u8>,
        n_states: usize,
        seed: u64,
    ) -> Result<()> {
        let stream = context.default_stream();

        let threads = 256;
        let blocks = (n_states + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_states_i32 = n_states as i32;

        let mut launch = stream.launch_builder(rng_kernel);
        launch.arg(rng_states);
        launch.arg(&seed);
        launch.arg(&n_states_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;

        Ok(())
    }

    /// Evaluate policies on GPU
    ///
    /// # Returns
    /// Vector of expected free energy values (one per policy)
    pub fn evaluate_policies_gpu(
        &mut self,
        model: &HierarchicalModel,
        policies: &[Policy],
        observation_matrix: &Array2<f64>,
        preferred_obs: &Array1<f64>,
    ) -> Result<Vec<f64>> {
        let start_total = std::time::Instant::now();
        println!("[GPU-POLICY] ========================================");
        println!("[GPU-POLICY] Starting GPU policy evaluation");
        println!("[GPU-POLICY] Policies: {}, Horizon: {}", self.n_policies, self.horizon);

        if policies.len() != self.n_policies {
            return Err(anyhow!("Expected {} policies, got {}", self.n_policies, policies.len()));
        }

        let stream = self.context.default_stream();

        // Step 1: Upload initial state and matrices
        let upload_start = std::time::Instant::now();
        self.upload_initial_state(model)?;
        self.upload_policies(policies)?;
        self.upload_matrices(observation_matrix, preferred_obs)?;
        println!("[GPU-POLICY] Upload took {:?}", upload_start.elapsed());

        // Step 2: Predict trajectories for all policies
        let traj_start = std::time::Instant::now();
        self.predict_all_trajectories()?;
        println!("[GPU-POLICY] Trajectory prediction took {:?}", traj_start.elapsed());

        // Step 3: Predict observations at each future state
        let obs_start = std::time::Instant::now();
        self.predict_all_observations()?;
        println!("[GPU-POLICY] Observation prediction took {:?}", obs_start.elapsed());

        // Step 4: Compute EFE components
        let efe_start = std::time::Instant::now();
        self.compute_efe_components(model)?;
        println!("[GPU-POLICY] EFE computation took {:?}", efe_start.elapsed());

        // Step 5: Download results
        let download_start = std::time::Instant::now();
        let risk_vec = stream.memcpy_dtov(&self.efe_buffers.risk)?;
        let ambiguity_vec = stream.memcpy_dtov(&self.efe_buffers.ambiguity)?;
        let novelty_vec = stream.memcpy_dtov(&self.efe_buffers.novelty)?;
        println!("[GPU-POLICY] Download took {:?}", download_start.elapsed());

        // Debug: Print individual components
        println!("[GPU-POLICY] EFE Components:");
        for i in 0..self.n_policies {
            println!("[GPU-POLICY]   Policy {}: risk={:.6}, ambiguity={:.6}, novelty={:.6}",
                     i, risk_vec[i], ambiguity_vec[i], novelty_vec[i]);
        }

        // Compute total EFE: risk + ambiguity - novelty
        let efe_values: Vec<f64> = (0..self.n_policies)
            .map(|i| risk_vec[i] + ambiguity_vec[i] - novelty_vec[i])
            .collect();

        let total_elapsed = start_total.elapsed();
        println!("[GPU-POLICY] ========================================");
        println!("[GPU-POLICY] TOTAL GPU policy evaluation: {:?}", total_elapsed);
        println!("[GPU-POLICY] EFE values: {:?}", efe_values);
        println!("[GPU-POLICY] ========================================");

        Ok(efe_values)
    }

    /// Upload initial model state to GPU
    fn upload_initial_state(&mut self, model: &HierarchicalModel) -> Result<()> {
        let stream = self.context.default_stream();

        // Replicate initial state for all policies (they all start from same state)
        let mut satellite_replicated = Vec::with_capacity(self.n_policies * self.dims.satellite);
        let mut atmosphere_replicated = Vec::with_capacity(self.n_policies * self.dims.atmosphere);
        let mut windows_replicated = Vec::with_capacity(self.n_policies * self.dims.windows);
        let mut variances_replicated = Vec::with_capacity(self.n_policies * self.dims.windows);

        for _ in 0..self.n_policies {
            // Satellite state (6 dimensions)
            satellite_replicated.extend_from_slice(
                model.level3.belief.mean.as_slice()
                    .ok_or_else(|| anyhow!("Satellite state not contiguous"))?
            );

            // Atmosphere modes (50 dimensions)
            atmosphere_replicated.extend_from_slice(
                model.level2.belief.mean.as_slice()
                    .ok_or_else(|| anyhow!("Atmosphere state not contiguous"))?
            );

            // Window phases (900 dimensions)
            windows_replicated.extend_from_slice(
                model.level1.belief.mean.as_slice()
                    .ok_or_else(|| anyhow!("Window state not contiguous"))?
            );

            // Variances (900 dimensions)
            variances_replicated.extend_from_slice(
                model.level1.belief.variance.as_slice()
                    .ok_or_else(|| anyhow!("Window variance not contiguous"))?
            );
        }

        // Upload to GPU (re-create slices - cudarc doesn't have copy_host_to_device)
        self.model_buffers.initial_satellite = stream.memcpy_stod(&satellite_replicated)?;
        self.model_buffers.initial_atmosphere = stream.memcpy_stod(&atmosphere_replicated)?;
        self.model_buffers.initial_windows = stream.memcpy_stod(&windows_replicated)?;
        self.model_buffers.initial_variances = stream.memcpy_stod(&variances_replicated)?;

        // Upload prior variance (for novelty calculation)
        self.model_buffers.prior_variance = stream.memcpy_stod(
            model.level1.belief.variance.as_slice()
                .ok_or_else(|| anyhow!("Prior variance not contiguous"))?
        )?;

        println!("[GPU-POLICY] Initial state uploaded: {} policies", self.n_policies);

        Ok(())
    }

    /// Upload policy actions to GPU
    fn upload_policies(&mut self, policies: &[Policy]) -> Result<()> {
        let stream = self.context.default_stream();

        // Flatten all policy actions into single buffer
        // Shape: [n_policies × horizon × n_windows]
        let mut actions_flat = Vec::with_capacity(
            self.n_policies * self.horizon * self.dims.windows
        );

        for policy in policies {
            if policy.actions.len() != self.horizon {
                return Err(anyhow!("Policy {} has {} actions, expected {}",
                                  policy.id, policy.actions.len(), self.horizon));
            }

            for action in &policy.actions {
                // Extract phase_correction (n_windows dimensions)
                let phase_slice = action.phase_correction.as_slice()
                    .ok_or_else(|| anyhow!("Phase correction not contiguous"))?;

                // Pad or truncate to n_windows if needed
                if phase_slice.len() == self.dims.windows {
                    actions_flat.extend_from_slice(phase_slice);
                } else if phase_slice.len() < self.dims.windows {
                    // Pad with zeros
                    actions_flat.extend_from_slice(phase_slice);
                    actions_flat.resize(actions_flat.len() + self.dims.windows - phase_slice.len(), 0.0);
                } else {
                    // Truncate
                    actions_flat.extend_from_slice(&phase_slice[..self.dims.windows]);
                }
            }
        }

        // Upload to GPU
        self.model_buffers.actions = stream.memcpy_stod(&actions_flat)?;

        println!("[GPU-POLICY] Actions uploaded: {} policies × {} horizon = {} values",
                 self.n_policies, self.horizon, actions_flat.len());

        Ok(())
    }

    /// Upload observation matrix and preferred observations
    fn upload_matrices(
        &mut self,
        observation_matrix: &Array2<f64>,
        preferred_obs: &Array1<f64>,
    ) -> Result<()> {
        let stream = self.context.default_stream();

        // Observation matrix (100 × 900 or actual dimensions)
        let obs_shape = observation_matrix.shape();
        if obs_shape[0] != self.dims.observations || obs_shape[1] != self.dims.windows {
            return Err(anyhow!(
                "Observation matrix shape {:?} doesn't match expected ({}, {})",
                obs_shape, self.dims.observations, self.dims.windows
            ));
        }

        // Flatten in row-major order
        let obs_matrix_flat: Vec<f64> = observation_matrix.iter().cloned().collect();
        self.model_buffers.observation_matrix = stream.memcpy_stod(&obs_matrix_flat)?;

        // Preferred observations (100 dimensions)
        if preferred_obs.len() != self.dims.observations {
            return Err(anyhow!(
                "Preferred observations length {} doesn't match expected {}",
                preferred_obs.len(), self.dims.observations
            ));
        }

        self.model_buffers.preferred_obs = stream.memcpy_stod(
            preferred_obs.as_slice().ok_or_else(|| anyhow!("Preferred obs not contiguous"))?
        )?;

        // For now, use fixed observation noise (could be parameter)
        let obs_noise = vec![0.01; self.dims.observations];
        self.model_buffers.observation_noise = stream.memcpy_stod(&obs_noise)?;

        println!("[GPU-POLICY] Observation matrix uploaded: {} × {}",
                 self.dims.observations, self.dims.windows);

        Ok(())
    }

    /// Predict trajectories for all policies
    fn predict_all_trajectories(&mut self) -> Result<()> {
        println!("[GPU-POLICY] Starting trajectory prediction...");

        // For each timestep in horizon
        for step in 0..self.horizon {
            println!("[GPU-POLICY]   Step {}/{}", step + 1, self.horizon);

            // Evolve all 3 hierarchical levels
            self.evolve_satellite_step(step)?;
            self.evolve_atmosphere_step(step)?;
            self.evolve_windows_step(step)?;
        }

        Ok(())
    }

    /// Evolve satellite states for all policies (one timestep)
    fn evolve_satellite_step(&mut self, step: usize) -> Result<()> {
        let stream = self.context.default_stream();

        let cfg = LaunchConfig {
            grid_dim: (self.n_policies as u32, 1, 1),
            block_dim: (6, 1, 1),  // 6 state dimensions
            shared_mem_bytes: 0,
        };

        let dt_satellite = 1.0;  // 1 second timestep
        let n_policies_i32 = self.n_policies as i32;

        // For now: simplified - always use initial state as source
        // Full implementation would chain through trajectory steps
        let mut launch = stream.launch_builder(&self.satellite_kernel);
        launch.arg(&self.model_buffers.initial_satellite);
        launch.arg(&mut self.trajectories.satellite_states);
        launch.arg(&dt_satellite);
        launch.arg(&n_policies_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;
        println!("[GPU-POLICY]     Satellite evolution step {} complete", step);

        Ok(())
    }

    /// Evolve atmosphere states for all policies (one timestep)
    fn evolve_atmosphere_step(&mut self, step: usize) -> Result<()> {
        let stream = self.context.default_stream();

        let cfg = LaunchConfig {
            grid_dim: (self.n_policies as u32, 1, 1),
            block_dim: (self.dims.atmosphere as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dt_atmosphere = 1.0;  // 1 second
        let n_policies_i32 = self.n_policies as i32;
        let n_modes_i32 = self.dims.atmosphere as i32;

        // Simplified: use initial state as source
        let mut launch = stream.launch_builder(&self.atmosphere_kernel);
        launch.arg(&self.model_buffers.initial_atmosphere);
        launch.arg(&self.model_buffers.initial_variances);
        launch.arg(&mut self.trajectories.atmosphere_states);
        launch.arg(&mut self.trajectories.variances);
        launch.arg(&self.rng_states_atmosphere);
        launch.arg(&dt_atmosphere);
        launch.arg(&self.decorrelation_rate);
        launch.arg(&self.c_n_squared);
        launch.arg(&n_policies_i32);
        launch.arg(&n_modes_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;
        println!("[GPU-POLICY]     Atmosphere evolution step {} complete", step);

        Ok(())
    }

    /// Evolve window states for all policies (one timestep)
    fn evolve_windows_step(&mut self, step: usize) -> Result<()> {
        let stream = self.context.default_stream();

        // Grid over policies × horizon (kernel processes all steps at once for each policy)
        let cfg = LaunchConfig {
            grid_dim: ((self.n_policies * self.horizon) as u32, 1, 1),
            block_dim: (256, 1, 1),  // Chunked over 900 windows
            shared_mem_bytes: 0,
        };

        let dt_windows = 0.01;  // 10ms
        let n_policies_i32 = self.n_policies as i32;
        let horizon_i32 = self.horizon as i32;
        let n_windows_i32 = self.dims.windows as i32;
        let n_modes_i32 = self.dims.atmosphere as i32;
        let substeps_i32 = self.substeps as i32;

        // Simplified: use initial state as source
        let mut launch = stream.launch_builder(&self.windows_kernel);
        launch.arg(&self.model_buffers.initial_windows);
        launch.arg(&self.model_buffers.initial_variances);
        launch.arg(&self.trajectories.atmosphere_states);
        launch.arg(&self.model_buffers.actions);
        launch.arg(&mut self.trajectories.window_states);
        launch.arg(&mut self.trajectories.variances);
        launch.arg(&self.rng_states_windows);
        launch.arg(&dt_windows);
        launch.arg(&self.damping);
        launch.arg(&self.diffusion);
        launch.arg(&n_policies_i32);
        launch.arg(&horizon_i32);
        launch.arg(&n_windows_i32);
        launch.arg(&n_modes_i32);
        launch.arg(&substeps_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;
        println!("[GPU-POLICY]     Window evolution step {} complete ({} substeps)",
                 step, self.substeps);

        Ok(())
    }

    /// Predict observations for all future states
    fn predict_all_observations(&mut self) -> Result<()> {
        let stream = self.context.default_stream();

        // One kernel launch for all policies × horizon
        let cfg = LaunchConfig {
            grid_dim: ((self.n_policies * self.horizon) as u32, 1, 1),
            block_dim: (self.dims.observations as u32, 1, 1),  // 100 threads
            shared_mem_bytes: 0,
        };

        let n_policies_i32 = self.n_policies as i32;
        let horizon_i32 = self.horizon as i32;
        let n_windows_i32 = self.dims.windows as i32;
        let obs_dim_i32 = self.dims.observations as i32;

        let mut launch = stream.launch_builder(&self.observation_kernel);
        launch.arg(&self.trajectories.window_states);
        launch.arg(&self.trajectories.variances);
        launch.arg(&self.model_buffers.observation_matrix);
        launch.arg(&self.model_buffers.observation_noise);
        launch.arg(&mut self.trajectories.predicted_obs);
        launch.arg(&mut self.trajectories.obs_variances);
        launch.arg(&n_policies_i32);
        launch.arg(&horizon_i32);
        launch.arg(&n_windows_i32);
        launch.arg(&obs_dim_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;
        println!("[GPU-POLICY] Observations predicted for {} states",
                 self.n_policies * self.horizon);

        Ok(())
    }

    /// Compute EFE components for all policies
    fn compute_efe_components(&mut self, model: &HierarchicalModel) -> Result<()> {
        let stream = self.context.default_stream();

        // Zero out EFE buffers before accumulation
        let zeros_risk = vec![0.0; self.n_policies];
        let zeros_ambiguity = vec![0.0; self.n_policies];
        let zeros_novelty = vec![0.0; self.n_policies];
        self.efe_buffers.risk = stream.memcpy_stod(&zeros_risk)?;
        self.efe_buffers.ambiguity = stream.memcpy_stod(&zeros_ambiguity)?;
        self.efe_buffers.novelty = stream.memcpy_stod(&zeros_novelty)?;

        let cfg = LaunchConfig {
            grid_dim: (self.n_policies as u32, 1, 1),
            block_dim: (256, 1, 1),  // Parallel reduction
            shared_mem_bytes: 0,
        };

        let n_policies_i32 = self.n_policies as i32;
        let horizon_i32 = self.horizon as i32;
        let obs_dim_i32 = self.dims.observations as i32;
        let n_windows_i32 = self.dims.windows as i32;

        let mut launch = stream.launch_builder(&self.efe_kernel);
        launch.arg(&self.trajectories.predicted_obs);
        launch.arg(&self.trajectories.obs_variances);
        launch.arg(&self.model_buffers.preferred_obs);
        launch.arg(&self.trajectories.variances);
        launch.arg(&self.model_buffers.prior_variance);
        launch.arg(&mut self.efe_buffers.risk);
        launch.arg(&mut self.efe_buffers.ambiguity);
        launch.arg(&mut self.efe_buffers.novelty);
        launch.arg(&n_policies_i32);
        launch.arg(&horizon_i32);
        launch.arg(&obs_dim_i32);
        launch.arg(&n_windows_i32);
        unsafe { launch.launch(cfg)?; }

        stream.synchronize()?;
        println!("[GPU-POLICY] EFE components computed for {} policies", self.n_policies);

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_policy_evaluator_creation() {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaContext;
            if let Ok(context) = CudaContext::new(0) {
                let evaluator = GpuPolicyEvaluator::new(
                    Arc::new(context),
                    5,   // n_policies
                    3,   // horizon
                    10,  // substeps
                );

                assert!(evaluator.is_ok(), "GPU policy evaluator should initialize");

                if let Ok(eval) = evaluator {
                    println!("GPU policy evaluator created successfully");
                    println!("Memory allocated for {} policies, {} horizon",
                             eval.n_policies, eval.horizon);
                }
            } else {
                println!("CUDA not available, skipping GPU test");
            }
        }

            }

    #[test]
    fn test_dimensions() {
        let dims = StateDimensions::default();
        assert_eq!(dims.satellite, 6);
        assert_eq!(dims.atmosphere, 50);
        assert_eq!(dims.windows, 900);
        assert_eq!(dims.observations, 100);
    }
}
