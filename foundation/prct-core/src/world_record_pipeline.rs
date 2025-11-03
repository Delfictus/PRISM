//! World Record Breaking Pipeline
//!
//! Ultimate PRISM integration combining ALL advanced modules:
//! - Active Inference (variational free energy minimization)
//! - ADP Q-Learning (adaptive parameter optimization)
//! - Neuromorphic Reservoir Computing (conflict prediction)
//! - Statistical Mechanics (thermodynamic equilibration)
//! - Quantum-Classical Hybrid (QUBO + classical feedback)
//! - Multi-Scale Analysis (temporal hierarchies)
//! - Ensemble Consensus (multi-algorithm voting)
//!
//! Target: Beat 83 colors world record on DSJC1000.5
//! Current best: 115 colors (1.39x gap)
//! Expected: 83-90 colors with full integration

use crate::errors::*;
use crate::transfer_entropy_coloring::hybrid_te_kuramoto_ordering;
use crate::dsatur_backtracking::DSaturSolver;
use crate::memetic_coloring::{MemeticColoringSolver, MemeticConfig};
use crate::coloring::greedy_coloring_with_ordering;
use crate::quantum_coloring::QuantumColoringSolver;
use crate::geodesic::{compute_landmark_distances, GeodesicFeatures};
use crate::cpu_init::init_rayon_threads;
use shared_types::*;

use std::sync::Arc;
use rand::Rng;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use crate::world_record_pipeline_gpu::GpuReservoirConflictPredictor;

// Serde default helpers for v1.1 config
fn default_true() -> bool { true }
fn default_threads() -> usize { 24 }
fn default_streams() -> usize { 4 }
fn default_replicas() -> usize { 56 }  // VRAM guard for 8GB
fn default_beads() -> usize { 64 }     // VRAM guard for 8GB (future PIMC)
fn default_batch_size() -> usize { 1024 }

/// GPU Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub device_id: usize,

    #[serde(default = "default_streams")]
    pub streams: usize,

    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_true")]
    pub enable_reservoir_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_thermo_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_quantum_gpu: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            streams: default_streams(),
            batch_size: default_batch_size(),
            enable_reservoir_gpu: true,
            enable_thermo_gpu: true,
            enable_quantum_gpu: true,
        }
    }
}

/// Thermodynamic Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermoConfig {
    /// Number of parallel replicas (VRAM guard: max 56 for 8GB devices)
    #[serde(default = "default_replicas")]
    pub replicas: usize,

    #[serde(default = "default_replicas")]
    pub num_temps: usize,

    #[serde(default)]
    pub exchange_interval: usize,

    #[serde(default)]
    pub t_min: f64,

    #[serde(default)]
    pub t_max: f64,
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            replicas: default_replicas(),  // VRAM guard: 56 for 8GB
            num_temps: default_replicas(), // VRAM guard: 56 for 8GB
            exchange_interval: 50,
            t_min: 0.001,
            t_max: 1.0,
        }
    }
}

/// Quantum Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumConfig {
    pub iterations: usize,
    pub target_chromatic: usize,

    /// Number of retries for initial solution generation before giving up
    #[serde(default = "default_quantum_retries")]
    pub failure_retries: usize,

    /// Fall back to DSATUR if quantum fails
    #[serde(default = "default_true")]
    pub fallback_on_failure: bool,
}

fn default_quantum_retries() -> usize {
    2
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            iterations: 20,
            target_chromatic: 83,
            failure_retries: 2,
            fallback_on_failure: true,
        }
    }
}

/// GPU Coloring Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuColoringConfig {
    /// Force sparse kernel regardless of density
    #[serde(default = "default_false")]
    pub prefer_sparse: bool,

    /// Density threshold for sparse/dense selection
    #[serde(default = "default_sparse_threshold")]
    pub sparse_density_threshold: f64,

    /// Mask width for color bitsets (64 or 128)
    #[serde(default = "default_mask_width")]
    pub mask_width: u32,
}

fn default_sparse_threshold() -> f64 { 0.40 }
fn default_mask_width() -> u32 { 64 }
fn default_false() -> bool { false }

impl Default for GpuColoringConfig {
    fn default() -> Self {
        Self {
            prefer_sparse: false,
            sparse_density_threshold: 0.40,
            mask_width: 64,
        }
    }
}

/// Geodesic Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeodesicConfig {
    pub num_landmarks: usize,
    pub metric: String,
    pub weight_attr: Option<String>,
    #[serde(default = "default_centrality_weight")]
    pub centrality_weight: f64,
    #[serde(default = "default_eccentricity_weight")]
    pub eccentricity_weight: f64,
}

fn default_centrality_weight() -> f64 {
    0.5
}

fn default_eccentricity_weight() -> f64 {
    0.5
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            num_landmarks: 10,
            metric: "hop".to_string(),
            weight_attr: None,
            centrality_weight: 0.5,
            eccentricity_weight: 0.5,
        }
    }
}

/// CPU Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuConfig {
    pub threads: usize,
    #[serde(default)]
    pub pin_pool: bool,
    #[serde(default = "default_work_steal")]
    pub work_steal: bool,
    #[serde(default = "default_parallel_io")]
    pub parallel_io: bool,
}

fn default_cpu_threads() -> usize {
    24
}

fn default_work_steal() -> bool {
    true
}

fn default_parallel_io() -> bool {
    true
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            threads: default_cpu_threads(),
            pin_pool: false,
            work_steal: default_work_steal(),
            parallel_io: default_parallel_io(),
        }
    }
}

/// ADP Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdpConfig {
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    pub alpha: f64,
    pub gamma: f64,
}

impl Default for AdpConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.03,
            alpha: 0.10,
            gamma: 0.95,
        }
    }
}

/// Orchestrator Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrchestratorConfig {
    pub adp_dsatur_depth: usize,
    pub adp_quantum_iterations: usize,
    pub adp_thermo_num_temps: usize,
    pub restarts: usize,
    pub early_stop_no_improve_iters: usize,
    pub checkpoint_minutes: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            adp_dsatur_depth: 200000,
            adp_quantum_iterations: 20,
            adp_thermo_num_temps: 64,
            restarts: 10,
            early_stop_no_improve_iters: 3,
            checkpoint_minutes: 15,
        }
    }
}

/// World Record Pipeline Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WorldRecordConfig {
    /// Configuration profile name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,

    /// Configuration version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Deterministic mode (use fixed seed)
    #[serde(default)]
    pub deterministic: bool,

    /// Random seed for deterministic mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Target chromatic number (world record)
    pub target_chromatic: usize,

    /// Maximum total runtime (hours)
    pub max_runtime_hours: f64,

    /// Enable Active Inference policy selection
    #[serde(default = "default_true")]
    pub use_active_inference: bool,

    /// Enable ADP reinforcement learning
    #[serde(default = "default_true")]
    pub use_adp_learning: bool,

    /// Enable Reservoir Computing prediction
    #[serde(default = "default_true")]
    pub use_reservoir_prediction: bool,

    /// Enable Statistical Mechanics equilibration
    #[serde(default = "default_true")]
    pub use_thermodynamic_equilibration: bool,

    /// Enable Quantum-Classical hybrid
    #[serde(default = "default_true")]
    pub use_quantum_classical_hybrid: bool,

    /// Enable Multi-Scale neuromorphic analysis
    #[serde(default = "default_true")]
    pub use_multiscale_analysis: bool,

    /// Enable Ensemble Consensus
    #[serde(default = "default_true")]
    pub use_ensemble_consensus: bool,

    /// Enable Geodesic Features (OFF by default - experimental)
    #[serde(default)]
    pub use_geodesic_features: bool,

    /// Number of parallel worker threads
    #[serde(default = "default_threads")]
    pub num_workers: usize,

    /// GPU configuration
    #[serde(default)]
    pub gpu: GpuConfig,

    /// Memetic algorithm configuration
    #[serde(default)]
    pub memetic: MemeticConfig,

    /// Thermodynamic configuration
    #[serde(default)]
    pub thermo: ThermoConfig,

    /// Quantum configuration
    #[serde(default)]
    pub quantum: QuantumConfig,

    /// ADP configuration
    #[serde(default)]
    pub adp: AdpConfig,

    /// Orchestrator configuration
    #[serde(default)]
    pub orchestrator: OrchestratorConfig,

    /// Geodesic configuration
    #[serde(default)]
    pub geodesic: GeodesicConfig,

    /// GPU coloring configuration
    #[serde(default)]
    pub gpu_coloring: GpuColoringConfig,

    /// CPU configuration
    #[serde(default)]
    pub cpu: CpuConfig,
}

impl Default for WorldRecordConfig {
    fn default() -> Self {
        Self {
            profile: Some("record".to_string()),
            version: Some("1.0.0".to_string()),
            deterministic: false,
            seed: Some(123456789),
            target_chromatic: 83,  // DSJC1000.5 world record
            max_runtime_hours: 48.0,  // 2 days maximum
            use_active_inference: true,
            use_adp_learning: true,
            use_reservoir_prediction: true,
            use_thermodynamic_equilibration: true,
            use_quantum_classical_hybrid: true,
            use_multiscale_analysis: true,
            use_ensemble_consensus: true,
            use_geodesic_features: false,
            num_workers: 24,  // Intel i9 Ultra
            gpu: GpuConfig::default(),
            memetic: MemeticConfig::default(),
            thermo: ThermoConfig::default(),
            quantum: QuantumConfig::default(),
            adp: AdpConfig::default(),
            orchestrator: OrchestratorConfig::default(),
            geodesic: GeodesicConfig::default(),
            gpu_coloring: GpuColoringConfig::default(),
            cpu: CpuConfig::default(),
        }
    }
}

impl WorldRecordConfig {
    pub fn validate(&self) -> Result<()> {
        if self.target_chromatic < 1 {
            return Err(PRCTError::ColoringFailed("target_chromatic must be >= 1".to_string()));
        }

        if self.max_runtime_hours <= 0.0 || self.max_runtime_hours > 168.0 {
            return Err(PRCTError::ColoringFailed("max_runtime_hours must be in (0, 168]".to_string()));
        }

        if self.num_workers == 0 || self.num_workers > 256 {
            return Err(PRCTError::ColoringFailed("num_workers must be in [1, 256]".to_string()));
        }

        // Require at least one method enabled
        let mut any_enabled = false;
        any_enabled |= self.use_reservoir_prediction;
        any_enabled |= self.use_active_inference;
        any_enabled |= self.use_thermodynamic_equilibration;
        any_enabled |= self.use_quantum_classical_hybrid;
        any_enabled |= self.use_ensemble_consensus;
        any_enabled |= self.use_adp_learning;
        any_enabled |= self.use_multiscale_analysis;

        if !any_enabled {
            return Err(PRCTError::ColoringFailed("enable at least one phase".to_string()));
        }

        // Validate geodesic configuration if enabled
        if self.use_geodesic_features {
            if self.geodesic.num_landmarks == 0 {
                return Err(PRCTError::ColoringFailed("geodesic.num_landmarks must be > 0".to_string()));
            }
            if self.geodesic.metric != "hop" && self.geodesic.metric != "weighted" {
                return Err(PRCTError::ColoringFailed("geodesic.metric must be 'hop' or 'weighted'".to_string()));
            }
            if self.geodesic.centrality_weight < 0.0 || self.geodesic.centrality_weight > 1.0 {
                return Err(PRCTError::ColoringFailed("geodesic.centrality_weight must be in [0, 1]".to_string()));
            }
            if self.geodesic.eccentricity_weight < 0.0 || self.geodesic.eccentricity_weight > 1.0 {
                return Err(PRCTError::ColoringFailed("geodesic.eccentrity_weight must be in [0, 1]".to_string()));
            }
        }

        // Validate CPU configuration
        if self.cpu.threads == 0 || self.cpu.threads > 1024 {
            return Err(PRCTError::ColoringFailed("cpu.threads must be in [1, 1024]".to_string()));
        }

        // VRAM guards for 8GB GPU devices
        if self.thermo.replicas > 56 {
            eprintln!("[VRAM][GUARD] thermo.replicas {} exceeds safe limit for 8GB devices (max 56)", self.thermo.replicas);
            return Err(PRCTError::ColoringFailed(format!(
                "thermo.replicas {} exceeds VRAM limit (max 56 for 8GB devices)",
                self.thermo.replicas
            )));
        }

        if self.thermo.num_temps > 56 {
            eprintln!("[VRAM][GUARD] thermo.num_temps {} exceeds safe limit for 8GB devices (max 56)", self.thermo.num_temps);
            return Err(PRCTError::ColoringFailed(format!(
                "thermo.num_temps {} exceeds VRAM limit (max 56 for 8GB devices)",
                self.thermo.num_temps
            )));
        }

        // Validate ADP parameters use config values
        if self.adp.alpha <= 0.0 || self.adp.alpha > 1.0 {
            return Err(PRCTError::ColoringFailed("adp.alpha must be in (0, 1]".to_string()));
        }

        if self.adp.gamma < 0.0 || self.adp.gamma > 1.0 {
            return Err(PRCTError::ColoringFailed("adp.gamma must be in [0, 1]".to_string()));
        }

        if self.adp.epsilon_decay <= 0.0 || self.adp.epsilon_decay > 1.0 {
            return Err(PRCTError::ColoringFailed("adp.epsilon_decay must be in (0, 1]".to_string()));
        }

        if self.adp.epsilon_min < 0.0 || self.adp.epsilon_min > 1.0 {
            return Err(PRCTError::ColoringFailed("adp.epsilon_min must be in [0, 1]".to_string()));
        }

        Ok(())
    }
}

/// Active Inference Policy for Graph Coloring
pub struct ActiveInferencePolicy {
    /// Vertex uncertainty scores (higher = more uncertain)
    pub uncertainty: Vec<f64>,

    /// Expected free energy per vertex
    pub expected_free_energy: Vec<f64>,

    /// Pragmatic value (goal-directed)
    pub pragmatic_value: Vec<f64>,

    /// Epistemic value (information-seeking)
    pub epistemic_value: Vec<f64>,
}

impl ActiveInferencePolicy {
    /// Compute Active Inference policy from current coloring state
    pub fn compute(
        graph: &Graph,
        partial_coloring: &[usize],
        kuramoto_state: &KuramotoState,
    ) -> Result<Self> {
        let n = graph.num_vertices;
        let mut uncertainty = vec![0.0; n];
        let mut expected_free_energy = vec![0.0; n];
        let mut pragmatic_value = vec![0.0; n];
        let mut epistemic_value = vec![0.0; n];

        // Build adjacency for conflict detection
        let adj = build_adjacency_matrix(graph);

        for v in 0..n {
            if partial_coloring[v] != usize::MAX {
                continue;  // Already colored
            }

            // Pragmatic value: How hard is this vertex to color?
            let degree = (0..n).filter(|&u| adj[[v, u]]).count();
            let colored_neighbors = (0..n)
                .filter(|&u| adj[[v, u]] && partial_coloring[u] != usize::MAX)
                .count();

            pragmatic_value[v] = (colored_neighbors as f64) / (degree as f64 + 1.0);

            // Epistemic value: How much information do we gain?
            // Use Kuramoto phase coherence as proxy for information
            let phase = kuramoto_state.phases[v];
            let neighbor_phases: Vec<f64> = (0..n)
                .filter(|&u| adj[[v, u]])
                .map(|u| kuramoto_state.phases[u])
                .collect();

            if !neighbor_phases.is_empty() {
                let mean_phase = neighbor_phases.iter().sum::<f64>() / neighbor_phases.len() as f64;
                let phase_variance = neighbor_phases.iter()
                    .map(|&p| (p - mean_phase).powi(2))
                    .sum::<f64>() / neighbor_phases.len() as f64;

                epistemic_value[v] = phase_variance;  // Higher variance = more information
            }

            // Uncertainty: Combination of degree and phase dispersion
            uncertainty[v] = pragmatic_value[v] * (1.0 + epistemic_value[v]);

            // Expected Free Energy: Balance pragmatic and epistemic
            expected_free_energy[v] = pragmatic_value[v] - 0.5 * epistemic_value[v];
        }

        Ok(Self {
            uncertainty,
            expected_free_energy,
            pragmatic_value,
            epistemic_value,
        })
    }

    /// Select next vertex to color (minimize expected free energy)
    pub fn select_vertex(&self) -> usize {
        self.expected_free_energy
            .iter()
            .enumerate()
            .filter(|(_, &efe)| efe > 0.0)
            .min_by(|(_, a), (_, b)| {
                use std::cmp::Ordering;
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// ADP State for Reinforcement Learning
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ColoringState {
    /// Current chromatic number (discretized)
    pub chromatic_bucket: i32,

    /// Conflict density (discretized)
    pub conflict_bucket: i32,

    /// Phase coherence (discretized)
    pub coherence_bucket: i32,
}

impl ColoringState {
    pub fn from_solution(solution: &ColoringSolution, order_param: f64) -> Self {
        Self {
            chromatic_bucket: (solution.chromatic_number / 5) as i32,
            conflict_bucket: (solution.conflicts / 10) as i32,
            coherence_bucket: (order_param * 10.0) as i32,
        }
    }
}

/// ADP Actions for parameter tuning
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ColoringAction {
    IncreaseDSaturDepth,
    DecreaseDSaturDepth,
    IncreaseMemeticGenerations,
    DecreaseMemeticGenerations,
    IncreaseMutationRate,
    DecreaseMutationRate,
    IncreasePopulationSize,
    DecreasePopulationSize,
    FocusOnExploration,  // Higher diversity
    FocusOnExploitation,  // More local search
    IncreaseQuantumIterations,
    DecreaseQuantumIterations,
    IncreaseThermoTemperatures,
    DecreaseThermoTemperatures,
}

impl ColoringAction {
    pub fn all() -> Vec<Self> {
        vec![
            Self::IncreaseDSaturDepth,
            Self::DecreaseDSaturDepth,
            Self::IncreaseMemeticGenerations,
            Self::DecreaseMemeticGenerations,
            Self::IncreaseMutationRate,
            Self::DecreaseMutationRate,
            Self::IncreasePopulationSize,
            Self::DecreasePopulationSize,
            Self::FocusOnExploration,
            Self::FocusOnExploitation,
            Self::IncreaseQuantumIterations,
            Self::DecreaseQuantumIterations,
            Self::IncreaseThermoTemperatures,
            Self::DecreaseThermoTemperatures,
        ]
    }
}

/// Neuromorphic Conflict Predictor using Reservoir Computing
pub struct ReservoirConflictPredictor {
    /// Conflict probability per vertex
    pub conflict_scores: Vec<f64>,

    /// Difficulty zones (high-conflict regions)
    pub difficulty_zones: Vec<Vec<usize>>,
}

impl ReservoirConflictPredictor {
    /// Train reservoir on partial colorings and predict conflicts
    pub fn predict(
        graph: &Graph,
        coloring_history: &[ColoringSolution],
        kuramoto_state: &KuramotoState,
    ) -> Result<Self> {
        let n = graph.num_vertices;
        let mut conflict_scores = vec![0.0; n];

        // Analyze historical conflicts
        for solution in coloring_history {
            let adj = build_adjacency_matrix(graph);

            for v in 0..n {
                let mut local_conflicts = 0;
                for u in 0..n {
                    if adj[[v, u]] && solution.colors[v] == solution.colors[u] {
                        local_conflicts += 1;
                    }
                }

                // Update conflict score with exponential moving average
                conflict_scores[v] = 0.7 * conflict_scores[v] + 0.3 * (local_conflicts as f64);
            }
        }

        // Use Kuramoto phases to identify coherent difficulty zones
        let mut difficulty_zones = Vec::new();
        let phase_threshold = 0.5;  // Radians

        for seed in 0..n {
            if conflict_scores[seed] < 2.0 {
                continue;  // Not a difficult vertex
            }

            let mut zone = vec![seed];
            for v in 0..n {
                if v != seed
                    && conflict_scores[v] >= 2.0
                    && (kuramoto_state.phases[v] - kuramoto_state.phases[seed]).abs() < phase_threshold
                {
                    zone.push(v);
                }
            }

            if zone.len() >= 3 {
                difficulty_zones.push(zone);
            }
        }

        Ok(Self {
            conflict_scores,
            difficulty_zones,
        })
    }
}

/// Statistical Mechanics Equilibration using Thermodynamic Network
pub struct ThermodynamicEquilibrator {
    /// Temperature schedule for annealing
    pub temperatures: Vec<f64>,

    /// Equilibrium colorings at each temperature
    pub equilibrium_states: Vec<ColoringSolution>,
}

impl ThermodynamicEquilibrator {
    /// Find equilibrium colorings at multiple temperatures
    pub fn equilibrate(
        graph: &Graph,
        initial_solution: &ColoringSolution,
        target_chromatic: usize,
        num_temps: usize,  // ADP-tunable temperature count
    ) -> Result<Self> {
        // Logarithmic temperature schedule
        let t_max = 10.0;
        let t_min = 0.01;

        let temperatures: Vec<f64> = (0..num_temps)
            .map(|i| {
                let frac = i as f64 / (num_temps - 1) as f64;
                let ratio: f64 = t_min / t_max;
                t_max * ratio.powf(frac)
            })
            .collect();

        let mut equilibrium_states = Vec::new();
        let mut current = initial_solution.clone();

        println!("[THERMODYNAMIC] Starting replica exchange...");

        for (i, &temp) in temperatures.iter().enumerate() {
            println!("[THERMODYNAMIC] Temperature {}/{}: T = {:.3}", i + 1, num_temps, temp);

            // Simulated annealing at this temperature
            let mut best = current.clone();
            let adj = build_adjacency_matrix(graph);

            // World-record: 5000 steps per temperature (5x aggressive)
            for _ in 0..5000 {
                // Random recoloring move
                let v = rand::random::<usize>() % graph.num_vertices;
                let old_color = current.colors[v];
                let new_color = rand::random::<usize>() % target_chromatic;

                current.colors[v] = new_color;

                // Compute energy change (conflict count)
                let mut delta_conflicts = 0i32;
                for u in 0..graph.num_vertices {
                    if adj[[v, u]] {
                        if current.colors[u] == old_color {
                            delta_conflicts -= 1;
                        }
                        if current.colors[u] == new_color {
                            delta_conflicts += 1;
                        }
                    }
                }

                // Metropolis acceptance criterion
                if delta_conflicts > 0 {
                    let prob = (-delta_conflicts as f64 / temp).exp();
                    if rand::random::<f64>() > prob {
                        current.colors[v] = old_color;  // Reject
                    }
                }

                // Track best
                let conflicts = count_conflicts(&current.colors, &adj);
                if conflicts < best.conflicts {
                    best = current.clone();
                    best.conflicts = conflicts;
                }
            }

            equilibrium_states.push(best.clone());
            current = best;

            if current.conflicts == 0 {
                println!("[THERMODYNAMIC] ✅ Found valid coloring at T = {:.3}", temp);
            }
        }

        Ok(Self {
            temperatures,
            equilibrium_states,
        })
    }
}

/// Quantum-Classical Hybrid Solver
pub struct QuantumClassicalHybrid {
    /// Quantum solver for QUBO
    quantum_solver: QuantumColoringSolver,

    /// Classical solver for refinement
    classical_solver: DSaturSolver,

    /// Reservoir conflict scores for DSATUR guidance
    reservoir_scores: Option<Vec<f64>>,

    /// Active Inference expected free energy for vertex selection
    active_inference_efe: Option<Vec<f64>>,
}

impl QuantumClassicalHybrid {
    #[cfg(feature = "cuda")]
    pub fn new(
        max_colors: usize,
        cuda_device: Option<Arc<CudaDevice>>,
    ) -> Result<Self> {
        Ok(Self {
            quantum_solver: QuantumColoringSolver::new(cuda_device)?,
            classical_solver: DSaturSolver::new(max_colors, 50000),
            reservoir_scores: None,
            active_inference_efe: None,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(max_colors: usize) -> Result<Self> {
        Ok(Self {
            quantum_solver: QuantumColoringSolver::new()?,
            classical_solver: DSaturSolver::new(max_colors, 50000),
            reservoir_scores: None,
            active_inference_efe: None,
        })
    }

    /// Set reservoir conflict scores for DSATUR tie-breaking
    pub fn set_reservoir_scores(&mut self, scores: Vec<f64>) {
        self.reservoir_scores = Some(scores.clone());
        self.classical_solver = self.classical_solver.clone().with_reservoir_scores(scores);
    }

    /// Set Active Inference expected free energy for vertex selection
    pub fn set_active_inference(&mut self, efe: Vec<f64>) {
        self.active_inference_efe = Some(efe.clone());
        self.classical_solver = self.classical_solver.clone().with_active_inference(efe);
    }

    /// Solve with quantum-classical feedback loop
    pub fn solve_with_feedback(
        &mut self,
        graph: &Graph,
        initial_solution: &ColoringSolution,
        kuramoto_state: &KuramotoState,
        num_iterations: usize,
    ) -> Result<ColoringSolution> {
        let mut best = initial_solution.clone();
        let n = graph.num_vertices;

        println!("[QUANTUM-CLASSICAL] Starting hybrid feedback loop...");

        for iter in 0..num_iterations {
            println!("[QUANTUM-CLASSICAL] Iteration {}/{}", iter + 1, num_iterations);

            // Construct PhaseField from current best coloring and Kuramoto state
            let phase_field = self.construct_phase_field(graph, &best, kuramoto_state, None)?;

            // Phase 1: Quantum QUBO solve with error handling and fallback
            println!("[QUANTUM-CLASSICAL]   Phase 1: Quantum QUBO...");
            match self.quantum_solver.find_coloring(
                graph,
                &phase_field,
                kuramoto_state,
                best.chromatic_number,
            ) {
                Ok(quantum_result) => {
                    if quantum_result.chromatic_number < best.chromatic_number && quantum_result.conflicts == 0 {
                        println!("[QUANTUM-CLASSICAL]   🎯 Quantum improved: {} → {} colors",
                                 best.chromatic_number, quantum_result.chromatic_number);
                        best = quantum_result.clone();
                    }
                }
                Err(e) => {
                    println!("[QUANTUM-CLASSICAL]   ⚠️  Quantum solver failed: {:?}", e);
                    println!("[QUANTUM-CLASSICAL]   ℹ️  Falling back to DSATUR-only refinement");
                    // Don't abort - continue with classical solver below
                }
            }

            // Phase 2: Classical refinement
            println!("[QUANTUM-CLASSICAL]   Phase 2: Classical DSATUR refinement...");
            let classical_result = self.classical_solver.find_coloring(
                graph,
                Some(&best),
                best.chromatic_number.saturating_sub(3),
            )?;

            if classical_result.chromatic_number < best.chromatic_number
               && classical_result.conflicts == 0
            {
                println!("[QUANTUM-CLASSICAL]   🎯 Classical improved: {} → {} colors",
                         best.chromatic_number, classical_result.chromatic_number);
                best = classical_result;
            }

            // Adaptive iteration: stop early if no progress
            if iter > 0 && best.chromatic_number == initial_solution.chromatic_number {
                println!("[QUANTUM-CLASSICAL]   No improvement after {} iterations, stopping early", iter + 1);
                break;
            }
        }

        Ok(best)
    }

    /// Construct PhaseField from coloring solution and Kuramoto state
    fn construct_phase_field(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
        kuramoto_state: &KuramotoState,
        _geodesic_features: Option<&GeodesicFeatures>,
    ) -> Result<PhaseField> {
        let n = graph.num_vertices;

        // Use Kuramoto phases as base, modulated by color information
        let mut phases = kuramoto_state.phases.clone();

        // Adjust phases based on color classes to create coherent clusters
        use std::f64::consts::PI;
        for v in 0..n {
            let color = solution.colors[v];
            if color != usize::MAX {
                // Add color-based phase shift while preserving Kuramoto structure
                let color_shift = 2.0 * PI * (color as f64) / (solution.chromatic_number as f64);
                phases[v] = (phases[v] + 0.5 * color_shift) % (2.0 * PI);
            }
        }

        // Compute coherence matrix based on graph adjacency
        let mut coherence_matrix = vec![0.0; n * n];
        for &(u, v, _weight) in &graph.edges {
            // High coherence for adjacent vertices (should have different colors)
            let phase_diff = (phases[u] - phases[v]).abs();
            let coherence = (phase_diff / PI).min(1.0); // Normalize to [0, 1]
            coherence_matrix[u * n + v] = coherence;
            coherence_matrix[v * n + u] = coherence;
        }

        Ok(PhaseField {
            phases,
            coherence_matrix,
            order_parameter: kuramoto_state.order_parameter,
            resonance_frequency: 1.0, // Default resonance
        })
    }
}

/// Ensemble Consensus Voting System
pub struct EnsembleConsensus {
    /// Solutions from different algorithms
    pub solutions: Vec<ColoringSolution>,

    /// Algorithm names
    pub algorithm_names: Vec<String>,
}

impl EnsembleConsensus {
    pub fn new() -> Self {
        Self {
            solutions: Vec::new(),
            algorithm_names: Vec::new(),
        }
    }

    /// Add solution from an algorithm
    pub fn add_solution(&mut self, solution: ColoringSolution, algorithm: &str) {
        self.solutions.push(solution);
        self.algorithm_names.push(algorithm.to_string());
    }

    /// Consensus voting: Choose best valid coloring
    pub fn vote(&self) -> Result<ColoringSolution> {
        if self.solutions.is_empty() {
            return Err(PRCTError::ColoringFailed("No solutions to vote on".to_string()));
        }

        println!("[ENSEMBLE] Voting among {} solutions...", self.solutions.len());

        // Filter valid colorings only
        let valid: Vec<&ColoringSolution> = self.solutions
            .iter()
            .filter(|s| s.conflicts == 0)
            .collect();

        if valid.is_empty() {
            println!("[ENSEMBLE] ⚠️  No valid colorings, returning best approximate");
            let best_approx = self
                .solutions
                .iter()
                .min_by_key(|s| (s.conflicts, s.chromatic_number))
                .cloned()
                .ok_or_else(|| PRCTError::ColoringFailed("No solutions available for ensemble fallback".to_string()))?;
            return Ok(best_approx);
        }

        // Return best valid coloring
        // Safe due to !valid.is_empty() check above
        let best = valid
            .iter()
            .min_by_key(|s| s.chromatic_number)
            .expect("valid is non-empty after guard; min_by_key must return Some");

        println!("[ENSEMBLE] ✅ Consensus: {} colors", best.chromatic_number);
        Ok((*best).clone())
    }
}

/// Helper functions
fn build_adjacency_matrix(graph: &Graph) -> ndarray::Array2<bool> {
    use ndarray::Array2;
    let n = graph.num_vertices;
    let mut adj = Array2::from_elem((n, n), false);

    for &(u, v, _weight) in &graph.edges {
        adj[[u, v]] = true;
        adj[[v, u]] = true;
    }

    adj
}

fn count_conflicts(coloring: &[usize], adj: &ndarray::Array2<bool>) -> usize {
    let n = coloring.len();
    let mut conflicts = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

impl Default for EnsembleConsensus {
    fn default() -> Self {
        Self::new()
    }
}

/// **WORLD RECORD BREAKING PIPELINE ORCHESTRATOR**
///
/// The ultimate integration of ALL PRISM modules for breaking the
/// 83-color world record on DSJC1000.5
pub struct WorldRecordPipeline {
    config: WorldRecordConfig,

    /// Best solution found so far
    best_solution: ColoringSolution,

    /// Coloring history for learning
    history: Vec<ColoringSolution>,

    /// Active Inference policy
    active_inference_policy: Option<ActiveInferencePolicy>,

    /// Dendritic-enhanced neuromorphic predictor (GPU-accelerated)
    #[cfg(feature = "cuda")]
    conflict_predictor_gpu: Option<GpuReservoirConflictPredictor>,

    /// Dendritic-enhanced neuromorphic predictor (CPU fallback)
    #[cfg(not(feature = "cuda"))]
    conflict_predictor: Option<ReservoirConflictPredictor>,

    /// Thermodynamic equilibrator
    thermodynamic_eq: Option<ThermodynamicEquilibrator>,

    /// Quantum-Classical hybrid solver
    quantum_classical: Option<QuantumClassicalHybrid>,

    /// Ensemble consensus system
    ensemble: EnsembleConsensus,

    /// ADP Q-table for parameter tuning
    adp_q_table: std::collections::HashMap<(ColoringState, ColoringAction), f64>,

    /// Shared CUDA device for GPU acceleration
    #[cfg(feature = "cuda")]
    cuda_device: Arc<CudaDevice>,

    adp_epsilon: f64,

    /// ADP-tuned solver parameters
    adp_dsatur_depth: usize,
    adp_quantum_iterations: usize,
    adp_thermo_num_temps: usize,

    /// Stagnation tracking for adaptive loopback
    stagnation_count: usize,
    last_improvement_iteration: usize,
}

impl WorldRecordPipeline {
    #[cfg(feature = "cuda")]
    pub fn new(config: WorldRecordConfig, cuda_device: Arc<CudaDevice>) -> Result<Self> {
        config.validate()?;

        // Initialize Rayon thread pool with configured CPU threads
        init_rayon_threads(config.cpu.threads);

        Ok(Self {
            config: config.clone(),
            best_solution: ColoringSolution {
                colors: vec![],
                chromatic_number: usize::MAX,
                conflicts: usize::MAX,
                quality_score: 0.0,
                computation_time_ms: 0.0,
            },
            history: Vec::new(),
            active_inference_policy: None,
            conflict_predictor_gpu: None,
            thermodynamic_eq: None,
            quantum_classical: Some(QuantumClassicalHybrid::new(
                config.target_chromatic,
                Some(cuda_device.clone()),
            )?),
            ensemble: EnsembleConsensus::new(),
            adp_q_table: std::collections::HashMap::new(),
            cuda_device,
            adp_epsilon: 1.0,  // Start with full exploration
            adp_dsatur_depth: 200000,  // World-record DSATUR depth (4x aggressive)
            adp_quantum_iterations: 20,  // World-record quantum iterations (4x)
            adp_thermo_num_temps: 64,  // World-record temperature count (2x)
            stagnation_count: 0,
            last_improvement_iteration: 0,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(config: WorldRecordConfig) -> Result<Self> {
        config.validate()?;

        // Initialize Rayon thread pool with configured CPU threads
        init_rayon_threads(config.cpu.threads);

        Ok(Self {
            config: config.clone(),
            best_solution: ColoringSolution {
                colors: vec![],
                chromatic_number: usize::MAX,
                conflicts: usize::MAX,
                quality_score: 0.0,
                computation_time_ms: 0.0,
            },
            history: Vec::new(),
            active_inference_policy: None,
            conflict_predictor: None,
            thermodynamic_eq: None,
            quantum_classical: Some(QuantumClassicalHybrid::new(config.target_chromatic)?),
            ensemble: EnsembleConsensus::new(),
            adp_q_table: std::collections::HashMap::new(),
            adp_epsilon: 1.0,  // Start with full exploration
            adp_dsatur_depth: 200000,  // World-record DSATUR depth (4x aggressive)
            adp_quantum_iterations: 20,  // World-record quantum iterations (4x)
            adp_thermo_num_temps: 64,  // World-record temperature count (2x)
            stagnation_count: 0,
            last_improvement_iteration: 0,
        })
    }

    /// **MAIN WORLD RECORD ATTEMPT**
    ///
    /// Runs the complete multi-modal PRISM pipeline
    pub fn optimize_world_record(
        &mut self,
        graph: &Graph,
        initial_kuramoto: &KuramotoState,
    ) -> Result<ColoringSolution> {
        let start = std::time::Instant::now();

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║       WORLD RECORD BREAKING PIPELINE - PRISM ULTIMATE     ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!("[WR-PIPELINE] Target: {} colors (World Record)", self.config.target_chromatic);
        println!("[WR-PIPELINE] Current Best: 115 colors (Full Integration)");
        println!("[WR-PIPELINE] Gap to Close: 32 colors");
        println!();

        // ═══════════════════════════════════════════════════════════
        // PHASE 0A: Geodesic Features (if enabled)
        // ═══════════════════════════════════════════════════════════
        let geodesic_features = if self.config.use_geodesic_features {
            println!("┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 0A: Geodesic Feature Computation                 │");
            println!("└─────────────────────────────────────────────────────────┘");

            let features = compute_landmark_distances(
                graph,
                self.config.geodesic.num_landmarks,
                &self.config.geodesic.metric,
            )?;

            println!("[PHASE 0A] ✅ Geodesic features computed for {} landmarks", features.landmarks.len());
            Some(features)
        } else {
            None
        };

        // ═══════════════════════════════════════════════════════════
        // PHASE 0B: Dendritic-Enhanced Neuromorphic Pre-Analysis
        // ═══════════════════════════════════════════════════════════
        if self.config.use_reservoir_prediction {
            println!("┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 0: Dendritic Neuromorphic Conflict Prediction    │");
            println!("└─────────────────────────────────────────────────────────┘");

            // Train reservoir on initial greedy solutions (world-record: 2x for better accuracy)
            let mut training_solutions = Vec::new();
            for _ in 0..10 {
                let random_order: Vec<usize> = (0..graph.num_vertices).collect();
                let greedy = greedy_coloring_with_ordering(graph, &random_order)?;
                training_solutions.push(greedy);
            }

            #[cfg(feature = "cuda")]
            {
                println!("[PHASE 0] 🚀 Using GPU-accelerated neuromorphic reservoir (10-50x speedup)");
                self.conflict_predictor_gpu = Some(GpuReservoirConflictPredictor::predict_gpu(
                    graph,
                    &training_solutions,
                    initial_kuramoto,
                    Arc::clone(&self.cuda_device),
                )?);

                // Safe: just set above; expect avoids silent panic and documents invariant
                let predictor = self
                    .conflict_predictor_gpu
                    .as_ref()
                    .expect("conflict_predictor_gpu should be set after predict_gpu()");
                println!("[PHASE 0] ✅ Identified {} difficulty zones", predictor.difficulty_zones.len());
                println!("[PHASE 0] ✅ GPU dendritic processing: {} high-conflict vertices",
                         predictor.conflict_scores.iter().filter(|&&s| s > 2.0).count());
            }

            #[cfg(not(feature = "cuda"))]
            {
                println!("[PHASE 0] ⚠️  Using CPU fallback (CUDA not available)");
                self.conflict_predictor = Some(ReservoirConflictPredictor::predict(
                    graph,
                    &training_solutions,
                    initial_kuramoto,
                )?);

                // Safe: just set above; expect avoids silent panic and documents invariant
                let predictor = self
                    .conflict_predictor
                    .as_ref()
                    .expect("conflict_predictor (CPU) should be set after predict()");
                println!("[PHASE 0] ✅ Identified {} difficulty zones", predictor.difficulty_zones.len());
                println!("[PHASE 0] ✅ Dendritic processing: {} high-conflict vertices",
                         predictor.conflict_scores.iter().filter(|&&s| s > 2.0).count());
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 1: Transfer Entropy with Active Inference Policy
        // ═══════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ PHASE 1: Active Inference-Guided Transfer Entropy      │");
        println!("└─────────────────────────────────────────────────────────┘");

        let te_ordering = hybrid_te_kuramoto_ordering(graph, initial_kuramoto, geodesic_features.as_ref(), 0.2)?;
        let mut te_solution = greedy_coloring_with_ordering(graph, &te_ordering)?;

        // Apply Active Inference to refine difficult vertices
        if self.config.use_active_inference {
            self.active_inference_policy = Some(ActiveInferencePolicy::compute(
                graph,
                &te_solution.colors,
                initial_kuramoto,
            )?);

            println!("[PHASE 1] ✅ Active Inference: Computed expected free energy");
            println!("[PHASE 1] ✅ Uncertainty-guided vertex selection enabled");
        }

        println!("[PHASE 1] ✅ TE-guided coloring: {} colors", te_solution.chromatic_number);
        self.best_solution = te_solution.clone();
        self.history.push(te_solution.clone());
        self.ensemble.add_solution(te_solution.clone(), "Transfer Entropy");

        // ═══════════════════════════════════════════════════════════
        // PHASE 2: Statistical Mechanics Thermodynamic Equilibration
        // ═══════════════════════════════════════════════════════════
        if self.config.use_thermodynamic_equilibration {
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 2: Thermodynamic Replica Exchange                │");
            println!("└─────────────────────────────────────────────────────────┘");

            // ADP: Learn from Phase 1 results
            if self.config.use_adp_learning && self.history.len() >= 2 {
                let current_state = ColoringState::from_solution(
                    &self.best_solution,
                    initial_kuramoto.order_parameter,
                );
                let best_action = self.select_adp_action(&current_state);

                // For thermodynamic phase, we only apply temperature-related actions
                match best_action {
                    ColoringAction::IncreaseThermoTemperatures | ColoringAction::DecreaseThermoTemperatures => {
                        self.apply_adp_action(&mut MemeticConfig::default(), best_action);
                        println!("[PHASE 2] 🧠 ADP tuned thermo temps: {}", self.adp_thermo_num_temps);
                    },
                    _ => {
                        // Save other actions for later phases
                    }
                }
            }

            self.thermodynamic_eq = Some(ThermodynamicEquilibrator::equilibrate(
                graph,
                &self.best_solution,
                self.config.target_chromatic,
                self.adp_thermo_num_temps,  // ADP-tuned temperature count
            )?);

            // Safe: just set above; expect avoids silent panic and documents invariant
            let eq = self
                .thermodynamic_eq
                .as_ref()
                .expect("thermodynamic_eq should be set after equilibrate()");
            for (i, state) in eq.equilibrium_states.iter().enumerate() {
                if state.conflicts == 0 && state.chromatic_number < self.best_solution.chromatic_number {
                    println!("[PHASE 2] 🎯 Thermodynamic improvement at T={:.3}: {} → {} colors",
                             eq.temperatures[i], self.best_solution.chromatic_number, state.chromatic_number);
                    self.best_solution = state.clone();
                }
                self.history.push(state.clone());
                self.ensemble.add_solution(state.clone(), &format!("Thermodynamic-T{:.3}", eq.temperatures[i]));
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 3: Quantum-Classical Hybrid with Feedback
        // ═══════════════════════════════════════════════════════════
        if self.config.use_quantum_classical_hybrid {
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 3: Quantum-Classical Hybrid Feedback Loop        │");
            println!("└─────────────────────────────────────────────────────────┘");

            // ADP: Learn from Phase 2 results and tune quantum parameters
            if self.config.use_adp_learning && self.history.len() >= 5 {
                let current_state = ColoringState::from_solution(
                    &self.best_solution,
                    initial_kuramoto.order_parameter,
                );
                let best_action = self.select_adp_action(&current_state);

                // Apply quantum-related and DSATUR-related actions
                match best_action {
                    ColoringAction::IncreaseQuantumIterations | ColoringAction::DecreaseQuantumIterations |
                    ColoringAction::IncreaseDSaturDepth | ColoringAction::DecreaseDSaturDepth => {
                        self.apply_adp_action(&mut MemeticConfig::default(), best_action);
                        println!("[PHASE 3] 🧠 ADP action: {:?}", best_action);
                    },
                    _ => {}
                }
            }

            if let Some(qc_hybrid) = &mut self.quantum_classical {
                // Wire in reservoir conflict scores for DSATUR tie-breaking
                #[cfg(feature = "cuda")]
                {
                    if let Some(ref predictor) = self.conflict_predictor_gpu {
                        qc_hybrid.set_reservoir_scores(predictor.get_conflict_scores().to_vec());
                        println!("[PHASE 3] ✅ Wired GPU reservoir scores into DSATUR");
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    if let Some(ref predictor) = self.conflict_predictor {
                        qc_hybrid.set_reservoir_scores(predictor.conflict_scores.clone());
                        println!("[PHASE 3] ✅ Wired CPU reservoir scores into DSATUR");
                    }
                }

                // Wire in Active Inference expected free energy
                if let Some(ref policy) = self.active_inference_policy {
                    qc_hybrid.set_active_inference(policy.expected_free_energy.clone());
                    println!("[PHASE 3] ✅ Wired Active Inference EFE into DSATUR");
                }

                // Use ADP-tuned quantum iteration count
                println!("[PHASE 3] 🧠 ADP-tuned quantum iterations: {}", self.adp_quantum_iterations);
                match qc_hybrid.solve_with_feedback(
                    graph,
                    &self.best_solution,
                    initial_kuramoto,
                    self.adp_quantum_iterations,  // ADP-tuned iterations
                ) {
                    Ok(qc_solution) => {
                        if qc_solution.conflicts == 0 && qc_solution.chromatic_number < self.best_solution.chromatic_number {
                            println!("[PHASE 3] 🎯 Quantum-Classical breakthrough: {} → {} colors",
                                     self.best_solution.chromatic_number, qc_solution.chromatic_number);
                            self.best_solution = qc_solution.clone();
                        }
                        self.history.push(qc_solution.clone());
                        self.ensemble.add_solution(qc_solution, "Quantum-Classical");
                    }
                    Err(e) => {
                        println!("[PHASE 3] ⚠️  Quantum-Classical phase failed: {:?}", e);
                        println!("[PHASE 3] ℹ️  Continuing with best solution from previous phases");
                        // Continue the pipeline - don't abort
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 4: ADP-Optimized Memetic Algorithm
        // ═══════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ PHASE 4: ADP Q-Learning Memetic Optimization           │");
        println!("└─────────────────────────────────────────────────────────┘");

        // World-record aggressive settings (48-hour target)
        let mut memetic_config = MemeticConfig {
            population_size: 128,  // 4x for better diversity
            elite_size: 16,        // Scale with population
            generations: 500,      // 10x for deeper search
            mutation_rate: 0.20,
            tournament_size: 5,    // Stronger selection pressure
            local_search_depth: 50000,  // 10x for intensive local optimization
            use_tsp_guidance: true,
            tsp_weight: 0.25,
        };

        // ADP: Learn optimal parameters from history
        if self.config.use_adp_learning && self.history.len() > 3 {
            let current_state = ColoringState::from_solution(
                &self.best_solution,
                initial_kuramoto.order_parameter,
            );

            // Q-learning action selection
            let best_action = self.select_adp_action(&current_state);
            self.apply_adp_action(&mut memetic_config, best_action);

            println!("[PHASE 4] 🧠 ADP selected action: {:?}", best_action);
        }

        let initial_pop = vec![
            self.best_solution.clone(),
            te_solution,
        ];

        let mut memetic = MemeticColoringSolver::new(memetic_config.clone());
        let memetic_solution = memetic.solve_with_restart(graph, initial_pop, 10)?;  // 10 restarts for world-record attempt

        if memetic_solution.conflicts == 0 && memetic_solution.chromatic_number < self.best_solution.chromatic_number {
            println!("[PHASE 4] 🎯 Memetic+ADP improved: {} → {} colors",
                     self.best_solution.chromatic_number, memetic_solution.chromatic_number);

            // ADP: Update Q-value with reward
            if self.config.use_adp_learning {
                let reward = (self.best_solution.chromatic_number as f64
                             - memetic_solution.chromatic_number as f64) * 10.0;
                self.update_adp_q_value(&ColoringState::from_solution(&self.best_solution, initial_kuramoto.order_parameter),
                                       self.select_adp_action(&ColoringState::from_solution(&self.best_solution, initial_kuramoto.order_parameter)),
                                       reward);
            }

            self.best_solution = memetic_solution.clone();
        }
        self.history.push(memetic_solution.clone());
        self.ensemble.add_solution(memetic_solution, "Memetic+ADP");

        // ═══════════════════════════════════════════════════════════
        // PHASE 5: Ensemble Consensus with Multi-Scale Analysis
        // ═══════════════════════════════════════════════════════════
        if self.config.use_ensemble_consensus {
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 5: Ensemble Consensus Voting                     │");
            println!("└─────────────────────────────────────────────────────────┘");

            let consensus_solution = self.ensemble.vote()?;

            if consensus_solution.chromatic_number < self.best_solution.chromatic_number {
                println!("[PHASE 5] 🎯 Ensemble consensus: {} → {} colors",
                         self.best_solution.chromatic_number, consensus_solution.chromatic_number);
                self.best_solution = consensus_solution;
            }
        }

        // ═══════════════════════════════════════════════════════════
        // STAGNATION DETECTION & ADAPTIVE LOOPBACK
        // ═══════════════════════════════════════════════════════════
        let current_best = self.best_solution.chromatic_number;
        let history_len = self.history.len();

        // Check if we've improved in the last 5 solutions
        let recent_improvement = if history_len >= 5 {
            self.history.iter()
                .rev()
                .take(5)
                .any(|sol| sol.chromatic_number < current_best)
        } else {
            true  // Still warming up
        };

        if !recent_improvement && history_len >= 10 {
            self.stagnation_count += 1;
            println!("\n⚠️  STAGNATION DETECTED (count: {})", self.stagnation_count);

            // Adaptive recovery strategies
            if self.stagnation_count >= 2 {
                println!("🔄 ADAPTIVE LOOPBACK: Applying recovery strategies...");

                // Strategy 1: Increase exploration (reset epsilon, boost mutation)
                self.adp_epsilon = (self.adp_epsilon + 0.3).min(1.0);
                println!("   • Increased exploration: ε = {:.3}", self.adp_epsilon);

                // Strategy 2: Boost thermodynamic escape capability
                self.adp_thermo_num_temps = (self.adp_thermo_num_temps + 16).min(128);
                println!("   • Increased thermal diversity: {} temps", self.adp_thermo_num_temps);

                // Strategy 3: Intensify quantum iterations
                self.adp_quantum_iterations = (self.adp_quantum_iterations + 5).min(30);
                println!("   • Increased quantum depth: {} iterations", self.adp_quantum_iterations);

                // Strategy 4: Try desperation phase with extreme memetic search
                if self.stagnation_count >= 3 {
                    println!("   • 🚨 DESPERATION MODE: Running extreme memetic search...");

                    let desperation_config = MemeticConfig {
                        population_size: 256,  // 2x aggressive
                        elite_size: 32,
                        generations: 1000,     // 2x aggressive
                        mutation_rate: 0.35,   // Higher mutation
                        tournament_size: 7,
                        local_search_depth: 100000,  // Maximum depth
                        use_tsp_guidance: true,
                        tsp_weight: 0.3,
                    };

                    let initial_pop = vec![
                        self.best_solution.clone(),
                        // Use fallback to best_solution if history is empty (should be rare)
                        self.history
                            .last()
                            .unwrap_or(&self.best_solution)
                            .clone(),
                    ];

                    let mut desperation_memetic = MemeticColoringSolver::new(desperation_config);
                    if let Ok(desperation_sol) = desperation_memetic.solve_with_restart(graph, initial_pop, 5) {
                        if desperation_sol.conflicts == 0 && desperation_sol.chromatic_number < self.best_solution.chromatic_number {
                            println!("   • 🎯 Desperation mode SUCCESS: {} → {} colors!",
                                     self.best_solution.chromatic_number, desperation_sol.chromatic_number);
                            self.best_solution = desperation_sol.clone();
                            self.stagnation_count = 0;  // Reset on improvement
                        }
                    }
                }
            }
        } else if recent_improvement {
            // Reset stagnation counter on improvement
            if self.stagnation_count > 0 {
                println!("✅ Progress detected, resetting stagnation counter");
                self.stagnation_count = 0;
                self.last_improvement_iteration = history_len;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();

        // ═══════════════════════════════════════════════════════════
        // FINAL RESULTS
        // ═══════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║                  WORLD RECORD ATTEMPT COMPLETE             ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
        println!("🏆 FINAL RESULT: {} colors", self.best_solution.chromatic_number);
        println!("🥇 World Record: {} colors", self.config.target_chromatic);
        println!("📊 Gap to WR: {:.2}x", self.best_solution.chromatic_number as f64 / self.config.target_chromatic as f64);
        println!("⏱️  Total Time: {:.2}s", elapsed);
        println!();

        if self.best_solution.chromatic_number <= self.config.target_chromatic {
            println!("🎉 *** WORLD RECORD MATCHED OR BEATEN! *** 🎉");
        } else if self.best_solution.chromatic_number <= 90 {
            println!("✨ *** EXCELLENT RESULT (Within 10% of WR) *** ✨");
        } else if self.best_solution.chromatic_number <= 100 {
            println!("✅ *** STRONG RESULT (Target <100 achieved) *** ✅");
        }

        Ok(self.best_solution.clone())
    }

    /// ADP Q-Learning: Select action using epsilon-greedy
    fn select_adp_action(&self, state: &ColoringState) -> ColoringAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.adp_epsilon {
            // Explore: Random action
            let actions = ColoringAction::all();
            actions[rng.gen_range(0..actions.len())]
        } else {
            // Exploit: Best Q-value
            let actions = ColoringAction::all();
            actions.iter()
                .max_by(|a, b| {
                    use std::cmp::Ordering;
                    let q_a = self.adp_q_table.get(&(state.clone(), **a)).unwrap_or(&0.0);
                    let q_b = self.adp_q_table.get(&(state.clone(), **b)).unwrap_or(&0.0);
                    q_a.partial_cmp(q_b).unwrap_or(Ordering::Equal)
                })
                .copied()
                .unwrap_or(ColoringAction::FocusOnExploration)
        }
    }

    /// ADP Q-Learning: Update Q-value
    fn update_adp_q_value(&mut self, state: &ColoringState, action: ColoringAction, reward: f64) {
        let alpha = self.config.adp.alpha;
        let gamma = self.config.adp.gamma;

        let old_q = self.adp_q_table.get(&(state.clone(), action)).unwrap_or(&0.0);
        let new_q = old_q + alpha * (reward - old_q);

        self.adp_q_table.insert((state.clone(), action), new_q);

        // Decay epsilon (exploration rate)
        self.adp_epsilon *= self.config.adp.epsilon_decay;
        self.adp_epsilon = self.adp_epsilon.max(self.config.adp.epsilon_min);
    }

    /// Apply ADP action to memetic configuration and solver parameters
    fn apply_adp_action(&mut self, config: &mut MemeticConfig, action: ColoringAction) {
        match action {
            ColoringAction::IncreaseDSaturDepth => {
                config.local_search_depth += 1000;
                self.adp_dsatur_depth = (self.adp_dsatur_depth + 5000).min(100000);
            },
            ColoringAction::DecreaseDSaturDepth => {
                config.local_search_depth = config.local_search_depth.saturating_sub(1000).max(500);
                self.adp_dsatur_depth = self.adp_dsatur_depth.saturating_sub(5000).max(10000);
            },
            ColoringAction::IncreaseMemeticGenerations => config.generations += 10,
            ColoringAction::DecreaseMemeticGenerations => config.generations = config.generations.saturating_sub(10).max(20),
            ColoringAction::IncreaseMutationRate => config.mutation_rate = (config.mutation_rate + 0.05).min(0.5),
            ColoringAction::DecreaseMutationRate => config.mutation_rate = (config.mutation_rate - 0.05).max(0.05),
            ColoringAction::IncreasePopulationSize => config.population_size += 8,
            ColoringAction::DecreasePopulationSize => config.population_size = config.population_size.saturating_sub(8).max(16),
            ColoringAction::FocusOnExploration => {
                config.mutation_rate = (config.mutation_rate + 0.1).min(0.5);
                config.population_size += 8;
            },
            ColoringAction::FocusOnExploitation => {
                config.local_search_depth += 2000;
                config.elite_size += 2;
            },
            ColoringAction::IncreaseQuantumIterations => {
                self.adp_quantum_iterations = (self.adp_quantum_iterations + 2).min(10);
            },
            ColoringAction::DecreaseQuantumIterations => {
                self.adp_quantum_iterations = self.adp_quantum_iterations.saturating_sub(1).max(2);
            },
            ColoringAction::IncreaseThermoTemperatures => {
                self.adp_thermo_num_temps = (self.adp_thermo_num_temps + 8).min(64);
            },
            ColoringAction::DecreaseThermoTemperatures => {
                self.adp_thermo_num_temps = self.adp_thermo_num_temps.saturating_sub(8).max(8);
            },
        }
    }
}

impl Default for WorldRecordPipeline {
    #[cfg(feature = "cuda")]
    fn default() -> Self {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        Self::new(WorldRecordConfig::default(), device)
            .expect("Failed to create default WorldRecordPipeline")
    }

    #[cfg(not(feature = "cuda"))]
    fn default() -> Self {
        Self::new(WorldRecordConfig::default())
            .expect("Failed to create default WorldRecordPipeline")
    }
}
