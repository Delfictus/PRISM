//! PRCT Core Domain
//!
//! Pure domain logic for Phase Resonance Chromatic-TSP algorithm.
//! This crate contains ONLY business logic - no infrastructure dependencies.
//!
//! Architecture: Hexagonal (Ports & Adapters)
//! - Domain logic depends on port abstractions (traits)
//! - Infrastructure adapters implement ports
//! - Dependency arrows point INWARD to domain

pub mod ports;
pub mod algorithm;
pub mod drpp_algorithm;
pub mod coupling;
pub mod cpu_init;
pub mod coloring;
pub mod simulated_annealing;
pub mod tsp;
pub mod errors;
pub mod dimacs_parser;
pub mod adapters; // ADDED: Adapter implementations

// Re-export main types
pub use ports::*;
pub use algorithm::*;
pub use drpp_algorithm::*;
pub use coupling::*;
pub use cpu_init::init_rayon_threads;
pub use coloring::{phase_guided_coloring, greedy_coloring_with_ordering};
pub use simulated_annealing::*;
pub use errors::*;
pub use dimacs_parser::{parse_dimacs_file, parse_mtx_file, parse_graph_file};

// Re-export adapters
#[cfg(feature = "cuda")]
pub use adapters::NeuromorphicAdapter;
pub use adapters::{QuantumAdapter, CouplingAdapter};

// Re-export shared types for convenience
pub use shared_types::*;
pub mod gpu_prct;
pub use gpu_prct::GpuPRCT;

#[cfg(feature = "cuda")]
pub mod gpu_kuramoto;
#[cfg(feature = "cuda")]
pub use gpu_kuramoto::GpuKuramotoSolver;

#[cfg(feature = "cuda")]
pub mod gpu_quantum;
#[cfg(feature = "cuda")]
pub use gpu_quantum::GpuQuantumSolver;

pub mod quantum_coloring;
pub use quantum_coloring::QuantumColoringSolver;

pub mod sparse_qubo;
pub use sparse_qubo::{SparseQUBO, ChromaticBounds};

pub mod dsatur_backtracking;
pub use dsatur_backtracking::DSaturSolver;

pub mod transfer_entropy_coloring;
pub use transfer_entropy_coloring::{
    compute_transfer_entropy_ordering,
    hybrid_te_kuramoto_ordering,
};

pub mod memetic_coloring;
pub use memetic_coloring::{MemeticColoringSolver, MemeticConfig};

pub mod geodesic;

pub mod cascading_pipeline;
pub use cascading_pipeline::CascadingPipeline;

pub mod world_record_pipeline;
pub use world_record_pipeline::{
    WorldRecordPipeline,
    WorldRecordConfig,
    GpuConfig,
    ThermoConfig,
    QuantumConfig,
    AdpConfig,
    OrchestratorConfig,
    ActiveInferencePolicy,
    ReservoirConflictPredictor,
    ThermodynamicEquilibrator,
    QuantumClassicalHybrid,
    EnsembleConsensus,
};

pub mod config_io;

#[cfg(feature = "cuda")]
pub mod world_record_pipeline_gpu;
#[cfg(feature = "cuda")]
pub use world_record_pipeline_gpu::GpuReservoirConflictPredictor;

#[cfg(feature = "cuda")]
pub mod gpu_transfer_entropy;
#[cfg(feature = "cuda")]
pub use gpu_transfer_entropy::compute_transfer_entropy_ordering_gpu;

#[cfg(feature = "cuda")]
pub mod gpu_thermodynamic;
#[cfg(feature = "cuda")]
pub use gpu_thermodynamic::equilibrate_thermodynamic_gpu;

#[cfg(feature = "cuda")]
pub mod gpu_active_inference;
#[cfg(feature = "cuda")]
pub use gpu_active_inference::{active_inference_policy_gpu, ActiveInferencePolicy as GpuActiveInferencePolicy};

#[cfg(feature = "cuda")]
pub mod gpu_quantum_annealing;
#[cfg(feature = "cuda")]
pub use gpu_quantum_annealing::{gpu_qubo_simulated_annealing, qubo_solution_to_coloring, GpuQuboSolver};
