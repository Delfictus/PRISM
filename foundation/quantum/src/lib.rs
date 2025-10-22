//! Quantum-Inspired Computing Engine
//!
//! World's first software-based quantum Hamiltonian operator for optimization
//! Implements complete PRCT (Phase Resonance Chromatic-TSP) Algorithm

pub mod gpu_coloring;
pub mod gpu_tsp;
pub mod hamiltonian;
pub mod prct_coloring;
pub mod prct_tsp;
pub mod qubo;
pub mod robust_eigen;
pub mod security;
pub mod types;

// Re-export main types
pub use gpu_coloring::GpuChromaticColoring;
pub use gpu_tsp::GpuTspSolver;
pub use hamiltonian::{calculate_ground_state, Hamiltonian, PRCTDiagnostics, PhaseResonanceField};
pub use prct_coloring::ChromaticColoring;
pub use prct_tsp::TSPPathOptimizer;
pub use qubo::GpuQuboSolver;
pub use robust_eigen::{EigenDiagnostics, RobustEigenConfig, RobustEigenSolver, SolverMethod};
pub use security::{SecurityError, SecurityValidator};
pub use types::*;
pub mod gpu_k_opt;
pub use gpu_k_opt::GpuKOpt;
