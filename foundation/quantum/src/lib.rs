//! Quantum-Inspired Computing Engine
//!
//! World's first software-based quantum Hamiltonian operator for optimization
//! Implements complete PRCT (Phase Resonance Chromatic-TSP) Algorithm

pub mod types;
pub mod hamiltonian;
pub mod security;
pub mod robust_eigen;
pub mod prct_coloring;
pub mod gpu_coloring;
pub mod prct_tsp;
pub mod gpu_tsp;
pub mod qubo;

// Re-export main types
pub use types::*;
pub use hamiltonian::{Hamiltonian, calculate_ground_state, PhaseResonanceField, PRCTDiagnostics};
pub use security::{SecurityValidator, SecurityError};
pub use robust_eigen::{RobustEigenSolver, RobustEigenConfig, EigenDiagnostics, SolverMethod};
pub use prct_coloring::ChromaticColoring;
pub use gpu_coloring::GpuChromaticColoring;
pub use prct_tsp::TSPPathOptimizer;
pub use gpu_tsp::GpuTspSolver;
pub use qubo::GpuQuboSolver;pub mod gpu_k_opt;
pub use gpu_k_opt::GpuKOpt;
