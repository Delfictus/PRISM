//! Quantum Annealing Module for Phase 6 CMA
//!
//! Contains real quantum annealing implementations:
//! - Path Integral Monte Carlo (CPU)
//! - GPU-accelerated PIMC
//! - Spectral gap tracking
//! - Adaptive scheduling
//!
//! # Constitution Reference
//! Phase 6 Implementation Constitution - Sprint 1.3

pub mod path_integral;
pub mod pimc_gpu;

pub use path_integral::{PathIntegralMonteCarlo, ProblemHamiltonian};
pub use pimc_gpu::GpuPathIntegralMonteCarlo;