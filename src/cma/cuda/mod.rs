//! CUDA kernels for Phase 6 CMA
//!
//! Contains GPU implementations of:
//! - KSG transfer entropy computation
//! - Path integral Monte Carlo
//! - Quantum annealing operations
//!
//! # Build Process
//! CUDA kernels are compiled to PTX during build.rs execution

// Module declaration for CUDA kernel files
// The actual kernels are in .cu files compiled to PTX