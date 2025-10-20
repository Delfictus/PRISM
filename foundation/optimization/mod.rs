//! Performance Optimization Module
//!
//! This module implements dynamic, self-tuning performance optimization for GPU workloads.
//! It provides auto-tuning capabilities, kernel occupancy analysis, and memory pipeline
//! optimization to maximize hardware utilization and meet strict latency SLOs.
//!
//! # Architecture
//!
//! The optimization system consists of three main components:
//!
//! 1. **PerformanceTuner**: Auto-tuning engine using Bayesian optimization
//! 2. **KernelTuner**: Hardware-aware kernel configuration and occupancy analysis
//! 3. **MemoryOptimizer**: Triple-buffered memory pipeline with CUDA streams
//!
//! # Mathematical Foundation
//!
//! ## Optimization Problem
//!
//! For a GPU workload W with input size N, find the optimal configuration θ*:
//!
//! ```text
//! θ* = argmax_{θ ∈ Θ} P(W_N, θ)
//! ```
//!
//! Where:
//! - θ = {block_size, grid_size, shared_memory, ...}
//! - P(W_N, θ) = performance metric (throughput or latency)
//! - Θ = valid configuration space for target GPU architecture
//!
//! ## Occupancy Model
//!
//! GPU occupancy is the ratio of active warps to maximum possible warps:
//!
//! ```text
//! Occupancy = w_active / w_max
//! ```
//!
//! This metric is crucial for hiding memory latency and maximizing throughput.
//!
//! # Design Principles
//!
//! - **Hardware-Aware**: All optimizations respect GPU architecture constraints
//! - **Profile-Based**: Cache optimal configurations for reuse
//! - **Lock-Free**: Concurrent access via DashMap
//! - **Validated**: >2x speedup target with >80% GPU utilization
//!
//! # Constitution Compliance
//!
//! This module implements Phase 4 Task 4.2 requirements:
//! - Auto-tuning achieves >2x speedup
//! - GPU utilization >80%
//! - Latency SLO conformance
//! - Production-grade error handling

pub mod performance_tuner;
pub mod kernel_tuner;
pub mod memory_optimizer;

pub use performance_tuner::{
    PerformanceTuner,
    TuningProfile,
    SearchAlgorithm,
    SearchSpace,
    PerformanceMetrics,
};

pub use kernel_tuner::{
    KernelTuner,
    GpuProperties,
    KernelConfig,
    OccupancyInfo,
};

pub use memory_optimizer::{
    MemoryOptimizer,
    PinnedMemoryPool,
    PipelineStats,
};
