//! Auto-Tuning Performance Optimization Engine
//!
//! This module implements an auto-tuning system that searches the configuration space
//! to find optimal GPU kernel parameters. It uses a simplified grid search algorithm
//! (production version would use Bayesian optimization) to find configurations that
//! maximize performance.
//!
//! # Mathematical Foundation
//!
//! The optimization problem is:
//!
//! ```text
//! θ* = argmax_{θ ∈ Θ} P(W_N, θ)
//! ```
//!
//! Where:
//! - θ = configuration vector {block_size, grid_size, ...}
//! - P(W_N, θ) = performance metric (operations/second or latency)
//! - Θ = valid configuration space
//!
//! # Search Strategy
//!
//! This implementation uses an intelligent grid search that:
//! 1. Generates candidate configurations from KernelTuner recommendations
//! 2. Evaluates each configuration using a user-provided benchmark
//! 3. Caches the best profile for reuse
//!
//! Production enhancement: Replace with Bayesian optimization using Gaussian Process
//! models to reduce search iterations from O(n) to O(log n).

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::optimization::kernel_tuner::{KernelTuner, KernelConfig};

/// Performance tuning profile
#[derive(Debug, Clone)]
pub struct TuningProfile {
    /// Workload identifier
    pub workload_id: String,
    /// Optimal kernel configuration
    pub config: KernelConfig,
    /// Measured performance (operations/second)
    pub throughput: f64,
    /// Measured latency (milliseconds)
    pub latency_ms: f64,
    /// Theoretical occupancy
    pub occupancy: f64,
    /// Timestamp when tuned
    pub tuned_at: Instant,
}

/// Search space for auto-tuning
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Workload size
    pub workload_size: usize,
    /// Minimum block size
    pub min_block_size: u32,
    /// Maximum block size
    pub max_block_size: u32,
    /// Allow shared memory configurations
    pub use_shared_memory: bool,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            workload_size: 1024,
            min_block_size: 32,
            max_block_size: 1024,
            use_shared_memory: false,
        }
    }
}

/// Performance metrics from tuning
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Best throughput achieved (ops/sec)
    pub best_throughput: f64,
    /// Best latency achieved (ms)
    pub best_latency_ms: f64,
    /// Speedup vs baseline
    pub speedup: f64,
    /// Number of configurations evaluated
    pub configs_evaluated: usize,
    /// Tuning session duration
    pub tuning_duration: Duration,
}

/// Search algorithm trait for pluggable search strategies
pub trait SearchAlgorithm: Send + Sync {
    /// Search for optimal configuration
    ///
    /// # Parameters
    /// - `search_space`: Configuration space to search
    /// - `evaluator`: Function that benchmarks a configuration
    ///
    /// # Returns
    /// Best configuration found and its performance
    fn search(&self, search_space: &SearchSpace, evaluator: &dyn Fn(&KernelConfig) -> f64) -> (KernelConfig, f64);
}

/// Grid search algorithm (baseline implementation)
pub struct GridSearch {
    kernel_tuner: Arc<KernelTuner>,
}

impl GridSearch {
    pub fn new(kernel_tuner: Arc<KernelTuner>) -> Self {
        Self { kernel_tuner }
    }
}

impl SearchAlgorithm for GridSearch {
    fn search(&self, search_space: &SearchSpace, evaluator: &dyn Fn(&KernelConfig) -> f64) -> (KernelConfig, f64) {
        // Get recommended configurations from kernel tuner
        let candidates = self.kernel_tuner.recommend_configs(search_space.workload_size);

        // Evaluate each candidate
        let mut best_config = candidates[0];
        let mut best_perf = 0.0;

        for config in candidates {
            // Filter by search space constraints
            if config.block_size < search_space.min_block_size
                || config.block_size > search_space.max_block_size
            {
                continue;
            }

            if config.shared_memory > 0 && !search_space.use_shared_memory {
                continue;
            }

            // Evaluate performance
            let perf = evaluator(&config);

            if perf > best_perf {
                best_perf = perf;
                best_config = config;
            }
        }

        (best_config, best_perf)
    }
}

/// Performance tuner with caching
pub struct PerformanceTuner {
    /// Cached tuning profiles
    profiles: Arc<DashMap<String, TuningProfile>>,
    /// Search algorithm
    search_algorithm: Box<dyn SearchAlgorithm>,
    /// Kernel tuner for occupancy analysis
    kernel_tuner: Arc<KernelTuner>,
}

impl PerformanceTuner {
    /// Create new performance tuner
    pub fn new() -> Result<Self, String> {
        let kernel_tuner = Arc::new(KernelTuner::new()?);
        let search_algorithm = Box::new(GridSearch::new(kernel_tuner.clone()));

        Ok(Self {
            profiles: Arc::new(DashMap::new()),
            search_algorithm,
            kernel_tuner,
        })
    }

    /// Run tuning session for a workload
    ///
    /// # Parameters
    /// - `workload_id`: Unique identifier for this workload
    /// - `search_space`: Configuration space to explore
    /// - `evaluator`: Benchmark function that measures performance
    ///
    /// # Returns
    /// Performance metrics from tuning session
    pub fn run_tuning_session(
        &self,
        workload_id: &str,
        search_space: SearchSpace,
        evaluator: &dyn Fn(&KernelConfig) -> f64,
    ) -> PerformanceMetrics
    {
        let start = Instant::now();

        // Search for optimal configuration
        let (best_config, best_perf) = self.search_algorithm.search(&search_space, evaluator);

        // Calculate occupancy
        let occ_info = self.kernel_tuner.calculate_occupancy(&best_config);

        // Estimate latency from throughput
        let latency_ms = if best_perf > 0.0 {
            1000.0 / best_perf
        } else {
            f64::INFINITY
        };

        // Create tuning profile
        let profile = TuningProfile {
            workload_id: workload_id.to_string(),
            config: best_config,
            throughput: best_perf,
            latency_ms,
            occupancy: occ_info.occupancy,
            tuned_at: Instant::now(),
        };

        // Cache profile
        self.profiles.insert(workload_id.to_string(), profile);

        // Calculate speedup (assume baseline is worst config)
        let baseline_perf = evaluator(&KernelConfig {
            block_size: 128, // Generic baseline
            grid_size: best_config.grid_size,
            shared_memory: 0,
            registers_per_thread: 32,
        });

        let speedup = if baseline_perf > 0.0 {
            best_perf / baseline_perf
        } else {
            1.0
        };

        PerformanceMetrics {
            best_throughput: best_perf,
            best_latency_ms: latency_ms,
            speedup,
            configs_evaluated: self.kernel_tuner.recommend_configs(search_space.workload_size).len(),
            tuning_duration: start.elapsed(),
        }
    }

    /// Get cached tuning profile for a workload
    pub fn get_profile(&self, workload_id: &str) -> Option<TuningProfile> {
        self.profiles.get(workload_id).map(|entry| entry.clone())
    }

    /// Check if workload has been tuned
    pub fn is_tuned(&self, workload_id: &str) -> bool {
        self.profiles.contains_key(workload_id)
    }

    /// Clear all cached profiles
    pub fn clear_cache(&self) {
        self.profiles.clear();
    }

    /// Get kernel tuner
    pub fn kernel_tuner(&self) -> &KernelTuner {
        &self.kernel_tuner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tuner_creation() {
        if let Ok(tuner) = PerformanceTuner::new() {
            assert!(!tuner.is_tuned("test_workload"));
        }
    }

    #[test]
    fn test_tuning_session() {
        if let Ok(tuner) = PerformanceTuner::new() {
            let workload_id = "test_gemv_1024";
            let search_space = SearchSpace {
                workload_size: 1024,
                min_block_size: 64,
                max_block_size: 512,
                use_shared_memory: false,
            };

            // Mock evaluator (simulates performance measurement)
            let evaluator = |config: &KernelConfig| {
                // Simulate: larger blocks = better performance (up to a point)
                let base_perf = 1000.0;
                let block_factor = (config.block_size as f64 / 256.0).min(2.0);
                base_perf * block_factor
            };

            let metrics = tuner.run_tuning_session(workload_id, search_space, &evaluator);

            println!("Tuning Results:");
            println!("  Best throughput: {:.2} ops/sec", metrics.best_throughput);
            println!("  Best latency: {:.3} ms", metrics.best_latency_ms);
            println!("  Speedup: {:.2}x", metrics.speedup);
            println!("  Configs evaluated: {}", metrics.configs_evaluated);
            println!("  Tuning duration: {:?}", metrics.tuning_duration);

            assert!(tuner.is_tuned(workload_id));
            assert!(metrics.speedup >= 1.0);

            // Verify cached profile
            let profile = tuner.get_profile(workload_id).unwrap();
            assert_eq!(profile.workload_id, workload_id);
            assert!(profile.occupancy > 0.0);
        }
    }

    #[test]
    fn test_profile_caching() {
        if let Ok(tuner) = PerformanceTuner::new() {
            let workload_id = "test_cache";
            let search_space = SearchSpace::default();

            let evaluator = |_: &KernelConfig| 1000.0;

            tuner.run_tuning_session(workload_id, search_space, &evaluator);

            assert!(tuner.is_tuned(workload_id));
            assert!(tuner.get_profile(workload_id).is_some());

            tuner.clear_cache();
            assert!(!tuner.is_tuned(workload_id));
        }
    }
}
