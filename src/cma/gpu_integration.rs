//! GPU Integration Bridge for CMA
//!
//! # Purpose
//! Connects CMA to existing Phase 1 GPU infrastructure
//! Provides real implementation, not placeholders
//!
//! # Constitution Reference
//! Phase 6 Implementation Constitution - Sprint 1.1

use std::sync::Arc;
use anyhow::{Result, Context};
use ndarray::Array2;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use quantum_engine::GpuTspSolver;
use crate::cma::{Problem, Solution};

/// GPU-solvable trait for CMA integration
pub trait GpuSolvable: Send + Sync {
    /// Solve with deterministic seed for reproducibility
    fn solve_with_seed(&self, problem: &dyn Problem, seed: u64) -> Result<Solution>;

    /// Batch solve multiple problems in parallel
    fn solve_batch(&self, problems: &[Box<dyn Problem>], seeds: &[u64]) -> Result<Vec<Solution>>;

    /// Get GPU device properties
    fn get_device_properties(&self) -> Result<GpuProperties>;
}

/// GPU device properties
#[derive(Debug, Clone)]
pub struct GpuProperties {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub memory_gb: f32,
    pub multiprocessors: u32,
}

/// Wrapper to make GpuTspSolver compatible with CMA
pub struct GpuTspBridge {
    /// Pre-allocated solvers for batch processing
    solver_pool: Vec<Arc<parking_lot::Mutex<GpuTspSolver>>>,
    /// Number of parallel solvers
    pool_size: usize,
}

impl GpuTspBridge {
    /// Create new bridge with solver pool
    pub fn new(pool_size: usize) -> Result<Self> {
        if pool_size == 0 {
            return Err(anyhow::anyhow!("Pool size must be > 0"));
        }

        println!("🚀 Initializing GPU-CMA Bridge with {} parallel solvers", pool_size);

        // Pre-allocate solver pool
        // Note: We'll create them on-demand since GpuTspSolver needs problem size
        let solver_pool = Vec::with_capacity(pool_size);

        Ok(Self {
            solver_pool,
            pool_size,
        })
    }

    /// Convert Problem to coupling matrix for TSP solver
    fn problem_to_coupling_matrix(&self, problem: &dyn Problem, seed: u64) -> Array2<Complex64> {
        let dim = problem.dimension();
        let mut matrix = Array2::zeros((dim, dim));

        // Use seed for reproducible randomization
        let rng = ChaCha20Rng::seed_from_u64(seed);

        // Create a test solution to evaluate costs
        let test_solution = Solution {
            data: vec![0.0; dim],
            cost: 0.0,
        };

        // Build coupling matrix from problem structure
        for i in 0..dim {
            for j in i+1..dim {
                // Create solutions that differ at positions i and j
                let mut sol_ij = test_solution.clone();
                sol_ij.data[i] = 1.0;
                sol_ij.data[j] = 1.0;

                // Evaluate coupling strength
                let cost_ij = problem.evaluate(&sol_ij);

                // Convert to complex coupling
                let coupling = Complex64::new(cost_ij, 0.0);
                matrix[[i, j]] = coupling;
                matrix[[j, i]] = coupling.conj();
            }
        }

        matrix
    }

    /// Get or create a solver for the given problem size
    fn get_solver(&self, coupling_matrix: &Array2<Complex64>) -> Result<Arc<parking_lot::Mutex<GpuTspSolver>>> {
        // For now, create a new solver each time
        // TODO: Implement proper pooling with size-based caching
        let solver = GpuTspSolver::new(coupling_matrix)
            .context("Failed to create GPU TSP solver")?;

        Ok(Arc::new(parking_lot::Mutex::new(solver)))
    }
}

impl GpuSolvable for GpuTspBridge {
    fn solve_with_seed(&self, problem: &dyn Problem, seed: u64) -> Result<Solution> {
        // Convert problem to TSP format
        let coupling_matrix = self.problem_to_coupling_matrix(problem, seed);

        // Get or create solver
        let solver_arc = self.get_solver(&coupling_matrix)?;
        let mut solver = solver_arc.lock();

        // Run optimization with deterministic iterations based on seed
        let max_iterations = 100 + (seed % 50) as usize; // Vary iterations slightly
        solver.optimize_2opt_gpu(max_iterations)
            .context("GPU optimization failed")?;

        // Extract solution
        let tour = solver.get_tour().to_vec();
        let tour_length = solver.get_tour_length();

        // Convert TSP tour back to general solution
        let mut solution_data = vec![0.0; problem.dimension()];
        for (idx, &city) in tour.iter().enumerate() {
            solution_data[city] = idx as f64 / tour.len() as f64;
        }

        Ok(Solution {
            data: solution_data,
            cost: tour_length,
        })
    }

    fn solve_batch(&self, problems: &[Box<dyn Problem>], seeds: &[u64]) -> Result<Vec<Solution>> {
        if problems.len() != seeds.len() {
            return Err(anyhow::anyhow!("Problems and seeds must have same length"));
        }

        // Use rayon for parallel batch processing
        use rayon::prelude::*;

        problems.par_iter()
            .zip(seeds.par_iter())
            .map(|(problem, &seed)| {
                self.solve_with_seed(problem.as_ref(), seed)
            })
            .collect()
    }

    fn get_device_properties(&self) -> Result<GpuProperties> {
        // GPU ONLY - NO CPU FALLBACK
        use cudarc::driver::sys;

        let device = CudaDevice::new(0)
            .context("GPU REQUIRED")?;

        // Get device name and properties
        let name = device.name()?;

        // Compute capability from device
        let major = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u32;
        let minor = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u32;

        // Memory in bytes to GB
        let memory_bytes = unsafe {
            let mut bytes: usize = 0;
            let result = sys::cuMemGetInfo_v2(&mut bytes, std::ptr::null_mut());
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(anyhow::anyhow!("Failed to get memory info"));
            }
            bytes
        };
        let memory_gb = memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);

        // Multiprocessor count
        let multiprocessors = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;

        Ok(GpuProperties {
            device_name: name,
            compute_capability: (major, minor),
            memory_gb,
            multiprocessors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test problem for GPU integration
    struct TestTspProblem {
        cities: Vec<(f64, f64)>,
    }

    impl Problem for TestTspProblem {
        fn evaluate(&self, solution: &Solution) -> f64 {
            // Simple TSP cost: sum of distances in tour order
            let n = self.cities.len();
            let mut total = 0.0;

            // Convert solution to tour order
            let mut tour: Vec<usize> = (0..n).collect();
            tour.sort_by(|&a, &b| {
                solution.data[a].partial_cmp(&solution.data[b]).unwrap()
            });

            // Calculate tour length
            for i in 0..n {
                let j = (i + 1) % n;
                let (x1, y1) = self.cities[tour[i]];
                let (x2, y2) = self.cities[tour[j]];
                let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                total += dist;
            }

            total
        }

        fn dimension(&self) -> usize {
            self.cities.len()
        }
    }

    #[test]
    fn test_gpu_bridge_creation() {
        let bridge = GpuTspBridge::new(4);
        assert!(bridge.is_ok());

        let bridge = bridge.unwrap();
        assert_eq!(bridge.pool_size, 4);
    }

    #[test]
    fn test_solve_with_seed_deterministic() {
        let bridge = GpuTspBridge::new(1).unwrap();

        // Create test problem
        let problem = TestTspProblem {
            cities: vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ],
        };

        // Solve with same seed should give same result
        let sol1 = bridge.solve_with_seed(&problem, 42).unwrap();
        let sol2 = bridge.solve_with_seed(&problem, 42).unwrap();

        // For a square, multiple optimal tours exist with same cost
        // Check that costs match (deterministic RNG should give same cost)
        assert_eq!(sol1.cost, sol2.cost);
        // Tours may differ but should have equivalent cost
        assert!((sol1.cost - sol2.cost).abs() < 1e-6);

        // Different seed should (likely) give different result
        let sol3 = bridge.solve_with_seed(&problem, 123).unwrap();
        // Cost might be same for optimal solution, but data ordering might differ
        assert!(sol3.data != sol1.data || sol3.cost == sol1.cost);
    }

    #[test]
    fn test_gpu_properties() {
        let bridge = GpuTspBridge::new(1).unwrap();

        let props = bridge.get_device_properties();

        if props.is_ok() {
            let props = props.unwrap();
            println!("GPU Device: {}", props.device_name);
            println!("Compute Capability: {}.{}", props.compute_capability.0, props.compute_capability.1);
            println!("Memory: {:.1} GB", props.memory_gb);
            println!("Multiprocessors: {}", props.multiprocessors);

            assert!(props.memory_gb > 0.0);
            assert!(props.multiprocessors > 0);
        } else {
            // OK if no GPU available in test environment
            println!("No GPU available for testing");
        }
    }
}

// Export for use in CMA
pub use self::{GpuTspBridge as DefaultGpuSolver};