//! CUDA GPU acceleration module

use anyhow::Result;

pub mod dense_path_guard;
pub mod device_guard;
pub mod ensemble_generation; // Fixed cudarc API compatibility
pub mod gpu_coloring;
pub mod prct_adapters;
pub mod prct_algorithm;
pub mod prct_gpu;
pub mod prism_pipeline; // Fixed cudarc API compatibility

pub use ensemble_generation::{Ensemble, GpuEnsembleGenerator};
pub use gpu_coloring::{GpuColoringEngine, GpuColoringResult};
pub use prct_algorithm::{PRCTAlgorithm, PRCTConfig};
pub use prism_pipeline::{
    PrismCoherence, PrismConfig as GpuPrismConfig, PrismPipeline as FullGpuPipeline,
};

use dense_path_guard::DensePathGuard;

pub struct EnsembleGenerator {
    num_replicas: usize,
    temperature: f32,
}

impl EnsembleGenerator {
    pub fn new(num_replicas: usize, temperature: f32) -> Result<Self> {
        Ok(Self {
            num_replicas,
            temperature,
        })
    }

    pub fn generate(&self, adjacency: &[Vec<usize>]) -> Result<Vec<Vec<usize>>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let n = adjacency.len();
        let mut orderings = Vec::with_capacity(self.num_replicas);
        let mut rng = thread_rng();

        // Generate diverse random orderings
        for i in 0..self.num_replicas {
            let mut ordering: Vec<usize> = (0..n).collect();

            if i == 0 {
                // First one: natural order (baseline)
                // Keep as is
            } else if i == 1 {
                // Second: reverse order
                ordering.reverse();
            } else if i == 2 {
                // Third: degree-based ordering (highest degree first)
                ordering.sort_by_key(|&v| std::cmp::Reverse(adjacency[v].len()));
            } else if i == 3 {
                // Fourth: degree-based ordering (lowest degree first)
                ordering.sort_by_key(|&v| adjacency[v].len());
            } else {
                // Rest: random permutations
                ordering.shuffle(&mut rng);
            }

            orderings.push(ordering);
        }

        Ok(orderings)
    }
}

pub struct GPUColoring;

impl GPUColoring {
    pub fn color(&self, adjacency: &[Vec<usize>], ordering: &[usize]) -> Result<Vec<usize>> {
        // Greedy coloring following the given ordering
        let n = adjacency.len();
        let mut coloring = vec![0; n];

        // Use the ordering to assign colors
        for &vertex in ordering {
            if vertex >= n {
                continue;
            }

            // Find colors used by neighbors
            let mut used_colors = vec![false; n];
            for &neighbor in &adjacency[vertex] {
                if neighbor < n && neighbor != vertex {
                    used_colors[coloring[neighbor]] = true;
                }
            }

            // Assign smallest available color
            for color in 0..n {
                if !used_colors[color] {
                    coloring[vertex] = color;
                    break;
                }
            }
        }

        Ok(coloring)
    }
}

pub struct PRISMPipeline {
    config: crate::PrismConfig,
}

impl PRISMPipeline {
    pub fn new(config: crate::PrismConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn run(&self, adjacency: Vec<Vec<usize>>) -> Result<Vec<usize>> {
        // Placeholder pipeline
        let n = adjacency.len();
        let edges = adjacency.iter().map(|row| row.len()).sum::<usize>() / 2;
        let guard = DensePathGuard::new();
        guard.check_feasibility(n, edges);
        let n = adjacency.len();
        Ok(vec![0; n])
    }
}
