//! PRISM-AI: GPU-accelerated graph coloring with neuromorphic and quantum computing
//!
//! This library implements the complete PRISM-AI pipeline for graph coloring,
//! targeting world-record performance on DIMACS benchmark graphs.

pub mod cma;
pub mod cuda;
pub mod data;
pub mod features;
pub mod governance;
pub mod meta;
pub mod neuromorphic;
pub mod phase6;
pub mod quantum;
pub mod telemetry;

// Re-export key types
pub use cma::{neural::ColoringGNN, CMAAdapter};
pub use cuda::{EnsembleGenerator, GPUColoring, PRISMPipeline};
pub use data::{DIMACParser, GraphGenerator};
pub use governance::{BenchmarkManifest, DeterminismProof, DeterminismRecorder, PerformanceGate};

// Feature flags for conditional compilation
#[cfg(feature = "cuda")]
pub use cuda::gpu_coloring;

#[cfg(feature = "protein_folding")]
pub mod protein;

/// Main error type for PRISM-AI
#[derive(Debug, thiserror::Error)]
pub enum PrismError {
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Graph error: {0}")]
    GraphError(String),

    #[error("Neural network error: {0}")]
    NeuralError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, PrismError>;

/// Configuration for the PRISM pipeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrismConfig {
    /// Number of ensemble replicas to generate
    pub num_replicas: usize,

    /// Temperature parameter for thermodynamic sampling
    pub temperature: f32,

    /// Maximum iterations for optimization
    pub max_iterations: usize,

    /// Enable GPU acceleration
    pub use_gpu: bool,

    /// Enable neural network predictions
    pub use_gnn: bool,

    /// Target number of colors (if known)
    pub target_colors: Option<usize>,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            num_replicas: 1000,
            temperature: 1.0,
            max_iterations: 10000,
            use_gpu: cfg!(feature = "cuda"),
            use_gnn: true,
            target_colors: None,
        }
    }
}

/// Main entry point for the PRISM-AI pipeline
pub struct PrismAI {
    config: PrismConfig,
    #[cfg(feature = "cuda")]
    pipeline: Option<cuda::PRISMPipeline>,
}

impl PrismAI {
    pub fn new(config: PrismConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let pipeline = if config.use_gpu {
            Some(cuda::PRISMPipeline::new(config.clone()).map_err(|e| PrismError::Other(e))?)
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "cuda")]
            pipeline,
        })
    }

    /// Color a graph using the full PRISM-AI pipeline
    pub fn color_graph(&self, adjacency: Vec<Vec<usize>>) -> Result<Vec<usize>> {
        #[cfg(feature = "cuda")]
        if let Some(ref pipeline) = self.pipeline {
            return pipeline.run(adjacency).map_err(|e| PrismError::Other(e));
        }

        // Fallback to CPU implementation
        self.color_graph_cpu(adjacency)
    }

    /// Color a graph and capture determinism proof data.
    pub fn color_graph_with_proof(
        &self,
        adjacency: Vec<Vec<usize>>,
        master_seed: u64,
    ) -> Result<(Vec<usize>, DeterminismProof)> {
        let mut recorder = governance::DeterminismRecorder::new(master_seed);
        recorder.record_input(&adjacency)?;
        let colors = self.color_graph(adjacency)?;
        recorder.record_output(&colors)?;
        let proof = recorder.finalize();
        Ok((colors, proof))
    }

    fn color_graph_cpu(&self, adjacency: Vec<Vec<usize>>) -> Result<Vec<usize>> {
        // Simplified CPU implementation for testing
        let n = adjacency.len();
        let edges = adjacency.iter().map(|row| row.len()).sum::<usize>() / 2;

        let guard = crate::cuda::dense_path_guard::DensePathGuard::new();
        let _decision = guard.check_feasibility(n, edges);

        let mut colors = vec![0; n];

        for v in 0..n {
            let mut used = vec![false; n];
            for &neighbor in &adjacency[v] {
                if neighbor < v {
                    used[colors[neighbor]] = true;
                }
            }

            for c in 0..n {
                if !used[c] {
                    colors[v] = c;
                    break;
                }
            }
        }

        Ok(colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_coloring() {
        let config = PrismConfig::default();
        let prism = PrismAI::new(config).unwrap();

        // Triangle graph (K3)
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];

        let colors = prism.color_graph(adjacency).unwrap();
        assert_eq!(colors.len(), 3);

        // Verify valid coloring
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[0], colors[2]);
        assert_ne!(colors[1], colors[2]);
    }

    #[test]
    fn test_color_graph_with_proof() {
        let config = PrismConfig::default();
        let prism = PrismAI::new(config).unwrap();
        let adjacency = vec![vec![1], vec![0]];

        let (colors, proof) = prism.color_graph_with_proof(adjacency, 123).unwrap();
        assert_eq!(colors.len(), 2);
        assert_eq!(proof.master_seed, 123);
        assert!(!proof.input_hash.is_empty());
        assert!(!proof.output_hash.is_empty());
    }
}
