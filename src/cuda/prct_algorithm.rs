//! PRCT (Phase Resonance Chromatic-TSP) Module
//!
//! Full 3-layer integration: Neuromorphic → Quantum → Kuramoto → Phase-Guided Coloring

use anyhow::Result;
use prct_core::phase_guided_coloring;
use prct_core::ports::NeuromorphicEncodingParams;
use shared_types::*;
use std::sync::Arc;

use super::prct_adapters::{NeuromorphicAdapter, PhysicsCouplingAdapter, QuantumAdapter};

/// PRCT Algorithm Configuration
#[derive(Clone, Debug)]
pub struct PRCTConfig {
    /// Neuromorphic layer parameters
    pub neuro_base_frequency: f64,
    pub neuro_time_window: f64,

    /// Quantum layer parameters
    pub quantum_coupling_strength: f64,
    pub quantum_evolution_time: f64,

    /// Kuramoto synchronization parameters
    pub kuramoto_coupling: f64,
    pub kuramoto_steps: usize,

    /// Phase-guided coloring parameters
    pub target_colors: Option<usize>,
    pub coherence_threshold: f64,

    /// Enable GPU acceleration for PRCT
    pub gpu_accelerated: bool,
}

impl Default for PRCTConfig {
    fn default() -> Self {
        Self {
            neuro_base_frequency: 20.0,
            neuro_time_window: 100.0,
            quantum_coupling_strength: 1.0,
            quantum_evolution_time: 1.0,
            kuramoto_coupling: 1.0,
            kuramoto_steps: 100,
            target_colors: None,
            coherence_threshold: 0.5,
            gpu_accelerated: true,
        }
    }
}

/// PRCT Algorithm Engine with Full 3-Layer Pipeline
pub struct PRCTAlgorithm {
    config: PRCTConfig,
    neuro_adapter: Arc<NeuromorphicAdapter>,
    quantum_adapter: Arc<QuantumAdapter>,
    coupling_adapter: Arc<PhysicsCouplingAdapter>,
}

impl PRCTAlgorithm {
    /// Create new PRCT algorithm with default config
    pub fn new() -> Result<Self> {
        let neuro_adapter = Arc::new(NeuromorphicAdapter::new()?);
        let quantum_adapter = Arc::new(QuantumAdapter::new()?);
        let coupling_adapter = Arc::new(PhysicsCouplingAdapter::new(1.0)?);

        Ok(Self {
            config: PRCTConfig::default(),
            neuro_adapter,
            quantum_adapter,
            coupling_adapter,
        })
    }

    /// Create new PRCT algorithm with custom config
    pub fn with_config(config: PRCTConfig) -> Result<Self> {
        let neuro_adapter = Arc::new(NeuromorphicAdapter::new()?);
        let quantum_adapter = Arc::new(QuantumAdapter::new()?);
        let coupling_adapter = Arc::new(PhysicsCouplingAdapter::new(config.kuramoto_coupling)?);

        Ok(Self {
            config,
            neuro_adapter,
            quantum_adapter,
            coupling_adapter,
        })
    }

    /// Main PRCT coloring algorithm - Full 3-Layer Pipeline
    ///
    /// Pipeline: Graph → Neuromorphic → Quantum → Kuramoto → Phase-Guided Coloring
    pub fn color(&self, adjacency: &[Vec<usize>], ordering: &[usize]) -> Result<Vec<usize>> {
        log::info!("Starting full PRCT 3-layer pipeline");

        // Convert adjacency list to Graph structure
        let graph = self.build_graph_from_adjacency(adjacency);

        // ========================================
        // LAYER 1: NEUROMORPHIC PROCESSING
        // ========================================
        log::info!("Layer 1: Neuromorphic encoding and processing");
        let neuro_params = NeuromorphicEncodingParams {
            base_frequency: self.config.neuro_base_frequency,
            time_window: self.config.neuro_time_window,
            num_neurons: graph.num_vertices,
            enable_burst_encoding: true,
        };

        use prct_core::ports::NeuromorphicPort;
        let spike_pattern = self
            .neuro_adapter
            .encode_graph_as_spikes(&graph, &neuro_params)
            .map_err(|e| anyhow::anyhow!("Neuromorphic encoding failed: {:?}", e))?;
        let neuro_state = self
            .neuro_adapter
            .process_and_detect_patterns(&spike_pattern)
            .map_err(|e| anyhow::anyhow!("Neuromorphic processing failed: {:?}", e))?;
        log::info!("Neuromorphic coherence: {:.4}", neuro_state.coherence);

        // ========================================
        // LAYER 2: QUANTUM PROCESSING
        // ========================================
        log::info!("Layer 2: Quantum Hamiltonian construction and evolution");
        let evolution_params = EvolutionParams {
            dt: 0.01,
            strength: self.config.quantum_coupling_strength,
            damping: 0.1,
            temperature: 1.0,
        };

        use prct_core::ports::QuantumPort;
        let hamiltonian = self
            .quantum_adapter
            .build_hamiltonian(&graph, &evolution_params)
            .map_err(|e| anyhow::anyhow!("Hamiltonian construction failed: {:?}", e))?;
        log::info!(
            "Hamiltonian ground energy: {:.4}",
            hamiltonian.ground_state_energy
        );

        let initial_state = self
            .quantum_adapter
            .compute_ground_state(&hamiltonian)
            .map_err(|e| anyhow::anyhow!("Ground state computation failed: {:?}", e))?;
        let quantum_state = self
            .quantum_adapter
            .evolve_state(
                &hamiltonian,
                &initial_state,
                self.config.quantum_evolution_time,
            )
            .map_err(|e| anyhow::anyhow!("Quantum evolution failed: {:?}", e))?;
        log::info!("Quantum entanglement: {:.4}", quantum_state.entanglement);

        let phase_field = self
            .quantum_adapter
            .get_phase_field(&quantum_state)
            .map_err(|e| anyhow::anyhow!("Phase field extraction failed: {:?}", e))?;

        // ========================================
        // LAYER 2.5: KURAMOTO SYNCHRONIZATION
        // ========================================
        log::info!("Layer 2.5: Kuramoto phase synchronization");
        use prct_core::ports::PhysicsCouplingPort;
        let coupling = self
            .coupling_adapter
            .get_bidirectional_coupling(&neuro_state, &quantum_state)
            .map_err(|e| anyhow::anyhow!("Coupling computation failed: {:?}", e))?;
        log::info!(
            "Kuramoto order parameter: {:.4}",
            coupling.kuramoto_state.order_parameter
        );

        // ========================================
        // LAYER 3: PHASE-GUIDED COLORING
        // ========================================
        log::info!("Layer 3: Phase-guided chromatic coloring");
        let target_colors = self.config.target_colors.unwrap_or_else(|| {
            // Estimate from graph density
            let max_degree = adjacency.iter().map(|adj| adj.len()).max().unwrap_or(0);
            max_degree + 1
        });

        let solution = phase_guided_coloring(
            &graph,
            &phase_field,
            &coupling.kuramoto_state,
            target_colors,
        )
        .map_err(|e| anyhow::anyhow!("Phase-guided coloring failed: {:?}", e))?;

        log::info!(
            "PRCT coloring complete: {} colors",
            solution.chromatic_number
        );
        log::info!("Solution quality score: {:.4}", solution.quality_score);

        Ok(solution.colors)
    }

    /// Convert adjacency list to Graph structure
    fn build_graph_from_adjacency(&self, adjacency: &[Vec<usize>]) -> Graph {
        let n = adjacency.len();
        let mut adjacency_flat = vec![false; n * n];
        let mut edges = Vec::new();

        // Build flat adjacency matrix and edge list
        for (i, neighbors) in adjacency.iter().enumerate() {
            for &j in neighbors {
                if j < n && i < j {
                    adjacency_flat[i * n + j] = true;
                    adjacency_flat[j * n + i] = true;
                    edges.push((i, j, 1.0)); // Unit weight edges
                }
            }
        }

        // Count unique edges
        let num_edges = edges.len();

        Graph {
            num_vertices: n,
            num_edges,
            edges,
            adjacency: adjacency_flat,
            coordinates: None, // No coordinates for graph coloring problem
        }
    }

    /// Apply PRCT refinement pass
    ///
    /// Can be called multiple times to improve a coloring
    pub fn refine(&self, adjacency: &[Vec<usize>], coloring: &mut Vec<usize>) -> Result<usize> {
        // Re-run the full pipeline with current coloring as hint
        let improved = self.color(adjacency, &(0..adjacency.len()).collect::<Vec<_>>())?;

        // Check if improved is better
        let current_colors = *coloring.iter().max().unwrap_or(&0) + 1;
        let improved_colors = *improved.iter().max().unwrap_or(&0) + 1;

        if improved_colors < current_colors {
            *coloring = improved;
            log::info!(
                "PRCT refinement improved: {} -> {} colors",
                current_colors,
                improved_colors
            );
        }

        Ok(improved_colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prct_creation() {
        let prct = PRCTAlgorithm::new().unwrap();
        assert!(prct.config.gpu_accelerated);
    }

    #[test]
    fn test_prct_custom_config() {
        let config = PRCTConfig {
            neuro_base_frequency: 20.0,
            neuro_time_window: 100.0,
            quantum_coupling_strength: 1.0,
            quantum_evolution_time: 1.0,
            kuramoto_coupling: 1.0,
            kuramoto_steps: 100,
            target_colors: None,
            coherence_threshold: 0.5,
            gpu_accelerated: false,
        };

        let prct = PRCTAlgorithm::with_config(config).unwrap();
        assert_eq!(prct.config.kuramoto_steps, 100);
    }

    #[test]
    fn test_prct_coloring() {
        let prct = PRCTAlgorithm::new().unwrap();

        // Simple triangle graph
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];

        let ordering = vec![0, 1, 2];
        let coloring = prct.color(&adjacency, &ordering).unwrap();

        assert_eq!(coloring.len(), 3);

        // Verify valid coloring
        assert_ne!(coloring[0], coloring[1]);
        assert_ne!(coloring[1], coloring[2]);
        assert_ne!(coloring[0], coloring[2]);
    }
}
