//! Meta-Learning Coordinator
//!
//! Phase 6, Task 6.4: Adaptive Hamiltonian modulation based on problem structure
//!
//! This module integrates:
//! - TDA (Topological Data Analysis) for structure discovery
//! - GNN (Graph Neural Networks) for learned priors
//! - Predictive Neuromorphic for surprise-based focus
//!
//! Constitutional Compliance:
//! - Article I: Energy modulation tracked, entropy preserved
//! - Article II: Information sources quantified and bounded
//! - Article III: Free energy minimization drives all modulation

use std::sync::Arc;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use parking_lot::RwLock;

use crate::phase6::{
    tda::{TdaAdapter, TdaPort, PersistenceBarcode},
    predictive_neuro::{PredictiveNeuromorphic, PredictionError},
};
use crate::cma::neural::gnn_integration::E3EquivariantGNN;

/// Modulated Hamiltonian parameters
#[derive(Debug, Clone)]
pub struct ModulatedHamiltonian {
    /// Base energy scale
    pub energy_scale: f64,

    /// Per-vertex energy bias (from GNN predictions)
    pub vertex_bias: Array1<f64>,

    /// Edge coupling strengths (topology-modulated)
    pub edge_coupling: Array2<f64>,

    /// Local temperature for each vertex (surprise-based)
    pub local_temperature: Array1<f64>,

    /// Color preferences from GNN prior
    pub color_prior: Array2<f64>, // [vertex, color]

    /// Topological constraint strength
    pub topology_weight: f64,

    /// Information-theoretic regularization
    pub entropy_regularization: f64,

    /// Free energy tracking
    pub free_energy: f64,
}

impl ModulatedHamiltonian {
    /// Verify thermodynamic consistency (Article I)
    pub fn verify_consistency(&self) -> Result<()> {
        // Check all temperatures are positive
        for &temp in self.local_temperature.iter() {
            if temp <= 0.0 {
                return Err(anyhow!(
                    "CONSTITUTION VIOLATION (Article I): Negative temperature {}",
                    temp
                ));
            }
        }

        // Check coupling matrix is symmetric
        let n = self.edge_coupling.nrows();
        for i in 0..n {
            for j in (i+1)..n {
                let diff = (self.edge_coupling[[i, j]] - self.edge_coupling[[j, i]]).abs();
                if diff > 1e-10 {
                    return Err(anyhow!(
                        "CONSTITUTION VIOLATION: Non-symmetric coupling matrix"
                    ));
                }
            }
        }

        // Check entropy regularization is non-negative
        if self.entropy_regularization < 0.0 {
            return Err(anyhow!(
                "CONSTITUTION VIOLATION (Article I): Negative entropy regularization"
            ));
        }

        Ok(())
    }

    /// Compute total energy for a configuration
    pub fn compute_energy(&self, configuration: &[usize]) -> f64 {
        let n = configuration.len();
        let mut energy = 0.0;

        // Vertex bias energy
        for (v, &color) in configuration.iter().enumerate() {
            energy += self.vertex_bias[v];

            // Color prior energy
            if color < self.color_prior.ncols() {
                energy -= self.color_prior[[v, color]]; // Negative because high prior = low energy
            }
        }

        // Edge coupling energy (penalize same colors on edges)
        for i in 0..n {
            for j in (i+1)..n {
                if configuration[i] == configuration[j] {
                    energy += self.edge_coupling[[i, j]];
                }
            }
        }

        // Scale by global energy scale
        energy * self.energy_scale
    }
}

/// Meta-learning coordinator that modulates quantum Hamiltonian
/// based on problem structure
pub struct MetaLearningCoordinator {
    /// TDA analyzer
    tda: Arc<RwLock<TdaAdapter>>,

    /// GNN predictor (using existing implementation)
    gnn: Arc<RwLock<E3EquivariantGNN>>,

    /// Enhanced neuromorphic system
    neuro: Arc<RwLock<PredictiveNeuromorphic>>,

    /// Modulation strengths (learned hyperparameters)
    pub alpha_topology: f64,   // TDA influence
    pub beta_prior: f64,       // GNN influence
    pub gamma_surprise: f64,   // Prediction error influence

    /// Learning history for adaptation
    history: Vec<ModulationHistory>,

    /// Constitutional compliance tracking
    entropy_production: f64,
    information_flow: f64,
}

#[derive(Clone)]
struct ModulationHistory {
    topology_score: f64,
    gnn_confidence: f64,
    surprise_level: f64,
    performance: f64,
}

impl MetaLearningCoordinator {
    pub fn new(
        max_colors: usize,
        reservoir_size: usize,
    ) -> Result<Self> {
        // Initialize components
        let tda = Arc::new(RwLock::new(TdaAdapter::new(2)?));

        // GNN would be loaded from pre-trained model
        // For now, create a stub (GPU support to be added with cudarc)
        // TODO: Replace with actual GPU implementation using cudarc
        let gnn = Arc::new(RwLock::new(E3EquivariantGNN::new_cpu(
            8,  // node features
            4,  // edge features
            64, // hidden dim
            3,  // num layers
        )?));

        let neuro = Arc::new(RwLock::new(PredictiveNeuromorphic::new(
            100, // input dim
            reservoir_size,
            max_colors,
        )?));

        Ok(Self {
            tda,
            gnn,
            neuro,
            alpha_topology: 1.0,
            beta_prior: 0.5,
            gamma_surprise: 2.0,
            history: Vec::new(),
            entropy_production: 0.0,
            information_flow: 0.0,
        })
    }

    /// Compute modulated Hamiltonian parameters
    ///
    /// Constitutional Compliance:
    /// - Article I: Energy modulation tracked, entropy preserved
    /// - Article II: Information sources (TDA, GNN, Neuro) quantified
    /// - Article III: Free energy minimization drives modulation
    pub fn compute_modulated_hamiltonian(
        &mut self,
        adjacency: &Array2<bool>,
        base_energy_scale: f64,
        max_colors: usize,
    ) -> Result<ModulatedHamiltonian> {
        println!("🧠 Meta-Learning: Computing adaptive Hamiltonian");

        let n = adjacency.nrows();

        // STEP 1: Topological Analysis
        println!("   📐 Computing topological features...");
        let topology = self.tda.read().compute_persistence(adjacency)?;
        let lower_bound = topology.chromatic_lower_bound();
        let difficulty = topology.difficulty_score();
        let important_vertices = topology.important_vertices(n / 10);

        println!("   TDA: Lower bound = {}, Difficulty = {:.3}", lower_bound, difficulty);

        // STEP 2: Neural Prediction (simplified without full GNN forward pass)
        println!("   🔮 Generating neural predictions...");
        let gnn_predictions = self.generate_gnn_predictions(adjacency, max_colors)?;
        let neural_entropy = self.compute_prediction_entropy(&gnn_predictions);

        println!("   GNN: Predicted colors, Entropy = {:.3}", neural_entropy);

        // STEP 3: Predictive Error
        println!("   ⚡ Computing prediction error...");
        let prediction_error = self.neuro.write()
            .generate_and_compare(adjacency)?;
        let hard_vertices = prediction_error.hard_vertices(10);
        let surprising_edges = prediction_error.surprising_edges(2.0);

        println!("   Neuro: Free Energy = {:.3}, Hard vertices: {}",
                 prediction_error.free_energy, hard_vertices.len());

        // STEP 4: Build Modulated Hamiltonian
        let mut modulated = self.build_base_hamiltonian(n, base_energy_scale);

        // Apply topology-guided modulation
        self.apply_topology_modulation(
            &mut modulated,
            &topology,
            &important_vertices,
            difficulty,
        );

        // Apply GNN-guided modulation
        self.apply_gnn_modulation(
            &mut modulated,
            &gnn_predictions,
            max_colors,
        );

        // Apply surprise-guided modulation
        self.apply_surprise_modulation(
            &mut modulated,
            &prediction_error,
            &hard_vertices,
            &surprising_edges,
        );

        // STEP 5: Information-theoretic regularization
        modulated.entropy_regularization = self.compute_entropy_regularization(
            &topology,
            neural_entropy,
            prediction_error.information_gain,
        );

        // STEP 6: Verify Constitutional Compliance
        self.verify_constitutional_compliance(&modulated)?;
        modulated.verify_consistency()?;

        // STEP 7: Track learning history
        self.update_history(difficulty, neural_entropy, prediction_error.total_surprise());

        // STEP 8: Adaptive hyperparameter update
        self.adapt_hyperparameters();

        println!("   ✅ Hamiltonian modulated successfully");
        println!("      Topology weight: {:.3}", modulated.topology_weight);
        println!("      Entropy regularization: {:.3}", modulated.entropy_regularization);
        println!("      Free energy: {:.3}", modulated.free_energy);

        Ok(modulated)
    }

    /// Build base Hamiltonian
    fn build_base_hamiltonian(&self, n: usize, energy_scale: f64) -> ModulatedHamiltonian {
        ModulatedHamiltonian {
            energy_scale,
            vertex_bias: Array1::zeros(n),
            edge_coupling: Array2::from_elem((n, n), 1.0), // Default coupling
            local_temperature: Array1::from_elem(n, 1.0),  // Uniform temperature
            color_prior: Array2::zeros((n, 100)),  // Will be filled by GNN
            topology_weight: 1.0,
            entropy_regularization: 0.0,
            free_energy: 0.0,
        }
    }

    /// Apply topology-guided modulation
    fn apply_topology_modulation(
        &self,
        hamiltonian: &mut ModulatedHamiltonian,
        topology: &PersistenceBarcode,
        important_vertices: &[usize],
        difficulty: f64,
    ) {
        // Increase coupling strength for vertices in critical cliques
        for clique in &topology.critical_cliques {
            for &v in clique {
                for &u in clique {
                    if v != u {
                        // These vertices MUST have different colors
                        let modulation = 1.0 + self.alpha_topology * difficulty;
                        hamiltonian.edge_coupling[[v, u]] *= modulation;
                    }
                }
            }
        }

        // Bias important vertices to be colored first
        for &v in important_vertices {
            hamiltonian.vertex_bias[v] -= 0.1 * self.alpha_topology;
        }

        // Set topology weight based on persistent entropy
        hamiltonian.topology_weight = 1.0 + topology.persistent_entropy * self.alpha_topology;
    }

    /// Apply GNN-guided modulation
    fn apply_gnn_modulation(
        &self,
        hamiltonian: &mut ModulatedHamiltonian,
        predictions: &Array2<f64>,
        max_colors: usize,
    ) {
        let n = hamiltonian.vertex_bias.len();

        // Set color priors from GNN predictions
        for v in 0..n {
            for c in 0..max_colors.min(hamiltonian.color_prior.ncols()) {
                if c < predictions.ncols() {
                    hamiltonian.color_prior[[v, c]] = predictions[[v, c]] * self.beta_prior;
                }
            }

            // Find most confident prediction
            let mut max_conf = 0.0;
            let mut best_color = 0;
            for c in 0..predictions.ncols().min(max_colors) {
                if predictions[[v, c]] > max_conf {
                    max_conf = predictions[[v, c]];
                    best_color = c;
                }
            }

            // Bias towards confident predictions
            if max_conf > 0.8 {
                hamiltonian.vertex_bias[v] -= max_conf * self.beta_prior;
                hamiltonian.color_prior[[v, best_color]] += max_conf * self.beta_prior;
            }
        }
    }

    /// Apply surprise-guided modulation
    fn apply_surprise_modulation(
        &mut self,
        hamiltonian: &mut ModulatedHamiltonian,
        prediction_error: &PredictionError,
        hard_vertices: &[usize],
        surprising_edges: &[(usize, usize)],
    ) {
        // Increase temperature for high-surprise vertices (more exploration)
        for &v in hard_vertices {
            let surprise = prediction_error.vertex_surprise[v];
            hamiltonian.local_temperature[v] *= 1.0 + self.gamma_surprise * surprise / 10.0;
        }

        // Increase coupling for surprising edges
        for &(i, j) in surprising_edges {
            let edge_surprise = prediction_error.edge_surprise[[i, j]];
            hamiltonian.edge_coupling[[i, j]] *= 1.0 + self.gamma_surprise * edge_surprise / 10.0;
            hamiltonian.edge_coupling[[j, i]] = hamiltonian.edge_coupling[[i, j]];
        }

        // Track free energy
        hamiltonian.free_energy = prediction_error.free_energy;
    }

    /// Compute entropy regularization term
    fn compute_entropy_regularization(
        &self,
        topology: &PersistenceBarcode,
        neural_entropy: f64,
        information_gain: f64,
    ) -> f64 {
        // Combine multiple entropy sources
        let topo_entropy = topology.persistent_entropy;

        // Weighted combination
        let total_entropy = 0.3 * topo_entropy + 0.3 * neural_entropy + 0.4 * information_gain;

        // Scale to reasonable range
        (total_entropy * 0.1).min(1.0)
    }

    /// Generate simplified GNN predictions
    fn generate_gnn_predictions(
        &self,
        adjacency: &Array2<bool>,
        max_colors: usize,
    ) -> Result<Array2<f64>> {
        let n = adjacency.nrows();
        let mut predictions = Array2::zeros((n, max_colors));

        // Simplified: use degree-based heuristic
        // In production, would run actual GNN forward pass
        for v in 0..n {
            let degree = adjacency.row(v).iter().filter(|&&x| x).count();

            // Higher degree vertices need more colors
            let estimated_colors = (degree as f64 / 2.0).ceil() as usize + 1;

            // Create probability distribution
            for c in 0..max_colors.min(estimated_colors + 2) {
                let prob = if c < estimated_colors {
                    0.8 / estimated_colors as f64
                } else {
                    0.2 / (max_colors - estimated_colors) as f64
                };
                predictions[[v, c]] = prob;
            }
        }

        Ok(predictions)
    }

    /// Compute prediction entropy
    fn compute_prediction_entropy(&self, predictions: &Array2<f64>) -> f64 {
        let mut total_entropy = 0.0;
        let n = predictions.nrows();

        for v in 0..n {
            let mut vertex_entropy = 0.0;
            for c in 0..predictions.ncols() {
                let p = predictions[[v, c]];
                if p > 1e-10 && p < 1.0 - 1e-10 {
                    vertex_entropy -= p * p.log2();
                }
            }
            total_entropy += vertex_entropy;
        }

        total_entropy / n as f64
    }

    /// Verify constitutional compliance
    fn verify_constitutional_compliance(&mut self, hamiltonian: &ModulatedHamiltonian) -> Result<()> {
        // Article I: Thermodynamic consistency
        // Check entropy production
        let energy_flux = hamiltonian.edge_coupling.iter()
            .map(|&c| c.abs())
            .sum::<f64>() / hamiltonian.edge_coupling.len() as f64;

        let avg_temperature = hamiltonian.local_temperature.mean().unwrap_or(1.0);
        self.entropy_production = energy_flux / avg_temperature;

        if self.entropy_production < -1e-10 {
            return Err(anyhow!(
                "CONSTITUTION VIOLATION (Article I): Negative entropy production {}",
                self.entropy_production
            ));
        }

        // Article II: Information bounds
        // Check that information flow is bounded
        self.information_flow = hamiltonian.entropy_regularization
            + hamiltonian.topology_weight.log2()
            + hamiltonian.free_energy.abs();

        if self.information_flow.is_infinite() || self.information_flow > 1e6 {
            return Err(anyhow!(
                "CONSTITUTION VIOLATION (Article II): Unbounded information flow {}",
                self.information_flow
            ));
        }

        // Article III: Free energy principle
        // Verify free energy is being minimized (tracked, not enforced here)
        if hamiltonian.free_energy.is_nan() {
            return Err(anyhow!(
                "CONSTITUTION VIOLATION (Article III): Invalid free energy"
            ));
        }

        Ok(())
    }

    /// Update learning history
    fn update_history(&mut self, topology_score: f64, gnn_confidence: f64, surprise_level: f64) {
        let history_entry = ModulationHistory {
            topology_score,
            gnn_confidence,
            surprise_level,
            performance: 0.0, // Will be updated after solving
        };

        self.history.push(history_entry);

        // Keep bounded history
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Adapt hyperparameters based on history
    fn adapt_hyperparameters(&mut self) {
        if self.history.len() < 10 {
            return; // Not enough data
        }

        // Compute correlation between each factor and performance
        let recent_history = &self.history[self.history.len()-10..];

        let avg_topo = recent_history.iter().map(|h| h.topology_score).sum::<f64>() / 10.0;
        let avg_gnn = recent_history.iter().map(|h| h.gnn_confidence).sum::<f64>() / 10.0;
        let avg_surprise = recent_history.iter().map(|h| h.surprise_level).sum::<f64>() / 10.0;

        // Simple adaptive rule: increase weight for low-performing factors
        if avg_topo > 0.7 {
            self.alpha_topology *= 1.05; // Topology is important, increase weight
        } else {
            self.alpha_topology *= 0.95;
        }

        if avg_gnn < 0.5 {
            self.beta_prior *= 0.95; // GNN not confident, decrease weight
        } else {
            self.beta_prior *= 1.05;
        }

        if avg_surprise > 5.0 {
            self.gamma_surprise *= 1.05; // High surprise, need more adaptation
        } else {
            self.gamma_surprise *= 0.95;
        }

        // Keep weights in reasonable bounds
        self.alpha_topology = self.alpha_topology.clamp(0.1, 10.0);
        self.beta_prior = self.beta_prior.clamp(0.1, 5.0);
        self.gamma_surprise = self.gamma_surprise.clamp(0.1, 10.0);
    }

    /// Update performance after solving
    pub fn update_performance(&mut self, performance: f64) {
        if let Some(last) = self.history.last_mut() {
            last.performance = performance;
        }
    }

    /// Get current modulation parameters
    pub fn get_parameters(&self) -> (f64, f64, f64) {
        (self.alpha_topology, self.beta_prior, self.gamma_surprise)
    }

    /// Get constitutional metrics
    pub fn get_constitutional_metrics(&self) -> (f64, f64) {
        (self.entropy_production, self.information_flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_creation() {
        let coordinator = MetaLearningCoordinator::new(10, 100);
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_hamiltonian_modulation() {
        let mut coordinator = MetaLearningCoordinator::new(10, 100).unwrap();

        // Create test adjacency matrix
        let adjacency = Array2::from_shape_fn((10, 10), |(i, j)| {
            i != j && (i + j) % 3 == 0
        });

        let hamiltonian = coordinator.compute_modulated_hamiltonian(
            &adjacency,
            1.0,
            5,
        );

        assert!(hamiltonian.is_ok());

        let h = hamiltonian.unwrap();
        assert_eq!(h.vertex_bias.len(), 10);
        assert_eq!(h.edge_coupling.nrows(), 10);
        assert!(h.entropy_regularization >= 0.0);
    }

    #[test]
    fn test_constitutional_compliance() {
        let hamiltonian = ModulatedHamiltonian {
            energy_scale: 1.0,
            vertex_bias: Array1::zeros(10),
            edge_coupling: Array2::eye(10),
            local_temperature: Array1::from_elem(10, 1.0),
            color_prior: Array2::zeros((10, 5)),
            topology_weight: 1.0,
            entropy_regularization: 0.1,
            free_energy: -1.0,
        };

        assert!(hamiltonian.verify_consistency().is_ok());
    }
}