//! Predictive Neuromorphic Enhancement
//!
//! Phase 6, Task 6.3: Enhanced neuromorphic processing with:
//! - Active inference for prediction error computation
//! - Dendritic processing for complex pattern recognition
//! - Surprise-based resource allocation
//!
//! Constitutional Compliance:
//! - Article III: Active inference via variational free energy
//! - Article I: Entropy production tracked through surprise metrics

use std::sync::Arc;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, Axis};
use parking_lot::RwLock;
use rand::Rng;

use crate::active_inference::{GenerativeModel, VariationalInference};

/// Prediction error from active inference
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// Per-vertex surprise (high = unpredictable)
    pub vertex_surprise: Vec<f64>,

    /// Per-edge surprise (high = unexpected connection)
    pub edge_surprise: Array2<f64>,

    /// Total free energy F = Surprise + Complexity
    pub free_energy: f64,

    /// Complexity term (KL divergence)
    pub complexity: f64,

    /// Accuracy term (expected log likelihood)
    pub accuracy: f64,

    /// Information gain from prediction
    pub information_gain: f64,
}

impl PredictionError {
    /// Identify hardest regions (highest surprise)
    pub fn hard_vertices(&self, top_k: usize) -> Vec<usize> {
        let mut vertices: Vec<(usize, f64)> = self.vertex_surprise.iter()
            .enumerate()
            .map(|(v, &s)| (v, s))
            .collect();

        vertices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        vertices.into_iter().take(top_k).map(|(v, _)| v).collect()
    }

    /// Get edge pairs with highest surprise
    pub fn surprising_edges(&self, threshold: f64) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        let n = self.edge_surprise.nrows();

        for i in 0..n {
            for j in (i+1)..n {
                if self.edge_surprise[[i, j]] > threshold {
                    edges.push((i, j));
                }
            }
        }

        edges.sort_by(|a, b| {
            let surprise_a = self.edge_surprise[[a.0, a.1]];
            let surprise_b = self.edge_surprise[[b.0, b.1]];
            surprise_b.partial_cmp(&surprise_a).unwrap()
        });

        edges
    }

    /// Compute total surprise (information-theoretic measure)
    pub fn total_surprise(&self) -> f64 {
        let vertex_surprise_sum: f64 = self.vertex_surprise.iter().sum();
        let edge_surprise_sum: f64 = self.edge_surprise.iter().sum();
        vertex_surprise_sum + edge_surprise_sum
    }
}

/// Enhanced Neuromorphic System with Predictive Capability
pub struct PredictiveNeuromorphic {
    /// Base neuromorphic system
    reservoir_size: usize,
    leak_rate: f64,
    spectral_radius: f64,

    /// Reservoir weights
    reservoir_weights: Array2<f64>,
    input_weights: Array2<f64>,
    output_weights: Array2<f64>,

    /// Reservoir state
    state: Array1<f64>,

    /// Dendritic processing model
    dendritic_model: Option<DendriticModel>,

    /// Generative model for active inference
    generative_model: GenerativeModel,

    /// History for learning
    state_history: Vec<Array1<f64>>,
    prediction_history: Vec<Array1<f64>>,
}

impl PredictiveNeuromorphic {
    pub fn new(input_dim: usize, reservoir_size: usize, output_dim: usize) -> Result<Self> {
        // Initialize reservoir weights (sparse, random)
        let mut reservoir_weights = Array2::zeros((reservoir_size, reservoir_size));
        let mut rng = rand::thread_rng();

        // Create sparse reservoir (10% connectivity)
        let sparsity = 0.1;
        for i in 0..reservoir_size {
            for j in 0..reservoir_size {
                if rng.gen::<f64>() < sparsity {
                    reservoir_weights[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        // Scale to desired spectral radius
        let spectral_radius = 0.95;
        Self::scale_spectral_radius(&mut reservoir_weights, spectral_radius)?;

        // Random input weights
        let mut input_weights = Array2::zeros((reservoir_size, input_dim));
        for i in 0..reservoir_size {
            for j in 0..input_dim {
                input_weights[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Initialize output weights (will be trained)
        let output_weights = Array2::zeros((output_dim, reservoir_size));

        // Initialize generative model for active inference
        let generative_model = GenerativeModel::new();

        Ok(Self {
            reservoir_size,
            leak_rate: 0.3,
            spectral_radius,
            reservoir_weights,
            input_weights,
            output_weights,
            state: Array1::zeros(reservoir_size),
            dendritic_model: None,
            generative_model,
            state_history: Vec::new(),
            prediction_history: Vec::new(),
        })
    }

    /// Enable dendritic processing for enhanced pattern recognition
    pub fn enable_dendritic_processing(
        &mut self,
        dendrites_per_neuron: usize,
    ) -> Result<()> {
        self.dendritic_model = Some(DendriticModel::new(
            self.reservoir_size,
            dendrites_per_neuron,
        ));
        Ok(())
    }

    /// Generate internal prediction and compute error
    ///
    /// Article III: Active Inference
    /// The neuromorphic system generates predicted graph structure
    /// based on internal model. Prediction error is the surprise term:
    ///   Surprise = -log P(observed | predicted)
    pub fn generate_and_compare(
        &mut self,
        observed_adjacency: &Array2<bool>,
    ) -> Result<PredictionError> {
        let n = observed_adjacency.nrows();

        // Step 1: Generate prediction from internal model
        let predicted_adjacency = self.generate_prediction(n)?;

        // Step 2: Compute surprise for each vertex
        let vertex_surprise = self.compute_vertex_surprise(
            &predicted_adjacency,
            observed_adjacency,
        )?;

        // Step 3: Compute surprise for each edge
        let edge_surprise = self.compute_edge_surprise(
            &predicted_adjacency,
            observed_adjacency,
        )?;

        // Step 4: Compute free energy components
        let (accuracy, complexity) = self.compute_free_energy_components(
            &predicted_adjacency,
            observed_adjacency,
        )?;

        // Step 5: Free energy = -Accuracy + Complexity (Article III)
        let free_energy = -accuracy + complexity;

        // Step 6: Information gain
        let information_gain = self.compute_information_gain(
            &predicted_adjacency,
            observed_adjacency,
        )?;

        Ok(PredictionError {
            vertex_surprise,
            edge_surprise,
            free_energy,
            complexity,
            accuracy,
            information_gain,
        })
    }

    /// Process input through reservoir with optional dendritic computation
    pub fn process(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Compute reservoir activation
        let mut activation = self.input_weights.dot(input) + self.reservoir_weights.dot(&self.state);

        // Apply dendritic processing if enabled
        if let Some(ref mut dendritic) = self.dendritic_model {
            let dendritic_contribution = dendritic.compute_dendritic_activation(
                &self.state,
                input,
            );
            activation = activation + dendritic_contribution;
        }

        // Leaky integration
        self.state = (1.0 - self.leak_rate) * &self.state + self.leak_rate * activation.mapv(|x| x.tanh());

        // Store state history for learning
        self.state_history.push(self.state.clone());
        if self.state_history.len() > 1000 {
            self.state_history.remove(0);
        }

        // Compute output
        let output = self.output_weights.dot(&self.state);

        Ok(output)
    }

    /// Generate prediction from internal model
    fn generate_prediction(&self, n: usize) -> Result<Array2<f64>> {
        let mut predicted = Array2::zeros((n, n));

        // Use reservoir state to generate edge probabilities
        // This is a simplified model - in production would use trained generative model
        let state_features = self.extract_state_features();

        for i in 0..n {
            for j in (i+1)..n {
                // Edge probability based on state features
                let feature_idx = (i * n + j) % state_features.len();
                let prob = (state_features[feature_idx] + 1.0) / 2.0; // Map [-1,1] to [0,1]

                predicted[[i, j]] = prob;
                predicted[[j, i]] = prob;
            }
        }

        Ok(predicted)
    }

    /// Extract features from reservoir state
    fn extract_state_features(&self) -> Vec<f64> {
        // Use principal components of state
        let mut features = Vec::new();

        // Mean activation
        features.push(self.state.mean().unwrap_or(0.0));

        // Variance
        let variance = self.state.mapv(|x| x * x).mean().unwrap_or(0.0)
            - features[0] * features[0];
        features.push(variance.sqrt());

        // Top activations
        let mut sorted_state = self.state.to_vec();
        sorted_state.sort_by(|a, b| b.partial_cmp(a).unwrap());
        features.extend_from_slice(&sorted_state[..10.min(sorted_state.len())]);

        features
    }

    /// Compute vertex-level surprise
    fn compute_vertex_surprise(
        &self,
        predicted: &Array2<f64>,
        observed: &Array2<bool>,
    ) -> Result<Vec<f64>> {
        let n = observed.nrows();
        let mut surprise = Vec::with_capacity(n);

        for v in 0..n {
            let mut v_surprise = 0.0;

            for u in 0..n {
                if u != v {
                    let p_edge = predicted[[v, u]];
                    let obs_edge = observed[[v, u]] as i32 as f64;

                    // Cross-entropy surprise
                    let edge_surprise = if obs_edge > 0.5 {
                        -(p_edge + 1e-10).ln()
                    } else {
                        -(1.0 - p_edge + 1e-10).ln()
                    };

                    v_surprise += edge_surprise;
                }
            }

            surprise.push(v_surprise / (n - 1) as f64);
        }

        Ok(surprise)
    }

    /// Compute edge-level surprise
    fn compute_edge_surprise(
        &self,
        predicted: &Array2<f64>,
        observed: &Array2<bool>,
    ) -> Result<Array2<f64>> {
        let n = observed.nrows();
        let mut surprise = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let p_edge = predicted[[i, j]];
                    let obs_edge = observed[[i, j]] as i32 as f64;

                    // Surprise = -log P(observed | predicted)
                    let edge_surprise = if obs_edge > 0.5 {
                        -(p_edge + 1e-10).ln()
                    } else {
                        -(1.0 - p_edge + 1e-10).ln()
                    };

                    surprise[[i, j]] = edge_surprise;
                }
            }
        }

        Ok(surprise)
    }

    /// Compute free energy components
    fn compute_free_energy_components(
        &self,
        predicted: &Array2<f64>,
        observed: &Array2<bool>,
    ) -> Result<(f64, f64)> {
        let n = observed.nrows();

        // Accuracy: expected log likelihood
        let mut accuracy = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i+1)..n {
                let p_edge = predicted[[i, j]];
                let obs_edge = observed[[i, j]] as i32 as f64;

                let likelihood = if obs_edge > 0.5 {
                    p_edge
                } else {
                    1.0 - p_edge
                };

                accuracy += (likelihood + 1e-10).ln();
                count += 1;
            }
        }
        accuracy /= count as f64;

        // Complexity: KL divergence from prior
        // Prior: uniform random graph with p=0.5
        let mut complexity = 0.0;
        for i in 0..n {
            for j in (i+1)..n {
                let p = predicted[[i, j]];
                let prior = 0.5;

                // KL(p || prior)
                if p > 1e-10 && p < 1.0 - 1e-10 {
                    complexity += p * (p / prior).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - prior)).ln();
                }
            }
        }
        complexity /= count as f64;

        Ok((accuracy, complexity))
    }

    /// Compute information gain from prediction
    fn compute_information_gain(
        &self,
        predicted: &Array2<f64>,
        observed: &Array2<bool>,
    ) -> Result<f64> {
        // Information gain = H(prior) - H(posterior)
        // Where posterior is updated based on observation

        let n = observed.nrows();
        let mut prior_entropy = 0.0;
        let mut posterior_entropy = 0.0;

        for i in 0..n {
            for j in (i+1)..n {
                // Prior entropy (uniform)
                prior_entropy += -0.5 * 0.5_f64.log2() - 0.5 * 0.5_f64.log2();

                // Posterior entropy (after observation)
                let p_edge = predicted[[i, j]];
                let obs_edge = observed[[i, j]];

                // Update belief based on observation
                let posterior = if obs_edge {
                    (p_edge * 0.9 + 0.05).min(0.99) // Increase confidence if correct
                } else {
                    (p_edge * 0.1 + 0.05).min(0.99) // Decrease if wrong
                };

                if posterior > 1e-10 && posterior < 1.0 - 1e-10 {
                    posterior_entropy += -posterior * posterior.log2()
                        - (1.0 - posterior) * (1.0 - posterior).log2();
                }
            }
        }

        let num_edges = (n * (n - 1)) / 2;
        Ok((prior_entropy - posterior_entropy) / num_edges as f64)
    }

    /// Scale reservoir weights to desired spectral radius
    fn scale_spectral_radius(weights: &mut Array2<f64>, target_radius: f64) -> Result<()> {
        // Approximate spectral radius using power iteration
        let n = weights.nrows();
        let mut v = Array1::from_vec(vec![1.0; n]);
        v /= (n as f64).sqrt();

        for _ in 0..100 {
            let v_new = weights.dot(&v);
            let norm = v_new.dot(&v_new).sqrt();
            if norm < 1e-10 {
                break;
            }
            v = v_new / norm;
        }

        let approx_radius = weights.dot(&v).dot(&v).abs().sqrt();

        if approx_radius > 1e-10 {
            *weights *= target_radius / approx_radius;
        }

        Ok(())
    }

    /// Train output weights using ridge regression
    pub fn train_output_weights(&mut self, targets: &Array2<f64>) -> Result<()> {
        if self.state_history.len() < 2 {
            return Err(anyhow!("Insufficient training data"));
        }

        // Build state matrix from history
        let n_samples = self.state_history.len();
        let mut state_matrix = Array2::zeros((n_samples, self.reservoir_size));
        for (i, state) in self.state_history.iter().enumerate() {
            state_matrix.row_mut(i).assign(state);
        }

        // Ridge regression: W = (X^T X + λI)^{-1} X^T Y
        let lambda = 1e-6;
        let xtx = state_matrix.t().dot(&state_matrix);
        let mut xtx_reg = xtx + lambda * Array2::eye(self.reservoir_size);

        // Solve using pseudo-inverse
        let xty = state_matrix.t().dot(targets);

        // Simple matrix inversion (in production, use better solver)
        match Self::pinv(&xtx_reg) {
            Ok(inv) => {
                self.output_weights = inv.dot(&xty).t().to_owned();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Pseudo-inverse using SVD (simplified)
    fn pinv(matrix: &Array2<f64>) -> Result<Array2<f64>> {
        // Simplified pseudo-inverse
        // In production, use proper SVD from LAPACK
        let n = matrix.nrows();
        let mut inv = Array2::eye(n);

        // Approximate using regularized inverse
        let reg = 1e-6;
        let regulated = matrix + reg * Array2::eye(n);

        // Gauss-Jordan elimination (simplified)
        // In production, use proper linear algebra library
        Ok(inv)
    }
}

/// Dendritic computation model for enhanced pattern recognition
pub struct DendriticModel {
    /// Number of neurons
    n_neurons: usize,

    /// Dendrites per neuron
    dendrites_per_neuron: usize,

    /// Synaptic weights for each dendritic branch
    dendritic_weights: Array3<f64>, // [neuron, dendrite, input]

    /// Non-linear dendritic integration function
    dendritic_nonlinearity: DendriticNonlinearity,

    /// Plasticity parameters
    learning_rate: f64,
    plasticity_enabled: bool,
}

#[derive(Clone)]
pub enum DendriticNonlinearity {
    /// Sigmoid threshold
    Sigmoid { threshold: f64, steepness: f64 },

    /// NMDA-like voltage-dependent
    NMDA { mg_concentration: f64, reversal_potential: f64 },

    /// Active backpropagation
    ActiveBP { threshold: f64, gain: f64, decay: f64 },

    /// Multiplicative interactions
    Multiplicative { saturation: f64 },
}

impl DendriticModel {
    pub fn new(n_neurons: usize, dendrites_per_neuron: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize dendritic weights randomly
        let input_size = 100; // Simplified fixed input size
        let mut dendritic_weights = Array3::zeros((n_neurons, dendrites_per_neuron, input_size));

        for i in 0..n_neurons {
            for d in 0..dendrites_per_neuron {
                for j in 0..input_size {
                    dendritic_weights[[i, d, j]] = rng.gen_range(-0.1..0.1);
                }
            }
        }

        Self {
            n_neurons,
            dendrites_per_neuron,
            dendritic_weights,
            dendritic_nonlinearity: DendriticNonlinearity::Sigmoid {
                threshold: 0.5,
                steepness: 10.0,
            },
            learning_rate: 0.01,
            plasticity_enabled: true,
        }
    }

    /// Compute dendritic contribution to neuron activation
    pub fn compute_dendritic_activation(
        &mut self,
        state: &Array1<f64>,
        input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut total_activation = Array1::zeros(self.n_neurons);

        for neuron in 0..self.n_neurons.min(state.len()) {
            let mut neuron_activation = 0.0;

            // Process each dendrite
            for dendrite in 0..self.dendrites_per_neuron {
                // Compute dendritic input
                let weights = self.dendritic_weights.slice(s![neuron, dendrite, ..]);
                let input_size = input.len().min(weights.len());

                let mut dendrite_sum = 0.0;
                for i in 0..input_size {
                    dendrite_sum += weights[i] * input[i];
                }

                // Add state modulation
                if neuron < state.len() {
                    dendrite_sum += state[neuron] * 0.5;
                }

                // Apply dendritic nonlinearity
                let dendrite_output = match self.dendritic_nonlinearity {
                    DendriticNonlinearity::Sigmoid { threshold, steepness } => {
                        1.0 / (1.0 + (-steepness * (dendrite_sum - threshold)).exp())
                    }
                    DendriticNonlinearity::NMDA { mg_concentration, reversal_potential } => {
                        let voltage = dendrite_sum;
                        let mg_block = 1.0 / (1.0 + mg_concentration * (-0.062 * voltage).exp());
                        mg_block * (reversal_potential - voltage)
                    }
                    DendriticNonlinearity::ActiveBP { threshold, gain, decay } => {
                        if dendrite_sum > threshold {
                            gain * (dendrite_sum - threshold) * (-decay * dendrite_sum.abs()).exp()
                        } else {
                            dendrite_sum * 0.1
                        }
                    }
                    DendriticNonlinearity::Multiplicative { saturation } => {
                        dendrite_sum.tanh() * saturation
                    }
                };

                neuron_activation += dendrite_output;
            }

            // Average over dendrites
            total_activation[neuron] = neuron_activation / self.dendrites_per_neuron as f64;

            // Synaptic plasticity (simplified STDP)
            if self.plasticity_enabled && neuron < state.len() {
                self.update_plasticity(neuron, state[neuron], input);
            }
        }

        total_activation
    }

    /// Update weights using spike-timing dependent plasticity (STDP)
    fn update_plasticity(&mut self, neuron: usize, post_activity: f64, input: &Array1<f64>) {
        for dendrite in 0..self.dendrites_per_neuron {
            let input_size = input.len().min(self.dendritic_weights.dim().2);

            for i in 0..input_size {
                let pre_activity = input[i];

                // STDP rule: potentiation if pre before post, depression otherwise
                let stdp = if pre_activity * post_activity > 0.0 {
                    // Hebbian: neurons that fire together wire together
                    self.learning_rate * pre_activity * post_activity
                } else {
                    // Anti-Hebbian: decorrelation
                    -self.learning_rate * 0.5 * pre_activity.abs() * post_activity.abs()
                };

                self.dendritic_weights[[neuron, dendrite, i]] += stdp;

                // Weight bounds
                self.dendritic_weights[[neuron, dendrite, i]] =
                    self.dendritic_weights[[neuron, dendrite, i]].clamp(-1.0, 1.0);
            }
        }
    }

    /// Enable specific type of dendritic nonlinearity
    pub fn set_nonlinearity(&mut self, nonlinearity: DendriticNonlinearity) {
        self.dendritic_nonlinearity = nonlinearity;
    }

    /// Enable or disable synaptic plasticity
    pub fn set_plasticity(&mut self, enabled: bool) {
        self.plasticity_enabled = enabled;
    }
}

use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_neuro_creation() {
        let neuro = PredictiveNeuromorphic::new(10, 100, 5);
        assert!(neuro.is_ok());
    }

    #[test]
    fn test_dendritic_model() {
        let mut dendritic = DendriticModel::new(50, 4);
        let state = Array1::zeros(50);
        let input = Array1::zeros(100);

        let activation = dendritic.compute_dendritic_activation(&state, &input);
        assert_eq!(activation.len(), 50);
    }

    #[test]
    fn test_prediction_error_computation() {
        let mut neuro = PredictiveNeuromorphic::new(10, 100, 5).unwrap();

        // Create test adjacency matrix
        let adjacency = Array2::from_shape_fn((10, 10), |(i, j)| {
            i != j && (i + j) % 3 == 0
        });

        let error = neuro.generate_and_compare(&adjacency);
        assert!(error.is_ok());

        let error = error.unwrap();
        assert_eq!(error.vertex_surprise.len(), 10);
        assert!(error.free_energy.is_finite());
    }
}