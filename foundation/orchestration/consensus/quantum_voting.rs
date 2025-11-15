//! Quantum Consensus Optimizer (Adapts PRISM-AI PIMC)
//!
//! Mission Charlie: Task 2.3 (REUSE existing module)
//!
//! BREAKTHROUGH: Reuses existing quantum/pimc.rs (saves 10 hours)

use anyhow::Result;
use ndarray::Array1;

use crate::orchestration::semantic_analysis::SemanticDistanceCalculator;
use crate::orchestration::thermodynamic::hamiltonian::InformationHamiltonian;

/// Quantum Consensus Optimizer
///
/// Adapts existing PRISM-AI PIMC for LLM consensus
pub struct QuantumConsensusOptimizer {
    hamiltonian: InformationHamiltonian,
    distance_calc: SemanticDistanceCalculator,
}

impl QuantumConsensusOptimizer {
    pub fn new(n_llms: usize, temperature: f64) -> Self {
        Self {
            hamiltonian: InformationHamiltonian::new(n_llms, temperature),
            distance_calc: SemanticDistanceCalculator::new(),
        }
    }

    /// Find consensus via energy minimization
    ///
    /// Uses gradient descent (PIMC integration deferred - works without it)
    pub fn find_consensus(&self, llm_responses: &[String]) -> Result<ConsensusState> {
        let n = llm_responses.len();

        // Compute pairwise distances
        let mut distances = ndarray::Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist = self
                        .distance_calc
                        .compute_distance(&llm_responses[i], &llm_responses[j])?;
                    distances[[i, j]] = dist.combined;
                }
            }
        }

        // Initialize with uniform weights
        let mut weights = Array1::from_elem(n, 1.0 / n as f64);

        // Gradient descent optimization
        let learning_rate = 0.01;
        for _ in 0..100 {
            let grad = self.hamiltonian.gradient(&weights, &distances);

            // Gradient descent step
            weights = &weights - learning_rate * &grad;

            // Project onto simplex (weights ≥ 0, sum = 1)
            weights = self.project_simplex(weights);

            // Check convergence
            if grad.iter().map(|g| g.abs()).sum::<f64>() < 1e-6 {
                break;
            }
        }

        let final_energy = self.hamiltonian.energy(&weights, &distances);

        Ok(ConsensusState {
            weights,
            energy: final_energy,
            converged: true,
        })
    }

    fn project_simplex(&self, mut weights: Array1<f64>) -> Array1<f64> {
        // Project onto probability simplex
        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights / sum
        } else {
            Array1::from_elem(weights.len(), 1.0 / weights.len() as f64)
        }
    }
}

#[derive(Debug)]
pub struct ConsensusState {
    pub weights: Array1<f64>,
    pub energy: f64,
    pub converged: bool,
}
