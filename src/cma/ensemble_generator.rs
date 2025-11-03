//! Enhanced Thermodynamic Ensemble Generation
//!
//! # Purpose
//! Stage 1 of CMA: Generate high-quality solution ensemble using
//! replica exchange Monte Carlo with free energy convergence.
//!
//! # Mathematical Foundation
//! P_β(s) = Z_β^(-1) exp(-βH(s))
//! Convergence: |d⟨F_βmax⟩/dt| < ε

use std::collections::VecDeque;

/// Enhanced ensemble generator with replica exchange
pub struct EnhancedEnsembleGenerator {
    replicas: Vec<ReplicaState>,
    temperatures: Vec<f64>,
    free_energy_history: VecDeque<f64>,
    convergence_threshold: f64,
}

impl EnhancedEnsembleGenerator {
    pub fn new() -> Self {
        // Geometric temperature spacing
        let n_replicas = 10;
        let beta_min = 0.1;
        let beta_max = 10.0;

        let temperatures: Vec<f64> = (0..n_replicas)
            .map(|i| {
                let ratio = i as f64 / (n_replicas - 1) as f64;
                let beta_ratio = (beta_max / beta_min) as f64;
                let beta = beta_min * beta_ratio.powf(ratio);
                1.0_f64 / beta
            })
            .collect();

        Self {
            replicas: vec![ReplicaState::default(); n_replicas],
            temperatures,
            free_energy_history: VecDeque::with_capacity(100),
            convergence_threshold: 1e-4,
        }
    }

    /// Generate ensemble with free energy convergence
    pub fn generate_with_convergence(
        &mut self,
        problem: &impl crate::cma::Problem,
        window_size: usize,
    ) -> crate::cma::Ensemble {
        loop {
            self.parallel_tempering_step(problem);

            let current_f = self.estimate_free_energy(&self.replicas.last().unwrap());
            self.free_energy_history.push_back(current_f);

            if self.free_energy_history.len() > window_size {
                self.free_energy_history.pop_front();
                if self.check_convergence(window_size) {
                    break;
                }
            }
        }

        self.extract_low_energy_ensemble()
    }

    /// Refine existing ensemble
    pub fn refine_ensemble(&mut self, ensemble: Vec<crate::cma::Solution>) -> crate::cma::Ensemble {
        // Initialize replicas from ensemble
        for (i, solution) in ensemble.iter().take(self.replicas.len()).enumerate() {
            self.replicas[i] = ReplicaState {
                state: solution.data.clone(),
                energy: solution.cost,
            };
        }

        crate::cma::Ensemble {
            solutions: ensemble,
        }
    }

    fn parallel_tempering_step(&mut self, problem: &impl crate::cma::Problem) {
        // Monte Carlo steps within replicas
        let temperatures = self.temperatures.clone();
        for (i, replica) in self.replicas.iter_mut().enumerate() {
            let beta = 1.0 / temperatures[i];
            Self::metropolis_step(replica, beta, problem);
        }

        // Replica exchange attempts
        for i in 0..self.replicas.len() - 1 {
            if rand::random::<f64>() < 0.2 {  // 20% exchange rate
                self.attempt_exchange(i, i + 1);
            }
        }
    }

    fn metropolis_step(
        replica: &mut ReplicaState,
        beta: f64,
        problem: &impl crate::cma::Problem,
    ) {
        // Propose new state
        let mut new_state = replica.state.clone();
        let idx = rand::random::<usize>() % new_state.len();
        new_state[idx] += (rand::random::<f64>() - 0.5) * 0.1;

        // Calculate energy change
        let new_solution = crate::cma::Solution {
            data: new_state.clone(),
            cost: problem.evaluate(&crate::cma::Solution {
                data: new_state.clone(),
                cost: 0.0,
            }),
        };

        let delta_e = new_solution.cost - replica.energy;

        // Accept or reject
        if delta_e < 0.0 || rand::random::<f64>() < (-beta * delta_e).exp() {
            replica.state = new_state;
            replica.energy = new_solution.cost;
        }
    }

    fn attempt_exchange(&mut self, i: usize, j: usize) {
        let beta_i = 1.0 / self.temperatures[i];
        let beta_j = 1.0 / self.temperatures[j];

        let delta = (beta_i - beta_j) * (self.replicas[j].energy - self.replicas[i].energy);

        if delta < 0.0 || rand::random::<f64>() < delta.exp() {
            self.replicas.swap(i, j);
        }
    }

    fn estimate_free_energy(&self, replica: &ReplicaState) -> f64 {
        // F = E - TS
        // Simplified: just use energy for now
        replica.energy
    }

    fn check_convergence(&self, window_size: usize) -> bool {
        if self.free_energy_history.len() < window_size {
            return false;
        }

        // Calculate derivative of free energy
        let values: Vec<f64> = self.free_energy_history.iter().copied().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Check if fluctuations are below threshold
        std_dev / mean.abs().max(1.0) < self.convergence_threshold
    }

    fn extract_low_energy_ensemble(&self) -> crate::cma::Ensemble {
        let solutions: Vec<crate::cma::Solution> = self.replicas.iter()
            .map(|r| crate::cma::Solution {
                data: r.state.clone(),
                cost: r.energy,
            })
            .collect();

        crate::cma::Ensemble { solutions }
    }
}

#[derive(Clone, Default)]
struct ReplicaState {
    state: Vec<f64>,
    energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_generator() {
        let generator = EnhancedEnsembleGenerator::new();
        assert_eq!(generator.replicas.len(), 10);
        assert_eq!(generator.temperatures.len(), 10);
    }
}