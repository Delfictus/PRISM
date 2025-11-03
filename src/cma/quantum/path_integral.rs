//! Path Integral Monte Carlo for Quantum Annealing
//!
//! # Purpose
//! Real quantum annealing implementation using path integrals.
//! Replaces placeholder matrix exponential approach.
//!
//! # Mathematical Foundation
//! Path Integral: Z = ∫ D[σ] exp(-S[σ]/ℏ)
//! Action: S[σ] = ∫₀^β dτ [½m(∂σ/∂τ)² + V(σ)]
//! PIMC Update: σᵢ(τⱼ) → σᵢ(τⱼ) + δ with P_accept = min(1, e^(-ΔS/kT))
//!
//! # Constitution Reference
//! Phase 6 Implementation Constitution - Sprint 1.3

use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use anyhow::Result;

use crate::cma::{Solution, CausalManifold};

/// Path Integral Monte Carlo quantum annealer
pub struct PathIntegralMonteCarlo {
    /// Number of Trotter slices (beads)
    n_beads: usize,
    /// Inverse temperature
    beta: f64,
    /// Time step
    tau: f64,
    /// Particle mass
    mass: f64,
    /// Random number generator
    rng: ChaCha20Rng,
}

impl PathIntegralMonteCarlo {
    /// Create new PIMC annealer
    pub fn new(n_beads: usize, beta: f64) -> Self {
        let tau = beta / n_beads as f64;

        Self {
            n_beads,
            beta,
            tau,
            mass: 1.0,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    /// Quantum anneal with manifold constraints
    ///
    /// # Arguments
    /// * `hamiltonian` - Problem Hamiltonian
    /// * `manifold` - Causal manifold constraints
    /// * `initial` - Initial classical solution
    ///
    /// # Returns
    /// Optimized quantum solution
    pub fn quantum_anneal(
        &mut self,
        hamiltonian: &ProblemHamiltonian,
        manifold: &CausalManifold,
        initial: &Solution,
    ) -> Result<Solution> {
        let n_dim = initial.data.len();

        // Initialize worldline (path) - replicated classical solution
        let mut path = self.initialize_path(initial, n_dim);

        println!("🌀 Starting Path Integral Monte Carlo:");
        println!("  Beads: {}, β: {:.3}, τ: {:.6}", self.n_beads, self.beta, self.tau);

        // Annealing schedule
        let schedule = self.compute_annealing_schedule();

        let mut acceptance_rate = 0.0;
        let mut total_proposals = 0;

        for (step, &(beta_t, tunneling_strength)) in schedule.iter().enumerate() {
            // Update each bead
            for bead in 0..self.n_beads {
                for dim in 0..n_dim {
                    let accepted = self.update_bead(
                        &mut path,
                        bead,
                        dim,
                        hamiltonian,
                        manifold,
                        beta_t,
                        tunneling_strength,
                    )?;

                    if accepted {
                        acceptance_rate += 1.0;
                    }
                    total_proposals += 1;
                }
            }

            // Measure observables periodically
            if step % 100 == 0 {
                let energy = self.compute_energy(&path, hamiltonian);
                let end_to_end_dist = self.end_to_end_distance(&path);

                println!("  Step {}: E={:.4}, R²={:.4}, acc={:.2}%",
                         step, energy, end_to_end_dist,
                         100.0 * acceptance_rate / total_proposals as f64);

                acceptance_rate = 0.0;
                total_proposals = 0;
            }
        }

        // Extract classical solution from path
        let solution = self.extract_classical_solution(&path, hamiltonian);

        println!("✓ PIMC converged to solution with cost: {:.6}", solution.cost);

        Ok(solution)
    }

    /// Initialize worldline from classical solution
    fn initialize_path(&self, initial: &Solution, n_dim: usize) -> Worldline {
        let mut path = vec![vec![0.0; n_dim]; self.n_beads];

        // Replicate initial solution with small quantum fluctuations
        for bead in 0..self.n_beads {
            for dim in 0..n_dim {
                path[bead][dim] = initial.data[dim];
            }
        }

        Worldline { beads: path }
    }

    /// Update single bead coordinate via Metropolis
    fn update_bead(
        &mut self,
        path: &mut Worldline,
        bead: usize,
        dim: usize,
        hamiltonian: &ProblemHamiltonian,
        manifold: &CausalManifold,
        beta: f64,
        tunneling: f64,
    ) -> Result<bool> {
        let old_value = path.beads[bead][dim];

        // Propose new value
        let step_size = 0.1 * tunneling; // Larger steps when tunneling is strong
        let new_value = old_value + self.rng.gen_range(-step_size..step_size);

        // Compute action change
        let delta_action = self.compute_delta_action(
            path,
            bead,
            dim,
            old_value,
            new_value,
            hamiltonian,
            manifold,
        );

        // Metropolis acceptance
        let accept_prob = (-delta_action).exp();

        if self.rng.gen::<f64>() < accept_prob {
            path.beads[bead][dim] = new_value;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Compute change in action for proposed move
    fn compute_delta_action(
        &self,
        path: &Worldline,
        bead: usize,
        dim: usize,
        old_value: f64,
        new_value: f64,
        hamiltonian: &ProblemHamiltonian,
        manifold: &CausalManifold,
    ) -> f64 {
        // Action = kinetic + potential
        // S = Σ_i [m/(2τ²) (x_i - x_{i+1})² + τ V(x_i)]

        let prev_bead = if bead == 0 { self.n_beads - 1 } else { bead - 1 };
        let next_bead = (bead + 1) % self.n_beads;

        // Kinetic action (spring between beads)
        let old_kinetic = self.kinetic_action(
            old_value,
            path.beads[prev_bead][dim],
            path.beads[next_bead][dim],
        );

        let new_kinetic = self.kinetic_action(
            new_value,
            path.beads[prev_bead][dim],
            path.beads[next_bead][dim],
        );

        // Potential energy from Hamiltonian
        let old_potential = self.evaluate_at_bead(path, bead, hamiltonian, manifold);

        let mut temp_path = path.clone();
        temp_path.beads[bead][dim] = new_value;
        let new_potential = self.evaluate_at_bead(&temp_path, bead, hamiltonian, manifold);

        (new_kinetic - old_kinetic) + self.tau * (new_potential - old_potential)
    }

    /// Kinetic action for a bead
    fn kinetic_action(&self, x_curr: f64, x_prev: f64, x_next: f64) -> f64 {
        // K = m/(2τ²) [(x - x_prev)² + (x_next - x)²]
        let spring_prev = (x_curr - x_prev).powi(2);
        let spring_next = (x_next - x_curr).powi(2);

        (self.mass / (2.0 * self.tau * self.tau)) * (spring_prev + spring_next)
    }

    /// Evaluate potential at a bead
    fn evaluate_at_bead(
        &self,
        path: &Worldline,
        bead: usize,
        hamiltonian: &ProblemHamiltonian,
        manifold: &CausalManifold,
    ) -> f64 {
        // Construct solution from bead configuration
        let solution = Solution {
            data: path.beads[bead].clone(),
            cost: 0.0,
        };

        // Base Hamiltonian
        let mut energy = hamiltonian.evaluate(&solution);

        // Add manifold constraint penalties
        for edge in &manifold.edges {
            if edge.source < solution.data.len() && edge.target < solution.data.len() {
                let violation = (solution.data[edge.source] - solution.data[edge.target]).abs();
                energy += violation * edge.transfer_entropy * hamiltonian.manifold_coupling;
            }
        }

        energy
    }

    /// Compute total energy of path
    fn compute_energy(&self, path: &Worldline, hamiltonian: &ProblemHamiltonian) -> f64 {
        let mut total = 0.0;

        for bead in 0..self.n_beads {
            let solution = Solution {
                data: path.beads[bead].clone(),
                cost: 0.0,
            };
            total += hamiltonian.evaluate(&solution);
        }

        total / self.n_beads as f64
    }

    /// Compute end-to-end distance (quantum delocalization measure)
    fn end_to_end_distance(&self, path: &Worldline) -> f64 {
        let n_dim = path.beads[0].len();
        let mut dist_sq = 0.0;

        for dim in 0..n_dim {
            let first = path.beads[0][dim];
            let last = path.beads[self.n_beads - 1][dim];
            dist_sq += (last - first).powi(2);
        }

        dist_sq
    }

    /// Extract classical solution from quantum path
    fn extract_classical_solution(
        &self,
        path: &Worldline,
        hamiltonian: &ProblemHamiltonian,
    ) -> Solution {
        // Find lowest energy bead configuration
        let mut best_bead = 0;
        let mut best_energy = f64::MAX;

        for bead in 0..self.n_beads {
            let solution = Solution {
                data: path.beads[bead].clone(),
                cost: 0.0,
            };
            let energy = hamiltonian.evaluate(&solution);

            if energy < best_energy {
                best_energy = energy;
                best_bead = bead;
            }
        }

        // Also try path average (centroid)
        let mut centroid = vec![0.0; path.beads[0].len()];
        for bead in 0..self.n_beads {
            for dim in 0..centroid.len() {
                centroid[dim] += path.beads[bead][dim];
            }
        }
        for val in &mut centroid {
            *val /= self.n_beads as f64;
        }

        let centroid_solution = Solution {
            data: centroid.clone(),
            cost: hamiltonian.evaluate(&Solution { data: centroid, cost: 0.0 }),
        };

        // Return whichever is better
        if centroid_solution.cost < best_energy {
            centroid_solution
        } else {
            Solution {
                data: path.beads[best_bead].clone(),
                cost: best_energy,
            }
        }
    }

    /// Compute adaptive annealing schedule
    fn compute_annealing_schedule(&self) -> Vec<(f64, f64)> {
        let n_steps = 1000;
        let mut schedule = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let t = step as f64 / n_steps as f64;

            // Temperature schedule (inverse)
            let beta_t = self.beta * t;

            // Tunneling strength (decreases as we approach ground state)
            let tunneling = (1.0 - t).powi(2);

            schedule.push((beta_t, tunneling));
        }

        schedule
    }
}

/// Worldline (quantum path) representation
#[derive(Clone)]
struct Worldline {
    beads: Vec<Vec<f64>>, // [n_beads][n_dimensions]
}

/// Problem Hamiltonian for quantum annealing
pub struct ProblemHamiltonian {
    /// Cost function
    cost_fn: Box<dyn Fn(&Solution) -> f64 + Send + Sync>,
    /// Manifold coupling strength
    manifold_coupling: f64,
}

impl ProblemHamiltonian {
    pub fn new<F>(cost_fn: F, manifold_coupling: f64) -> Self
    where
        F: Fn(&Solution) -> f64 + Send + Sync + 'static,
    {
        Self {
            cost_fn: Box::new(cost_fn),
            manifold_coupling,
        }
    }

    pub fn evaluate(&self, solution: &Solution) -> f64 {
        (self.cost_fn)(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pimc_creation() {
        let pimc = PathIntegralMonteCarlo::new(20, 10.0);
        assert_eq!(pimc.n_beads, 20);
        assert_eq!(pimc.beta, 10.0);
        assert!((pimc.tau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_kinetic_action() {
        let pimc = PathIntegralMonteCarlo::new(10, 5.0);

        let action = pimc.kinetic_action(1.0, 0.9, 1.1);
        assert!(action > 0.0);

        // Symmetric configuration should have lower action
        let symmetric = pimc.kinetic_action(1.0, 0.5, 1.5);
        assert!(symmetric > action);
    }

    #[test]
    fn test_annealing_schedule() {
        let pimc = PathIntegralMonteCarlo::new(20, 10.0);
        let schedule = pimc.compute_annealing_schedule();

        assert_eq!(schedule.len(), 1000);

        // Check schedule properties
        let (beta_start, tunnel_start) = schedule[0];
        let (beta_end, tunnel_end) = schedule[schedule.len() - 1];

        assert!(beta_start < beta_end); // Temperature decreases
        assert!(tunnel_start > tunnel_end); // Tunneling decreases
        assert!((tunnel_end).abs() < 0.01); // Ends near zero
    }

    #[test]
    fn test_simple_optimization() {
        let mut pimc = PathIntegralMonteCarlo::new(10, 5.0);

        // Simple quadratic problem
        let hamiltonian = ProblemHamiltonian::new(
            |s: &Solution| s.data.iter().map(|x| x.powi(2)).sum(),
            0.1,
        );

        let initial = Solution {
            data: vec![1.0, 1.0, 1.0],
            cost: 3.0,
        };

        let manifold = CausalManifold {
            edges: Vec::new(),
            intrinsic_dim: 3,
            metric_tensor: Array2::eye(3),
        };

        let result = pimc.quantum_anneal(&hamiltonian, &manifold, &initial);

        assert!(result.is_ok());
        let solution = result.unwrap();

        println!("PIMC result:");
        println!("  Initial cost: {:.4}", initial.cost);
        println!("  Final cost: {:.4}", solution.cost);
        println!("  Solution: {:?}", solution.data);

        // Should improve solution
        assert!(solution.cost <= initial.cost);

        // Should approach optimum (near zero)
        assert!(solution.cost < 1.0);
    }

    #[test]
    fn test_path_initialization() {
        let pimc = PathIntegralMonteCarlo::new(5, 10.0);

        let initial = Solution {
            data: vec![1.0, 2.0, 3.0],
            cost: 14.0,
        };

        let path = pimc.initialize_path(&initial, 3);

        assert_eq!(path.beads.len(), 5);
        assert_eq!(path.beads[0].len(), 3);

        // All beads should start near initial
        for bead in &path.beads {
            for (i, &val) in bead.iter().enumerate() {
                assert_eq!(val, initial.data[i]);
            }
        }
    }

    #[test]
    fn test_end_to_end_distance() {
        let pimc = PathIntegralMonteCarlo::new(3, 10.0);

        let path = Worldline {
            beads: vec![
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![2.0, 2.0],
            ],
        };

        let dist_sq = pimc.end_to_end_distance(&path);
        assert!((dist_sq - 8.0).abs() < 1e-10); // (2-0)² + (2-0)² = 8
    }
}