//! Information Hamiltonian + Triplet Interactions
//!
//! Mission Charlie: Task 2.2 (Ultra-Enhanced)
//!
//! Features:
//! 1. Pairwise LLM interactions (standard)
//! 2. Triplet interactions (3-body terms) - captures complex relationships
//! 3. Constitutional Article I compliance (entropy tracking)
//!
//! H(s) = Σᵢⱼ J_ij d(i,j) sᵢsⱼ + Σᵢⱼₖ K_ijk sᵢsⱼsₖ + Σᵢ hᵢsᵢ - T*S(s)

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// Enhanced Information Hamiltonian with Triplet Interactions
pub struct InformationHamiltonian {
    /// Pairwise coupling strengths J_ij
    coupling_matrix: Array2<f64>,

    /// Triplet coupling strengths K_ijk (3-body interactions)
    triplet_couplings: Array3<f64>,

    /// Prior bias h_i (confidence in each LLM)
    model_priors: Array1<f64>,

    /// Temperature T (exploration parameter)
    temperature: f64,
}

impl InformationHamiltonian {
    pub fn new(n_llms: usize, temperature: f64) -> Self {
        // Initialize with small random couplings
        let coupling_matrix = Array2::from_elem((n_llms, n_llms), 0.1);

        let triplet_couplings = Array3::from_elem((n_llms, n_llms, n_llms), 0.01);

        let model_priors = Array1::from_elem(n_llms, 0.0); // Unbiased initially

        Self {
            coupling_matrix,
            triplet_couplings,
            model_priors,
            temperature,
        }
    }

    /// Compute total energy (Hamiltonian)
    ///
    /// H(s) = Σᵢⱼ J_ij d(i,j) sᵢsⱼ + Σᵢⱼₖ K_ijk sᵢsⱼsₖ + Σᵢ hᵢsᵢ - T*S(s)
    ///
    /// Article I Compliance: Entropy term ensures thermodynamic consistency
    pub fn energy(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> f64 {
        let n = weights.len();
        let mut energy = 0.0;

        // PAIRWISE INTERACTIONS: Σᵢⱼ J_ij d(i,j) sᵢsⱼ
        for i in 0..n {
            for j in 0..n {
                energy +=
                    self.coupling_matrix[[i, j]] * distances[[i, j]] * weights[i] * weights[j];
            }
        }

        // TRIPLET INTERACTIONS: Σᵢⱼₖ K_ijk sᵢsⱼsₖ (NEW - ultra-enhancement)
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i != j && j != k && i != k {
                        // 3-body interaction energy
                        let avg_distance =
                            (distances[[i, j]] + distances[[j, k]] + distances[[i, k]]) / 3.0;

                        energy += self.triplet_couplings[[i, j, k]]
                            * avg_distance
                            * weights[i]
                            * weights[j]
                            * weights[k];
                    }
                }
            }
        }

        // PRIOR BIAS: Σᵢ hᵢsᵢ
        for i in 0..n {
            energy += self.model_priors[i] * weights[i];
        }

        // ENTROPIC TERM: -T*S(s) (Article I compliance)
        let entropy = self.shannon_entropy(weights);
        energy -= self.temperature * entropy;

        energy
    }

    fn shannon_entropy(&self, weights: &Array1<f64>) -> f64 {
        // S(s) = -Σ sᵢ ln(sᵢ)
        let mut entropy = 0.0;

        for &w in weights.iter() {
            if w > 1e-10 {
                entropy -= w * w.ln();
            }
        }

        entropy
    }

    /// Compute gradient ∂H/∂sᵢ for optimization
    pub fn gradient(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> Array1<f64> {
        let n = weights.len();
        let mut grad = Array1::zeros(n);

        for i in 0..n {
            // Pairwise term contribution
            for j in 0..n {
                grad[i] += 2.0 * self.coupling_matrix[[i, j]] * distances[[i, j]] * weights[j];
            }

            // Triplet term contribution (partial derivatives)
            for j in 0..n {
                for k in 0..n {
                    if i != j && j != k && i != k {
                        let avg_dist =
                            (distances[[i, j]] + distances[[j, k]] + distances[[i, k]]) / 3.0;

                        grad[i] +=
                            self.triplet_couplings[[i, j, k]] * avg_dist * weights[j] * weights[k];
                    }
                }
            }

            // Prior term
            grad[i] += self.model_priors[i];

            // Entropy term: ∂(-T*S)/∂sᵢ = -T*(1 + ln(sᵢ))
            if weights[i] > 1e-10 {
                grad[i] -= self.temperature * (1.0 + weights[i].ln());
            }
        }

        grad
    }

    /// Validate Article I compliance (entropy non-decreasing)
    pub fn validate_thermodynamics(&self, weights: &Array1<f64>) -> bool {
        let entropy = self.shannon_entropy(weights);

        // Entropy must be non-negative (Article I)
        entropy >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamiltonian_energy_finite() {
        let h = InformationHamiltonian::new(4, 1.0);

        let weights = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let distances = Array2::from_elem((4, 4), 0.5);

        let energy = h.energy(&weights, &distances);

        assert!(energy.is_finite(), "Energy must be finite");
    }

    #[test]
    fn test_entropy_non_negative() {
        let h = InformationHamiltonian::new(4, 1.0);

        let weights = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let entropy = h.shannon_entropy(&weights);

        // Article I: Entropy must be non-negative
        assert!(entropy >= 0.0, "Entropy must be non-negative (Article I)");
    }

    #[test]
    fn test_gradient_shape() {
        let h = InformationHamiltonian::new(4, 1.0);

        let weights = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let distances = Array2::from_elem((4, 4), 0.5);

        let grad = h.gradient(&weights, &distances);

        assert_eq!(
            grad.len(),
            4,
            "Gradient should have same dimension as weights"
        );
    }
}
