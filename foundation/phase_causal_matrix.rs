//! Phase-Causal Matrix (PCM-Φ) Implementation
//!
//! Implements the core DRPP equation combining Kuramoto phase dynamics
//! with transfer entropy-based causal inference:
//!
//! **PCM-Φ**: Φ_ij = κ_ij * cos(θ_i - θ_j) + β_ij * TE(i→j)
//!
//! Where:
//! - κ_ij: Kuramoto coupling strength
//! - θ_i, θ_j: Phase angles of oscillators
//! - β_ij: Transfer entropy coupling weight
//! - TE(i→j): Transfer entropy from oscillator i to j
//!
//! This matrix captures both **phase synchronization** (Kuramoto term)
//! and **causal information flow** (transfer entropy term).

use anyhow::Result;
use ndarray::Array2;
use neuromorphic_engine::{TransferEntropyConfig, TransferEntropyEngine};

/// Phase-Causal Matrix configuration
#[derive(Debug, Clone)]
pub struct PcmConfig {
    /// Kuramoto coupling strength weight
    pub kappa_weight: f64,

    /// Transfer entropy coupling weight
    pub beta_weight: f64,

    /// Transfer entropy computation config
    pub te_config: TransferEntropyConfig,

    /// Normalization mode
    pub normalize: bool,
}

impl Default for PcmConfig {
    fn default() -> Self {
        Self {
            kappa_weight: 1.0, // Equal weight to synchronization
            beta_weight: 0.5,  // Moderate weight to causality
            te_config: TransferEntropyConfig::default(),
            normalize: true,
        }
    }
}

/// Phase-Causal Matrix processor
///
/// Computes the combined phase-synchronization and information-flow matrix
/// that governs DRPP dynamics.
pub struct PhaseCausalMatrixProcessor {
    config: PcmConfig,
    te_engine: TransferEntropyEngine,
}

impl PhaseCausalMatrixProcessor {
    /// Create new PCM processor
    pub fn new(config: PcmConfig) -> Self {
        let te_engine = TransferEntropyEngine::new(config.te_config.clone());

        Self { config, te_engine }
    }

    /// Compute Phase-Causal Matrix: Φ_ij = κ_ij * cos(θ_i - θ_j) + β_ij * TE(i→j)
    ///
    /// # Arguments
    /// * `phases` - Phase angles of oscillators [radians]
    /// * `time_series` - Historical time series for each oscillator (for TE calculation)
    /// * `kappa_matrix` - Kuramoto coupling strength matrix (optional, defaults to uniform)
    ///
    /// # Returns
    /// n×n Phase-Causal Matrix where element [i,j] represents the combined
    /// phase-synchronization and causal-information influence from i to j
    pub fn compute_pcm(
        &self,
        phases: &[f64],
        time_series: &[Vec<f64>],
        kappa_matrix: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>> {
        let n = phases.len();

        if time_series.len() != n {
            return Err(anyhow::anyhow!(
                "Phase count ({}) must match time series count ({})",
                n,
                time_series.len()
            ));
        }

        // Compute Kuramoto term: κ_ij * cos(θ_i - θ_j)
        let kuramoto_matrix = self.compute_kuramoto_term(phases, kappa_matrix)?;

        // Compute Transfer Entropy matrix: TE(i→j) for all pairs
        let te_matrix = self.te_engine.compute_te_matrix(time_series)?;

        // Combine: Φ_ij = κ * K_ij + β * TE_ij
        let mut pcm = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                pcm[[i, j]] = self.config.kappa_weight * kuramoto_matrix[[i, j]]
                    + self.config.beta_weight * te_matrix[[i, j]];
            }
        }

        // Normalize if requested
        if self.config.normalize {
            self.normalize_pcm(&mut pcm);
        }

        Ok(pcm)
    }

    /// Compute Kuramoto synchronization term: κ_ij * cos(θ_i - θ_j)
    fn compute_kuramoto_term(
        &self,
        phases: &[f64],
        kappa_matrix: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>> {
        let n = phases.len();
        let mut kuramoto = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let phase_diff = phases[i] - phases[j];
                    let sync_term = phase_diff.cos();

                    // Apply coupling strength
                    let kappa = if let Some(kappa_mat) = kappa_matrix {
                        kappa_mat[[i, j]]
                    } else {
                        1.0 / (n as f64) // Uniform all-to-all coupling
                    };

                    kuramoto[[i, j]] = kappa * sync_term;
                }
            }
        }

        Ok(kuramoto)
    }

    /// Normalize PCM matrix (row-wise normalization)
    fn normalize_pcm(&self, pcm: &mut Array2<f64>) {
        let n = pcm.nrows();

        for i in 0..n {
            let row_sum: f64 = pcm.row(i).iter().map(|x| x.abs()).sum();

            if row_sum > 1e-10 {
                for j in 0..n {
                    pcm[[i, j]] /= row_sum;
                }
            }
        }
    }

    /// Extract dominant causal pathways from PCM
    ///
    /// Returns list of (source, target, strength) tuples sorted by influence
    pub fn extract_causal_pathways(
        &self,
        pcm: &Array2<f64>,
        threshold: f64,
    ) -> Vec<(usize, usize, f64)> {
        let n = pcm.nrows();
        let mut pathways = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j && pcm[[i, j]].abs() > threshold {
                    pathways.push((i, j, pcm[[i, j]]));
                }
            }
        }

        // Sort by absolute strength (descending)
        pathways.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

        pathways
    }

    /// Compute DRPP phase evolution: dθ_k/dt = ω_k + Σ_j Φ_kj * sin(θ_j - θ_k)
    ///
    /// This is the DRPP-Δθ equation from the theoretical framework,
    /// where the Phase-Causal Matrix Φ governs coupling dynamics.
    ///
    /// # Arguments
    /// * `phases` - Current phase angles [radians]
    /// * `frequencies` - Natural frequencies of oscillators [Hz]
    /// * `pcm` - Phase-Causal Matrix
    /// * `dt` - Time step [seconds]
    ///
    /// # Returns
    /// Updated phase angles after one evolution step
    pub fn evolve_phases(
        &self,
        phases: &[f64],
        frequencies: &[f64],
        pcm: &Array2<f64>,
        dt: f64,
    ) -> Result<Vec<f64>> {
        let n = phases.len();

        if frequencies.len() != n {
            return Err(anyhow::anyhow!("Frequency count must match phase count"));
        }

        if pcm.nrows() != n || pcm.ncols() != n {
            return Err(anyhow::anyhow!("PCM dimensions must match phase count"));
        }

        let mut new_phases = vec![0.0; n];

        for k in 0..n {
            // Natural frequency term
            let mut dtheta = frequencies[k];

            // Coupling term: Σ_j Φ_kj * sin(θ_j - θ_k)
            for j in 0..n {
                if k != j {
                    let phase_diff = phases[j] - phases[k];
                    dtheta += pcm[[k, j]] * phase_diff.sin();
                }
            }

            // Euler integration
            new_phases[k] = phases[k] + dtheta * dt;

            // Wrap to [-π, π]
            new_phases[k] = ((new_phases[k] + std::f64::consts::PI) % (2.0 * std::f64::consts::PI))
                - std::f64::consts::PI;
        }

        Ok(new_phases)
    }

    /// Compute phase coherence order parameter: r = |⟨e^(iθ)⟩|
    pub fn compute_coherence(&self, phases: &[f64]) -> f64 {
        if phases.is_empty() {
            return 0.0;
        }

        let n = phases.len() as f64;
        let sum_cos: f64 = phases.iter().map(|&theta| theta.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|&theta| theta.sin()).sum();

        let mean_cos = sum_cos / n;
        let mean_sin = sum_sin / n;

        (mean_cos.powi(2) + mean_sin.powi(2)).sqrt()
    }

    /// Detect phase synchronization clusters
    ///
    /// Groups oscillators into clusters based on phase similarity
    pub fn detect_sync_clusters(&self, phases: &[f64], tolerance: f64) -> Vec<Vec<usize>> {
        let n = phases.len();
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; n];

        for i in 0..n {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..n {
                if !assigned[j] {
                    let phase_diff = (phases[i] - phases[j]).abs();
                    let phase_diff_wrapped =
                        phase_diff.min(2.0 * std::f64::consts::PI - phase_diff);

                    if phase_diff_wrapped < tolerance {
                        cluster.push(j);
                        assigned[j] = true;
                    }
                }
            }

            clusters.push(cluster);
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuramoto_term() {
        let config = PcmConfig::default();
        let processor = PhaseCausalMatrixProcessor::new(config);

        let phases = vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];

        let kuramoto = processor.compute_kuramoto_term(&phases, None).unwrap();

        // Check symmetry and bounds
        assert_eq!(kuramoto.nrows(), 3);
        assert_eq!(kuramoto.ncols(), 3);

        // Diagonal should be zero
        for i in 0..3 {
            assert_eq!(kuramoto[[i, i]], 0.0);
        }

        // Check coupling values are in [-1, 1] range
        for i in 0..3 {
            for j in 0..3 {
                assert!(kuramoto[[i, j]].abs() <= 1.0);
            }
        }
    }

    #[test]
    fn test_phase_evolution() {
        let config = PcmConfig::default();
        let processor = PhaseCausalMatrixProcessor::new(config);

        let phases = vec![0.0, 0.1, -0.1];
        let frequencies = vec![1.0, 1.0, 1.0];
        let pcm = Array2::from_shape_fn((3, 3), |(i, j)| if i != j { 0.1 } else { 0.0 });

        let new_phases = processor
            .evolve_phases(&phases, &frequencies, &pcm, 0.01)
            .unwrap();

        assert_eq!(new_phases.len(), 3);

        // Phases should have evolved
        for i in 0..3 {
            assert!(new_phases[i] != phases[i]);
        }

        // Phases should be in valid range
        for &phase in &new_phases {
            assert!(phase.abs() <= std::f64::consts::PI);
        }
    }

    #[test]
    fn test_coherence_calculation() {
        let config = PcmConfig::default();
        let processor = PhaseCausalMatrixProcessor::new(config);

        // Perfect synchronization
        let phases_sync = vec![0.0, 0.0, 0.0];
        let coherence_sync = processor.compute_coherence(&phases_sync);
        assert!((coherence_sync - 1.0).abs() < 1e-6);

        // Random phases (low coherence)
        let phases_random = vec![0.0, 2.0, 4.0];
        let coherence_random = processor.compute_coherence(&phases_random);
        assert!(coherence_random < 0.5);
    }

    #[test]
    fn test_sync_cluster_detection() {
        let config = PcmConfig::default();
        let processor = PhaseCausalMatrixProcessor::new(config);

        // Two clusters: [0, 1] and [2, 3]
        let phases = vec![0.0, 0.1, 3.0, 3.1];
        let tolerance = 0.2;

        let clusters = processor.detect_sync_clusters(&phases, tolerance);

        assert_eq!(clusters.len(), 2);
        assert!(clusters[0].len() == 2 || clusters[1].len() == 2);
    }
}
