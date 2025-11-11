//! Unified Multi-Phase RL State Representation
//!
//! Extends FluxNet state from Phase 2 only to ALL 4 pipeline phases.
//! Provides compact state representation for Q-table indexing while capturing
//! phase-specific metrics.
//!
//! # State Design
//!
//! The state space balances expressiveness with Q-table tractability:
//! - **Global metrics**: Chromatic number, conflicts, iteration progress, GPU utilization
//! - **Phase-specific metrics**: Reservoir quality, TE centrality, AI uncertainty, thermo band, quantum energy
//! - **Discretization**: 4-bit and 8-bit quantization for compact representation
//! - **Indexing**: Hash-based mapping to fixed-size Q-table (8K states)
//!
//! # GPU-First Compliance
//!
//! - ✅ All state metrics derived from GPU telemetry
//! - ✅ No CPU-specific metrics
//! - ✅ State updates respect phase boundaries
//! - ❌ NO global mutable state

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::telemetry::PhaseName;
use shared_types::ColoringSolution;

/// Unified multi-phase RL state
///
/// Compact representation for Q-table indexing.
/// Captures key metrics from all 4 phases.
///
/// # State Space Size
///
/// Full state space: 256^4 × 16^6 = 4.3 trillion states (intractable)
/// Hashed state space: 8192 states (tractable for tabular Q-learning)
///
/// # Discretization
///
/// - Global metrics: 8-bit (0-255)
/// - Phase metrics: 4-bit (0-15)
/// - Hash function: DefaultHasher with modulo 8192
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnifiedRLState {
    // ========== Global Metrics (8 bits each) ==========

    /// Discretized chromatic number (best_chromatic / max_expected * 255)
    pub chromatic_bin: u8,

    /// Discretized conflict count (log10(conflicts + 1) * 25)
    pub conflicts_bin: u8,

    /// Discretized iteration progress (iteration / max_iterations * 255)
    pub iteration_bin: u8,

    /// Discretized GPU utilization (SM% / 100 * 255)
    pub gpu_util_bin: u8,

    // ========== Phase-Specific Metrics (4 bits each, stored as u8) ==========

    /// Phase 0: Reservoir difficulty zone quality (0-15)
    /// Measures quality of difficulty-based vertex stratification
    pub phase0_difficulty_quality: u8,

    /// Phase 1: TE centrality concentration (0-15)
    /// Measures how concentrated high-TE edges are in hub vertices
    pub phase1_te_centrality: u8,

    /// Phase 1: AI uncertainty level (0-15)
    /// Measures active inference policy uncertainty
    pub phase1_ai_uncertainty: u8,

    /// Phase 2: Thermodynamic temperature band (0-15)
    /// Current position in temperature ladder
    pub phase2_temp_band: u8,

    /// Phase 2: Conflict escape rate (0-15)
    /// Rate of conflict reduction per temperature
    pub phase2_escape_rate: u8,

    /// Phase 3: QUBO energy quality (0-15)
    /// Relative QUBO energy (lower is better)
    pub phase3_qubo_quality: u8,

    // ========== Current Context ==========

    /// Current phase (used for phase-specific action filtering)
    pub current_phase: PhaseName,
}

impl UnifiedRLState {
    /// Create state from Phase 0 (Reservoir) output
    ///
    /// Captures difficulty zone quality and initial solution metrics.
    pub fn from_phase0(
        solution: &ColoringSolution,
        difficulty_zones: &[Vec<usize>],
        max_chromatic: usize,
        iteration: usize,
        max_iterations: usize,
        gpu_util: f32,
    ) -> Self {
        // Global metrics
        let chromatic_bin = ((solution.chromatic_number as f32 / max_chromatic as f32) * 255.0)
            .min(255.0) as u8;
        let conflicts_bin = ((solution.conflicts as f32 + 1.0).log10() * 25.0)
            .min(255.0) as u8;
        let iteration_bin = ((iteration as f32 / max_iterations as f32) * 255.0)
            .min(255.0) as u8;
        let gpu_util_bin = ((gpu_util / 100.0) * 255.0).min(255.0) as u8;

        // Phase 0 specific: Difficulty zone quality
        // High quality = more balanced zone sizes
        let zone_sizes: Vec<usize> = difficulty_zones.iter().map(|z| z.len()).collect();
        let zone_quality = if !zone_sizes.is_empty() {
            let mean = zone_sizes.iter().sum::<usize>() as f32 / zone_sizes.len() as f32;
            let variance = zone_sizes
                .iter()
                .map(|&s| {
                    let diff = s as f32 - mean;
                    diff * diff
                })
                .sum::<f32>()
                / zone_sizes.len() as f32;
            let cv = if mean > 0.0 {
                variance.sqrt() / mean
            } else {
                1.0
            };
            // Lower CV = higher quality, map to 0-15
            ((1.0 - cv.min(1.0)) * 15.0) as u8
        } else {
            0
        };

        Self {
            chromatic_bin,
            conflicts_bin,
            iteration_bin,
            gpu_util_bin,
            phase0_difficulty_quality: zone_quality,
            phase1_te_centrality: 0,
            phase1_ai_uncertainty: 0,
            phase2_temp_band: 0,
            phase2_escape_rate: 0,
            phase3_qubo_quality: 0,
            current_phase: PhaseName::Reservoir,
        }
    }

    /// Create state from Phase 1 (Transfer Entropy) output
    ///
    /// Captures TE centrality and AI uncertainty.
    pub fn from_phase1(
        solution: &ColoringSolution,
        te_hub_concentration: f32,
        ai_uncertainty: f32,
        max_chromatic: usize,
        iteration: usize,
        max_iterations: usize,
        gpu_util: f32,
    ) -> Self {
        // Global metrics
        let chromatic_bin = ((solution.chromatic_number as f32 / max_chromatic as f32) * 255.0)
            .min(255.0) as u8;
        let conflicts_bin = ((solution.conflicts as f32 + 1.0).log10() * 25.0)
            .min(255.0) as u8;
        let iteration_bin = ((iteration as f32 / max_iterations as f32) * 255.0)
            .min(255.0) as u8;
        let gpu_util_bin = ((gpu_util / 100.0) * 255.0).min(255.0) as u8;

        // Phase 1 specific
        let te_centrality = (te_hub_concentration.clamp(0.0, 1.0) * 15.0) as u8;
        let ai_uncertainty_bin = (ai_uncertainty.clamp(0.0, 1.0) * 15.0) as u8;

        Self {
            chromatic_bin,
            conflicts_bin,
            iteration_bin,
            gpu_util_bin,
            phase0_difficulty_quality: 0,
            phase1_te_centrality: te_centrality,
            phase1_ai_uncertainty: ai_uncertainty_bin,
            phase2_temp_band: 0,
            phase2_escape_rate: 0,
            phase3_qubo_quality: 0,
            current_phase: PhaseName::TransferEntropy,
        }
    }

    /// Create state from Phase 2 (Thermodynamic) output
    ///
    /// Captures temperature band and escape rate.
    pub fn from_phase2(
        solution: &ColoringSolution,
        temp_idx: usize,
        num_temps: usize,
        conflicts_before: usize,
        max_chromatic: usize,
        iteration: usize,
        max_iterations: usize,
        gpu_util: f32,
    ) -> Self {
        // Global metrics
        let chromatic_bin = ((solution.chromatic_number as f32 / max_chromatic as f32) * 255.0)
            .min(255.0) as u8;
        let conflicts_bin = ((solution.conflicts as f32 + 1.0).log10() * 25.0)
            .min(255.0) as u8;
        let iteration_bin = ((iteration as f32 / max_iterations as f32) * 255.0)
            .min(255.0) as u8;
        let gpu_util_bin = ((gpu_util / 100.0) * 255.0).min(255.0) as u8;

        // Phase 2 specific
        let temp_band = ((temp_idx as f32 / num_temps as f32) * 15.0).min(15.0) as u8;

        // Escape rate: How quickly conflicts reduced
        let conflicts_after = solution.conflicts;
        let reduction_ratio = if conflicts_before > 0 {
            1.0 - (conflicts_after as f32 / conflicts_before as f32)
        } else {
            0.0
        };
        let escape_rate = (reduction_ratio.clamp(0.0, 1.0) * 15.0) as u8;

        Self {
            chromatic_bin,
            conflicts_bin,
            iteration_bin,
            gpu_util_bin,
            phase0_difficulty_quality: 0,
            phase1_te_centrality: 0,
            phase1_ai_uncertainty: 0,
            phase2_temp_band: temp_band,
            phase2_escape_rate: escape_rate,
            phase3_qubo_quality: 0,
            current_phase: PhaseName::Thermodynamic,
        }
    }

    /// Create state from Phase 3 (Quantum) output
    ///
    /// Captures QUBO energy quality.
    pub fn from_phase3(
        solution: &ColoringSolution,
        qubo_energy: f64,
        qubo_energy_min: f64,
        qubo_energy_max: f64,
        max_chromatic: usize,
        iteration: usize,
        max_iterations: usize,
        gpu_util: f32,
    ) -> Self {
        // Global metrics
        let chromatic_bin = ((solution.chromatic_number as f32 / max_chromatic as f32) * 255.0)
            .min(255.0) as u8;
        let conflicts_bin = ((solution.conflicts as f32 + 1.0).log10() * 25.0)
            .min(255.0) as u8;
        let iteration_bin = ((iteration as f32 / max_iterations as f32) * 255.0)
            .min(255.0) as u8;
        let gpu_util_bin = ((gpu_util / 100.0) * 255.0).min(255.0) as u8;

        // Phase 3 specific: QUBO energy quality (normalized to 0-15)
        let energy_range = qubo_energy_max - qubo_energy_min;
        let qubo_quality = if energy_range > 0.0 {
            let normalized = ((qubo_energy - qubo_energy_min) / energy_range).clamp(0.0, 1.0);
            // Lower energy = higher quality, so invert
            ((1.0 - normalized) * 15.0) as u8
        } else {
            7 // Middle value if no range
        };

        Self {
            chromatic_bin,
            conflicts_bin,
            iteration_bin,
            gpu_util_bin,
            phase0_difficulty_quality: 0,
            phase1_te_centrality: 0,
            phase1_ai_uncertainty: 0,
            phase2_temp_band: 0,
            phase2_escape_rate: 0,
            phase3_qubo_quality: qubo_quality,
            current_phase: PhaseName::Quantum,
        }
    }

    /// Convert state to compact index for Q-table lookup
    ///
    /// Uses hash-based mapping to 8K state space.
    /// This allows tractable Q-table size while preserving state diversity.
    ///
    /// # Arguments
    /// - `table_size`: Size of Q-table state space (typically 8192)
    ///
    /// # Returns
    /// State index in range [0, table_size)
    pub fn to_index(&self, table_size: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as usize) % table_size
    }

    /// Get current phase
    pub fn phase(&self) -> PhaseName {
        self.current_phase
    }

    /// Get chromatic number (denormalized)
    pub fn chromatic_approx(&self, max_chromatic: usize) -> usize {
        ((self.chromatic_bin as f32 / 255.0) * max_chromatic as f32) as usize
    }

    /// Get conflicts (denormalized from log scale)
    pub fn conflicts_approx(&self) -> usize {
        let log_conflicts = self.conflicts_bin as f32 / 25.0;
        (10.0_f32.powf(log_conflicts) - 1.0) as usize
    }

    /// Get iteration progress (0.0 to 1.0)
    pub fn iteration_progress(&self) -> f32 {
        self.iteration_bin as f32 / 255.0
    }

    /// Get GPU utilization (0.0 to 100.0)
    pub fn gpu_utilization(&self) -> f32 {
        (self.gpu_util_bin as f32 / 255.0) * 100.0
    }

    /// Check if state is in critical condition (high conflicts, late iteration)
    pub fn is_critical(&self) -> bool {
        self.conflicts_bin > 200 && self.iteration_bin > 200
    }

    /// Check if state is improving (low conflicts, early iteration)
    pub fn is_improving(&self) -> bool {
        self.conflicts_bin < 50 && self.chromatic_bin < 128
    }
}

impl Default for UnifiedRLState {
    fn default() -> Self {
        Self {
            chromatic_bin: 128,
            conflicts_bin: 128,
            iteration_bin: 0,
            gpu_util_bin: 0,
            phase0_difficulty_quality: 7,
            phase1_te_centrality: 7,
            phase1_ai_uncertainty: 7,
            phase2_temp_band: 7,
            phase2_escape_rate: 7,
            phase3_qubo_quality: 7,
            current_phase: PhaseName::Reservoir,
        }
    }
}

impl std::fmt::Display for UnifiedRLState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RLState[phase={:?}, chrom={}, conf={}, iter={}, gpu={}%]",
            self.current_phase,
            self.chromatic_bin,
            self.conflicts_bin,
            self.iteration_bin,
            self.gpu_util_bin
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::Graph;

    #[test]
    fn test_state_indexing() {
        let state = UnifiedRLState::default();
        let idx1 = state.to_index(8192);
        let idx2 = state.to_index(8192);

        // Same state should map to same index
        assert_eq!(idx1, idx2);
        assert!(idx1 < 8192);
    }

    #[test]
    fn test_state_from_phase0() {
        let graph = Graph::new(10, vec![]);
        let solution = ColoringSolution::new(graph);
        let zones = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8, 9]];

        let state = UnifiedRLState::from_phase0(&solution, &zones, 200, 10, 100, 50.0);

        assert_eq!(state.current_phase, PhaseName::Reservoir);
        assert!(state.phase0_difficulty_quality <= 15);
        assert!(state.chromatic_bin <= 255);
    }

    #[test]
    fn test_state_from_phase1() {
        let graph = Graph::new(10, vec![]);
        let solution = ColoringSolution::new(graph);

        let state = UnifiedRLState::from_phase1(&solution, 0.75, 0.5, 200, 20, 100, 60.0);

        assert_eq!(state.current_phase, PhaseName::TransferEntropy);
        assert!(state.phase1_te_centrality <= 15);
        assert!(state.phase1_ai_uncertainty <= 15);
    }

    #[test]
    fn test_state_from_phase2() {
        let graph = Graph::new(10, vec![]);
        let solution = ColoringSolution::new(graph);

        let state = UnifiedRLState::from_phase2(&solution, 5, 20, 100, 200, 30, 100, 70.0);

        assert_eq!(state.current_phase, PhaseName::Thermodynamic);
        assert!(state.phase2_temp_band <= 15);
        assert!(state.phase2_escape_rate <= 15);
    }

    #[test]
    fn test_state_from_phase3() {
        let graph = Graph::new(10, vec![]);
        let solution = ColoringSolution::new(graph);

        let state = UnifiedRLState::from_phase3(&solution, 50.0, 0.0, 100.0, 200, 40, 100, 80.0);

        assert_eq!(state.current_phase, PhaseName::Quantum);
        assert!(state.phase3_qubo_quality <= 15);
    }

    #[test]
    fn test_state_conditions() {
        let mut state = UnifiedRLState::default();

        // Critical condition
        state.conflicts_bin = 220;
        state.iteration_bin = 220;
        assert!(state.is_critical());
        assert!(!state.is_improving());

        // Improving condition
        state.conflicts_bin = 30;
        state.chromatic_bin = 100;
        state.iteration_bin = 50;
        assert!(!state.is_critical());
        assert!(state.is_improving());
    }

    #[test]
    fn test_denormalization() {
        let mut state = UnifiedRLState::default();
        state.chromatic_bin = 128; // 50% of max
        state.conflicts_bin = 100; // log10(10^4) = 4 -> 4*25 = 100
        state.gpu_util_bin = 128; // 50% utilization

        assert_eq!(state.chromatic_approx(200), 100);
        assert!(state.conflicts_approx() > 9000 && state.conflicts_approx() < 11000);
        assert!((state.gpu_utilization() - 50.0).abs() < 1.0);
    }
}
