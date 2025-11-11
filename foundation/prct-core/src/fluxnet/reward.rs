//! Multi-Phase RL Reward Computation
//!
//! Defines reward functions for all 4 pipeline phases.
//! Rewards incentivize chromatic reduction, conflict resolution, and GPU efficiency.
//!
//! # Reward Design Principles
//!
//! 1. **Primary Goal**: Chromatic number reduction (100 points per color)
//! 2. **Secondary Goal**: Conflict resolution (0.1 points per conflict)
//! 3. **Efficiency**: Time penalty (-0.01 per second) and GPU bonus (+5 for >20% util)
//! 4. **Phase-Specific Bonuses**: Reward phase-appropriate behaviors
//!
//! # GPU-First Compliance
//!
//! - ✅ All rewards derived from GPU telemetry
//! - ✅ Reward computation uses f64 for precision
//! - ✅ No CPU-specific penalties

use crate::telemetry::PhaseName;

/// Compute reward for a phase transition
///
/// Reward = Δchromatic × 100 + Δconflicts × 0.1 - time_s × 0.01 + gpu_bonus + phase_bonus
///
/// # Arguments
/// - `phase`: Current phase
/// - `prev_chromatic`: Chromatic number before phase
/// - `curr_chromatic`: Chromatic number after phase
/// - `prev_conflicts`: Conflicts before phase
/// - `curr_conflicts`: Conflicts after phase
/// - `duration_ms`: Phase execution time in milliseconds
/// - `gpu_util_avg`: Average GPU SM utilization (0-100%)
///
/// # Returns
/// Reward value (higher is better)
pub fn compute_reward(
    phase: PhaseName,
    prev_chromatic: usize,
    curr_chromatic: usize,
    prev_conflicts: usize,
    curr_conflicts: usize,
    duration_ms: f64,
    gpu_util_avg: f32,
) -> f64 {
    // Primary: Chromatic reduction (100 points per color)
    let color_reward = (prev_chromatic as i32 - curr_chromatic as i32) as f64 * 100.0;

    // Secondary: Conflict reduction (0.1 points per conflict)
    let conflict_reward = (prev_conflicts as i32 - curr_conflicts as i32) as f64 * 0.1;

    // Efficiency: Time penalty (-0.01 per second)
    let time_penalty = -(duration_ms / 1000.0) * 0.01;

    // GPU utilization bonus (encourage GPU usage)
    let gpu_bonus = if gpu_util_avg > 20.0 { 5.0 } else { 0.0 };

    // Phase-specific bonuses
    let phase_bonus = phase_specific_bonus(
        phase,
        prev_chromatic,
        curr_chromatic,
        prev_conflicts,
        curr_conflicts,
    );

    color_reward + conflict_reward + time_penalty + gpu_bonus + phase_bonus
}

/// Compute phase-specific bonus rewards
///
/// Encourages phase-appropriate behaviors:
/// - **Phase 0 (Reservoir)**: Bonus for creating good difficulty stratification
/// - **Phase 1 (TE)**: Bonus for strong initial chromatic reduction
/// - **Phase 2 (Thermo)**: Big bonus for escaping to valid coloring
/// - **Phase 3 (Quantum)**: Bonus for QUBO refinement
fn phase_specific_bonus(
    phase: PhaseName,
    prev_chromatic: usize,
    curr_chromatic: usize,
    prev_conflicts: usize,
    curr_conflicts: usize,
) -> f64 {
    match phase {
        PhaseName::Reservoir => {
            // Bonus for reducing conflicts (good difficulty zones)
            if prev_conflicts > curr_conflicts && curr_conflicts < prev_conflicts / 2 {
                10.0
            } else {
                0.0
            }
        }

        PhaseName::TransferEntropy | PhaseName::ActiveInference => {
            // Bonus for strong initial chromatic reduction
            let color_reduction = (prev_chromatic as i32 - curr_chromatic as i32) as f64;
            if color_reduction >= 5.0 {
                50.0
            } else {
                0.0
            }
        }

        PhaseName::Thermodynamic => {
            // Big bonus for escaping to valid coloring
            if prev_conflicts > 10000 && curr_conflicts == 0 {
                100.0
            } else if prev_conflicts > 1000 && curr_conflicts == 0 {
                50.0
            } else if prev_conflicts > 0 && curr_conflicts == 0 {
                30.0
            } else {
                0.0
            }
        }

        PhaseName::Quantum => {
            // Bonus for QUBO refinement (color reduction with no conflicts)
            let color_reduction = (prev_chromatic as i32 - curr_chromatic as i32) as f64;
            if color_reduction > 0.0 && curr_conflicts == 0 {
                30.0
            } else {
                0.0
            }
        }

        PhaseName::Memetic | PhaseName::Ensemble | PhaseName::Validation => {
            // No specific bonus for these phases
            0.0
        }
    }
}

/// Compute normalized reward (for logging/comparison)
///
/// Normalizes reward to [-1.0, 1.0] range based on expected value ranges.
pub fn normalize_reward(reward: f64) -> f64 {
    // Expected reward range: [-100, 1000]
    // (worst case: -1 color, +10000 conflicts, 1000s runtime = -100)
    // (best case: +10 colors, -10000 conflicts = 1000 + 1000 = 2000)
    let normalized = reward / 2000.0;
    normalized.clamp(-1.0, 1.0)
}

/// Compute reward shaping for continuous learning
///
/// Applies potential-based reward shaping to encourage exploration.
/// See Ng et al. "Policy Invariance Under Reward Transformations".
///
/// # Arguments
/// - `state_value`: V(s) from value function estimate
/// - `next_state_value`: V(s') from value function estimate
/// - `gamma`: Discount factor
///
/// # Returns
/// Shaped reward F(s, a, s') = γV(s') - V(s)
pub fn shape_reward(state_value: f64, next_state_value: f64, gamma: f64) -> f64 {
    gamma * next_state_value - state_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_chromatic_reward() {
        // Reduce by 5 colors
        let reward = compute_reward(
            PhaseName::TransferEntropy,
            120,
            115,
            1000,
            1000,
            5000.0,
            50.0,
        );

        // 5 colors × 100 = 500 + gpu_bonus(5) + phase_bonus(50) - time_penalty(0.05) ≈ 554
        assert!(reward > 500.0);
        assert!(reward < 600.0);
    }

    #[test]
    fn test_negative_chromatic_penalty() {
        // Increase by 2 colors
        let reward = compute_reward(
            PhaseName::Quantum,
            115,
            117,
            0,
            0,
            5000.0,
            50.0,
        );

        // -2 colors × 100 = -200 + gpu_bonus(5) - time_penalty(0.05) ≈ -195
        assert!(reward < -190.0);
        assert!(reward > -200.0);
    }

    #[test]
    fn test_conflict_resolution_reward() {
        // Resolve 10000 conflicts
        let reward = compute_reward(
            PhaseName::Thermodynamic,
            120,
            120,
            10000,
            0,
            30000.0,
            70.0,
        );

        // 10000 conflicts × 0.1 = 1000 + phase_bonus(100) + gpu_bonus(5) - time_penalty(0.3) ≈ 1104
        assert!(reward > 1100.0);
    }

    #[test]
    fn test_time_penalty() {
        // Long execution time (100 seconds)
        let reward = compute_reward(
            PhaseName::Reservoir,
            120,
            120,
            1000,
            1000,
            100000.0,
            50.0,
        );

        // time_penalty = -100s × 0.01 = -1.0 + gpu_bonus(5) ≈ 4
        assert!(reward > 3.0);
        assert!(reward < 6.0);
    }

    #[test]
    fn test_gpu_utilization_bonus() {
        // High GPU utilization
        let reward_high = compute_reward(
            PhaseName::Thermodynamic,
            120,
            120,
            1000,
            1000,
            5000.0,
            80.0,
        );

        // Low GPU utilization
        let reward_low = compute_reward(
            PhaseName::Thermodynamic,
            120,
            120,
            1000,
            1000,
            5000.0,
            10.0,
        );

        // High GPU should get 5-point bonus
        assert!((reward_high - reward_low - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_phase_specific_bonuses() {
        // Phase 0: Reservoir (conflict reduction bonus)
        let reward_p0 = compute_reward(
            PhaseName::Reservoir,
            120,
            120,
            10000,
            4000,
            5000.0,
            50.0,
        );
        // Should get 10-point bonus
        assert!(reward_p0 > 600.0); // 6000 conflicts × 0.1 + 10 + 5 - 0.05

        // Phase 1: TE (chromatic reduction bonus)
        let reward_p1 = compute_reward(
            PhaseName::TransferEntropy,
            120,
            115,
            1000,
            1000,
            5000.0,
            50.0,
        );
        // Should get 50-point bonus for 5+ color reduction
        assert!(reward_p1 > 550.0); // 500 + 50 + 5 - 0.05

        // Phase 2: Thermo (escape bonus)
        let reward_p2 = compute_reward(
            PhaseName::Thermodynamic,
            120,
            120,
            15000,
            0,
            30000.0,
            70.0,
        );
        // Should get 100-point bonus for escaping from >10k conflicts
        assert!(reward_p2 > 1600.0); // 1500 + 100 + 5 - 0.3

        // Phase 3: Quantum (refinement bonus)
        let reward_p3 = compute_reward(
            PhaseName::Quantum,
            120,
            118,
            0,
            0,
            5000.0,
            50.0,
        );
        // Should get 30-point bonus for color reduction with no conflicts
        assert!(reward_p3 > 230.0); // 200 + 30 + 5 - 0.05
    }

    #[test]
    fn test_normalization() {
        // Very positive reward
        let reward = 1000.0;
        let normalized = normalize_reward(reward);
        assert!(normalized > 0.4);
        assert!(normalized <= 1.0);

        // Very negative reward
        let reward = -100.0;
        let normalized = normalize_reward(reward);
        assert!(normalized < 0.0);
        assert!(normalized >= -1.0);

        // Zero reward
        let reward = 0.0;
        let normalized = normalize_reward(reward);
        assert!((normalized - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_reward_shaping() {
        // Moving to better state
        let shaped = shape_reward(10.0, 20.0, 0.95);
        assert!((shaped - 9.0).abs() < 0.1); // 0.95 × 20 - 10 = 9

        // Moving to worse state
        let shaped = shape_reward(20.0, 10.0, 0.95);
        assert!((shaped - (-10.5)).abs() < 0.1); // 0.95 × 10 - 20 = -10.5
    }
}
