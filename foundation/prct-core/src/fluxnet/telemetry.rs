//! FluxNet RL Telemetry Extension
//!
//! Provides telemetry data structures for tracking FluxNet RL decisions,
//! force band statistics, and Q-learning updates during Phase 2 execution.

use serde::{Deserialize, Serialize};
use crate::fluxnet::{FluxNetAction, UnifiedRLState};
use crate::telemetry::PhaseName;

/// FluxNet-specific telemetry data
///
/// This structure is serialized to JSON and embedded in the `parameters` field
/// of `RunMetric` during Phase 2 (Thermodynamic) execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetTelemetry {
    /// Force band statistics (from ForceProfile)
    pub force_bands: ForceBandTelemetry,

    /// RL action and decision info
    pub rl_decision: RLDecisionTelemetry,

    /// Q-learning update details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_update: Option<QUpdateTelemetry>,

    /// FluxNet configuration snapshot
    pub config: FluxNetConfigSnapshot,
}

impl FluxNetTelemetry {
    /// Create new FluxNet telemetry snapshot
    pub fn new(
        force_bands: ForceBandTelemetry,
        rl_decision: RLDecisionTelemetry,
        q_update: Option<QUpdateTelemetry>,
        config: FluxNetConfigSnapshot,
    ) -> Self {
        Self {
            force_bands,
            rl_decision,
            q_update,
            config,
        }
    }
}

/// Force band statistics from ForceProfile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceBandTelemetry {
    /// Fraction of vertices in Strong band [0.0, 1.0]
    pub strong_fraction: f32,

    /// Fraction of vertices in Weak band [0.0, 1.0]
    pub weak_fraction: f32,

    /// Fraction of vertices in Neutral band [0.0, 1.0]
    pub neutral_fraction: f32,

    /// Mean force multiplier across all vertices
    pub mean_force: f32,

    /// Min force multiplier
    pub min_force: f32,

    /// Max force multiplier
    pub max_force: f32,

    /// Standard deviation of force multipliers
    pub force_stddev: f32,
}

impl ForceBandTelemetry {
    /// Create from force band counts and statistics
    pub fn from_stats(
        strong_count: usize,
        neutral_count: usize,
        weak_count: usize,
        total_vertices: usize,
        mean_force: f32,
        min_force: f32,
        max_force: f32,
        force_stddev: f32,
    ) -> Self {
        let total = total_vertices as f32;
        Self {
            strong_fraction: (strong_count as f32) / total,
            neutral_fraction: (neutral_count as f32) / total,
            weak_fraction: (weak_count as f32) / total,
            mean_force,
            min_force,
            max_force,
            force_stddev,
        }
    }

    /// Create from ForceBandStats (from ForceProfile)
    pub fn from_force_band_stats(stats: &crate::fluxnet::ForceBandStats) -> Self {
        Self {
            strong_fraction: stats.strong_fraction,
            neutral_fraction: stats.neutral_fraction,
            weak_fraction: stats.weak_fraction,
            mean_force: stats.mean_force,
            min_force: stats.min_force,
            max_force: stats.max_force,
            force_stddev: stats.std_force,
        }
    }
}

/// RL decision and action telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLDecisionTelemetry {
    /// Phase when decision was made
    pub phase: String,

    /// RL state observation (discretized)
    pub state: RLStateTelemetry,

    /// Action taken by RL controller
    pub action: FluxNetAction,

    /// Q-value for selected action (before update)
    pub q_value: f32,

    /// Exploration epsilon at decision time
    pub epsilon: f32,

    /// Whether action was exploratory (random) or exploitative (greedy)
    pub was_exploration: bool,

    /// State index (hashed)
    pub state_index_hashed: usize,

    /// State index (adaptive) - None if adaptive indexing not ready
    pub state_index_adaptive: Option<usize>,
}

impl RLDecisionTelemetry {
    /// Create from RL controller state and action
    pub fn new(
        phase: PhaseName,
        state: &UnifiedRLState,
        action: FluxNetAction,
        q_value: f32,
        epsilon: f32,
        was_exploration: bool,
        table_size: usize,
        state_index_adaptive: Option<usize>,
    ) -> Self {
        Self {
            phase: format!("{:?}", phase),
            state: RLStateTelemetry::from_unified_state(state),
            action,
            q_value,
            epsilon,
            was_exploration,
            state_index_hashed: state.to_index(table_size),
            state_index_adaptive,
        }
    }
}

/// RL state observation for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLStateTelemetry {
    /// Phase name
    pub phase: String,

    /// Chromatic bin (0-255)
    pub chromatic_bin: u8,

    /// Conflicts bin (0-255)
    pub conflicts_bin: u8,

    /// Iteration bin (0-255)
    pub iteration_bin: u8,

    /// GPU utilization bin (0-255)
    pub gpu_util_bin: u8,

    /// Phase 0 difficulty/quality score
    pub phase0_difficulty_quality: u8,

    /// Phase 1 TE centrality
    pub phase1_te_centrality: u8,

    /// Phase 1 AI uncertainty
    pub phase1_ai_uncertainty: u8,

    /// Phase 2 temperature band
    pub phase2_temp_band: u8,

    /// Phase 2 escape rate
    pub phase2_escape_rate: u8,

    /// Phase 3 QUBO quality
    pub phase3_qubo_quality: u8,
}

impl RLStateTelemetry {
    /// Create from UnifiedRLState
    pub fn from_unified_state(state: &UnifiedRLState) -> Self {
        Self {
            phase: format!("{:?}", state.phase()),
            chromatic_bin: state.chromatic_bin,
            conflicts_bin: state.conflicts_bin,
            iteration_bin: state.iteration_bin,
            gpu_util_bin: state.gpu_util_bin,
            phase0_difficulty_quality: state.phase0_difficulty_quality,
            phase1_te_centrality: state.phase1_te_centrality,
            phase1_ai_uncertainty: state.phase1_ai_uncertainty,
            phase2_temp_band: state.phase2_temp_band,
            phase2_escape_rate: state.phase2_escape_rate,
            phase3_qubo_quality: state.phase3_qubo_quality,
        }
    }
}

/// Q-learning update telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QUpdateTelemetry {
    /// Reward computed for this transition
    pub reward: f32,

    /// Previous Q-value (before update)
    pub q_old: f32,

    /// New Q-value (after update)
    pub q_new: f32,

    /// Q-value delta (q_new - q_old)
    pub q_delta: f32,

    /// Learning rate used for update
    pub learning_rate: f32,

    /// Whether this was a terminal state
    pub is_terminal: bool,

    /// Next state index (for debugging)
    pub next_state_index: usize,
}

impl QUpdateTelemetry {
    /// Create from Q-learning update parameters
    pub fn new(
        reward: f32,
        q_old: f32,
        q_new: f32,
        learning_rate: f32,
        is_terminal: bool,
        next_state_index: usize,
    ) -> Self {
        Self {
            reward,
            q_old,
            q_new,
            q_delta: q_new - q_old,
            learning_rate,
            is_terminal,
            next_state_index,
        }
    }
}

/// FluxNet configuration snapshot for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetConfigSnapshot {
    /// Memory tier: "compact" or "extended"
    pub memory_tier: String,

    /// Q-table state space size
    pub qtable_states: usize,

    /// Replay buffer capacity
    pub replay_capacity: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Discount factor (gamma)
    pub discount_factor: f32,

    /// Epsilon start value
    pub epsilon_start: f32,

    /// Epsilon decay rate
    pub epsilon_decay: f32,

    /// Epsilon minimum value
    pub epsilon_min: f32,
}

impl FluxNetConfigSnapshot {
    /// Create from FluxNetConfig
    pub fn from_config(config: &crate::fluxnet::FluxNetConfig) -> Self {
        Self {
            memory_tier: format!("{:?}", config.memory_tier),
            qtable_states: config.rl.get_qtable_states(config.memory_tier),
            replay_capacity: config.rl.get_replay_capacity(config.memory_tier),
            learning_rate: config.rl.learning_rate,
            discount_factor: config.rl.discount_factor,
            epsilon_start: config.rl.epsilon_start,
            epsilon_decay: config.rl.epsilon_decay,
            epsilon_min: config.rl.epsilon_min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_band_telemetry_fractions() {
        let telem = ForceBandTelemetry::from_stats(
            100, // strong
            300, // neutral
            600, // weak
            1000, // total
            1.0, 0.5, 1.5, 0.25,
        );

        assert_eq!(telem.strong_fraction, 0.1);
        assert_eq!(telem.neutral_fraction, 0.3);
        assert_eq!(telem.weak_fraction, 0.6);
    }

    #[test]
    fn test_telemetry_serialization() {
        let force_bands = ForceBandTelemetry {
            strong_fraction: 0.2,
            neutral_fraction: 0.5,
            weak_fraction: 0.3,
            mean_force: 1.0,
            min_force: 0.5,
            max_force: 1.5,
            force_stddev: 0.2,
        };

        let rl_state = RLStateTelemetry {
            phase: "Reservoir".to_string(),
            chromatic_bin: 100,
            conflicts_bin: 50,
            iteration_bin: 128,
            gpu_util_bin: 200,
            phase0_difficulty_quality: 5,
            phase1_te_centrality: 10,
            phase1_ai_uncertainty: 3,
            phase2_temp_band: 8,
            phase2_escape_rate: 12,
            phase3_qubo_quality: 7,
        };

        let rl_decision = RLDecisionTelemetry {
            phase: "Reservoir".to_string(),
            state: rl_state,
            action: FluxNetAction::NoOp,
            q_value: 0.8,
            epsilon: 0.1,
            was_exploration: false,
            state_index_hashed: 42,
            state_index_adaptive: Some(137),
        };

        let config = FluxNetConfigSnapshot {
            memory_tier: "Compact".to_string(),
            qtable_states: 256,
            replay_capacity: 1024,
            learning_rate: 0.001,
            discount_factor: 0.95,
            epsilon_start: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        };

        let telemetry = FluxNetTelemetry::new(
            force_bands,
            rl_decision,
            None,
            config,
        );

        let json = serde_json::to_string(&telemetry).expect("Failed to serialize");
        let _deserialized: FluxNetTelemetry = serde_json::from_str(&json)
            .expect("Failed to deserialize");
    }
}
