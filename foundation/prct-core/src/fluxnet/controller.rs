//! FluxNet RL Controller - Q-Learning Agent
//!
//! Implements tabular Q-learning for adaptive force profile control during
//! Phase 2 thermodynamic equilibration.
//!
//! # Architecture
//!
//! ```text
//! Per Temperature Step:
//! ┌─────────────────┐
//! │  Telemetry      │ (conflicts, colors, compaction)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  RLState        │ Discretize to state index
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  QTable         │ Q(s, a) lookup
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Epsilon-Greedy  │ Explore vs Exploit
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ForceCommand    │ Selected action
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ForceProfile    │ Apply command
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ GPU Kernel      │ Thermodynamic evolution
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Reward          │ Δconflicts, Δcolors
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Q-Update        │ Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
//! └─────────────────┘
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

use super::command::ForceCommand;
use super::config::RLConfig;

/// RL state observation (discretized telemetry)
///
/// Captures key metrics from thermodynamic equilibration:
/// - conflicts: Number of constraint violations
/// - chromatic: Number of colors used
/// - compaction_ratio: Convergence health metric
///
/// State is discretized into bins for tabular Q-learning.
/// Raw values are stored for telemetry reporting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RLState {
    /// Discretized conflict count (0-255)
    pub conflict_bin: u8,

    /// Discretized chromatic number (0-255)
    pub chromatic_bin: u8,

    /// Discretized compaction ratio (0-255)
    pub compaction_bin: u8,

    /// Raw conflict count (for telemetry)
    pub conflicts: usize,

    /// Raw chromatic number (for telemetry)
    pub chromatic_number: usize,

    /// Raw compaction ratio (for telemetry)
    pub compaction_ratio: f32,
}

// Manual PartialEq and Eq implementations that only compare bins (for hashing/Q-table lookup)
impl PartialEq for RLState {
    fn eq(&self, other: &Self) -> bool {
        self.conflict_bin == other.conflict_bin
            && self.chromatic_bin == other.chromatic_bin
            && self.compaction_bin == other.compaction_bin
    }
}

impl Eq for RLState {}

impl std::hash::Hash for RLState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.conflict_bin.hash(state);
        self.chromatic_bin.hash(state);
        self.compaction_bin.hash(state);
    }
}

impl RLState {
    /// Create state from raw telemetry values
    ///
    /// # Arguments
    /// - `conflicts`: Raw conflict count
    /// - `chromatic`: Raw chromatic number
    /// - `compaction_ratio`: Raw compaction ratio [0.0, 1.0]
    /// - `max_conflicts`: Maximum expected conflicts (for normalization)
    /// - `max_chromatic`: Maximum expected colors (for normalization)
    pub fn from_telemetry(
        conflicts: usize,
        chromatic: usize,
        compaction_ratio: f32,
        max_conflicts: usize,
        max_chromatic: usize,
    ) -> Self {
        // Discretize conflicts to 0-255
        let conflict_bin = ((conflicts.min(max_conflicts) as f32 / max_conflicts as f32) * 255.0)
            .min(255.0) as u8;

        // Discretize chromatic to 0-255
        let chromatic_bin =
            ((chromatic.min(max_chromatic) as f32 / max_chromatic as f32) * 255.0).min(255.0)
                as u8;

        // Discretize compaction_ratio [0.0, 1.0] to 0-255
        let compaction_bin = (compaction_ratio.clamp(0.0, 1.0) * 255.0).min(255.0) as u8;

        Self {
            conflict_bin,
            chromatic_bin,
            compaction_bin,
            conflicts,
            chromatic_number: chromatic,
            compaction_ratio,
        }
    }

    /// Convert state to index for Q-table lookup
    ///
    /// Uses bit-packing: [conflict_bin | chromatic_bin | compaction_bin]
    /// State space size: 256 × 256 × 256 = 16,777,216 (full)
    ///
    /// For compact mode, we reduce to top bits only
    pub fn to_index(&self, compact: bool) -> usize {
        if compact {
            // Compact: Use top 4 bits of each (16×16×16 = 4,096 states)
            let c1 = (self.conflict_bin >> 4) as usize;
            let c2 = (self.chromatic_bin >> 4) as usize;
            let c3 = (self.compaction_bin >> 4) as usize;
            (c1 << 8) | (c2 << 4) | c3
        } else {
            // Extended: Use all 8 bits (256×256×256 = 16,777,216 states)
            // Too large for tabular Q, so hash to smaller space
            let c1 = self.conflict_bin as usize;
            let c2 = self.chromatic_bin as usize;
            let c3 = self.compaction_bin as usize;
            ((c1 << 16) | (c2 << 8) | c3) % 1024 // Hash to 1K states
        }
    }
}

impl Default for RLState {
    fn default() -> Self {
        Self {
            conflict_bin: 128,
            chromatic_bin: 128,
            compaction_bin: 128,
        }
    }
}

/// Q-Table for tabular Q-learning
///
/// Stores Q-values Q(s, a) for state-action pairs.
/// Uses simple 2D array: states × actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values: [num_states × num_actions]
    q_values: Vec<Vec<f32>>,

    /// Number of states
    num_states: usize,

    /// Number of actions (always 7 for ForceCommand)
    num_actions: usize,
}

impl QTable {
    /// Create a new Q-table with zeros
    pub fn new(num_states: usize) -> Self {
        let num_actions = ForceCommand::ACTION_SPACE_SIZE;
        let q_values = vec![vec![0.0; num_actions]; num_states];

        Self {
            q_values,
            num_states,
            num_actions,
        }
    }

    /// Get Q-value for state-action pair
    pub fn get(&self, state_idx: usize, action_idx: usize) -> f32 {
        if state_idx < self.num_states && action_idx < self.num_actions {
            self.q_values[state_idx][action_idx]
        } else {
            0.0 // Out of bounds → default to 0
        }
    }

    /// Get Q-value for state-action pair (alias for get, used by telemetry)
    pub fn get_q_value(&self, state_idx: usize, action_idx: usize) -> f32 {
        self.get(state_idx, action_idx)
    }

    /// Set Q-value for state-action pair
    pub fn set(&mut self, state_idx: usize, action_idx: usize, value: f32) {
        if state_idx < self.num_states && action_idx < self.num_actions {
            self.q_values[state_idx][action_idx] = value;
        }
    }

    /// Get best action for state (argmax Q(s, a))
    pub fn best_action(&self, state_idx: usize) -> usize {
        if state_idx >= self.num_states {
            return ForceCommand::NoOp.to_action_index();
        }

        self.q_values[state_idx]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(ForceCommand::NoOp.to_action_index())
    }

    /// Get max Q-value for state (max_a Q(s, a))
    pub fn max_q_value(&self, state_idx: usize) -> f32 {
        if state_idx >= self.num_states {
            return 0.0;
        }

        self.q_values[state_idx]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Update Q-value using Q-learning rule
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    pub fn update(
        &mut self,
        state_idx: usize,
        action_idx: usize,
        reward: f32,
        next_state_idx: usize,
        alpha: f32,
        gamma: f32,
    ) {
        let current_q = self.get(state_idx, action_idx);
        let max_next_q = self.max_q_value(next_state_idx);

        let td_target = reward + gamma * max_next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + alpha * td_error;

        self.set(state_idx, action_idx, new_q);
    }

    /// Save Q-table to binary file
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self)
            .context("Failed to serialize Q-table")?;
        std::fs::write(path, bytes)
            .context("Failed to write Q-table to file")?;
        Ok(())
    }

    /// Load Q-table from binary file
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .context("Failed to read Q-table file")?;
        let qtable: QTable = bincode::deserialize(&bytes)
            .context("Failed to deserialize Q-table")?;
        Ok(qtable)
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: RLState,
    pub action: ForceCommand,
    pub reward: f32,
    pub next_state: RLState,
    pub done: bool,
}

/// Experience replay buffer for off-policy learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    /// Create new replay buffer with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample random batch from buffer
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let buffer_vec: Vec<_> = self.buffer.iter().cloned().collect();

        buffer_vec
            .choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .cloned()
            .collect()
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// FluxNet RL Controller
///
/// Q-learning agent for adaptive force profile control.
pub struct RLController {
    /// Q-table for state-action values
    qtable: QTable,

    /// Experience replay buffer
    replay_buffer: ReplayBuffer,

    /// RL hyperparameters
    config: RLConfig,

    /// Current epsilon (decays over time)
    epsilon: f32,

    /// Compact mode (reduces state space)
    compact: bool,

    /// Maximum conflicts for normalization
    max_conflicts: usize,

    /// Maximum chromatic for normalization
    max_chromatic: usize,
}

impl RLController {
    /// Create new RL controller
    pub fn new(config: RLConfig, num_states: usize, replay_capacity: usize, max_conflicts: usize, max_chromatic: usize, compact: bool) -> Self {
        Self {
            qtable: QTable::new(num_states),
            replay_buffer: ReplayBuffer::new(replay_capacity),
            config,
            epsilon: config.epsilon_start,
            compact,
            max_conflicts,
            max_chromatic,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&mut self, state: &RLState) -> ForceCommand {
        use rand::Rng;

        let state_idx = state.to_index(self.compact);

        // Epsilon-greedy
        if rand::thread_rng().gen::<f32>() < self.epsilon {
            // Explore: random action
            let action_idx = rand::thread_rng().gen_range(0..ForceCommand::ACTION_SPACE_SIZE);
            ForceCommand::from_action_index(action_idx)
        } else {
            // Exploit: best action from Q-table
            let action_idx = self.qtable.best_action(state_idx);
            ForceCommand::from_action_index(action_idx)
        }
    }

    /// Update Q-table from experience
    pub fn update(&mut self, state: RLState, action: ForceCommand, reward: f32, next_state: RLState, done: bool) {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        let next_state_idx = next_state.to_index(self.compact);

        // Store experience in replay buffer
        self.replay_buffer.push(Experience {
            state,
            action,
            reward,
            next_state,
            done,
        });

        // Q-learning update
        self.qtable.update(
            state_idx,
            action_idx,
            reward,
            next_state_idx,
            self.config.learning_rate,
            self.config.discount_factor,
        );

        // Decay epsilon
        self.epsilon *= self.config.epsilon_decay;
        self.epsilon = self.epsilon.max(self.config.epsilon_min);
    }

    /// Compute reward from telemetry delta
    pub fn compute_reward(&self, conflicts_before: usize, conflicts_after: usize, chromatic_before: usize, chromatic_after: usize, compaction_before: f32, compaction_after: f32) -> f32 {
        let delta_conflicts = (conflicts_before as f32 - conflicts_after as f32) / self.max_conflicts as f32;
        let delta_chromatic = (chromatic_before as f32 - chromatic_after as f32) / self.max_chromatic as f32;
        let delta_compaction = compaction_after - compaction_before;

        // Weighted reward
        self.config.reward_conflict_weight * delta_conflicts
            + self.config.reward_color_weight * delta_chromatic
            + self.config.reward_compaction_weight * delta_compaction
    }

    /// Get current epsilon value (for telemetry)
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get Q-value for a specific state-action pair (for telemetry)
    pub fn get_q_value(&self, state: &RLState, action: &ForceCommand) -> f32 {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        self.qtable.get_q_value(state_idx, action_idx)
    }

    /// Select action with telemetry metadata
    ///
    /// Returns (action, q_value, was_exploration)
    pub fn select_action_with_telemetry(&mut self, state: &RLState) -> (ForceCommand, f32, bool) {
        use rand::Rng;

        let state_idx = state.to_index(self.compact);
        let mut rng = rand::thread_rng();

        let was_exploration = rng.gen::<f32>() < self.epsilon;

        let (action, q_value) = if was_exploration {
            // Explore: random action
            let action_idx = rng.gen_range(0..ForceCommand::ACTION_SPACE_SIZE);
            let action = ForceCommand::from_action_index(action_idx);
            let q_val = self.qtable.get_q_value(state_idx, action_idx);
            (action, q_val)
        } else {
            // Exploit: best action from Q-table
            let action_idx = self.qtable.best_action(state_idx);
            let action = ForceCommand::from_action_index(action_idx);
            let q_val = self.qtable.get_q_value(state_idx, action_idx);
            (action, q_val)
        };

        (action, q_value, was_exploration)
    }

    /// Update with telemetry capture
    ///
    /// Returns (q_old, q_new, q_delta) for telemetry reporting
    pub fn update_with_telemetry(
        &mut self,
        state: RLState,
        action: ForceCommand,
        reward: f32,
        next_state: RLState,
        done: bool,
    ) -> (f32, f32, f32) {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        let next_state_idx = next_state.to_index(self.compact);

        // Get Q-value before update
        let q_old = self.qtable.get_q_value(state_idx, action_idx);

        // Store experience in replay buffer
        self.replay_buffer.push(Experience {
            state,
            action,
            reward,
            next_state,
            done,
        });

        // Q-learning update
        self.qtable.update(
            state_idx,
            action_idx,
            reward,
            next_state_idx,
            self.config.learning_rate,
            self.config.discount_factor,
        );

        // Get Q-value after update
        let q_new = self.qtable.get_q_value(state_idx, action_idx);
        let q_delta = q_new - q_old;

        // Decay epsilon
        self.epsilon *= self.config.epsilon_decay;
        self.epsilon = self.epsilon.max(self.config.epsilon_min);

        (q_old, q_new, q_delta)
    }

    /// Load pre-trained Q-table
    pub fn load_qtable(&mut self, path: &Path) -> Result<()> {
        self.qtable = QTable::load(path)?;
        Ok(())
    }

    /// Save Q-table
    pub fn save_qtable(&self, path: &Path) -> Result<()> {
        self.qtable.save(path)?;
        Ok(())
    }

    /// Get current epsilon
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get current replay buffer size
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_state_discretization() {
        let state = RLState::from_telemetry(50, 100, 0.75, 100, 200);
        assert_eq!(state.conflict_bin, 127); // 50/100 * 255 ≈ 127
        assert_eq!(state.chromatic_bin, 127); // 100/200 * 255 ≈ 127
        assert_eq!(state.compaction_bin, 191); // 0.75 * 255 ≈ 191
    }

    #[test]
    fn test_qtable_operations() {
        let mut qtable = QTable::new(256);

        qtable.set(0, 0, 1.5);
        assert_eq!(qtable.get(0, 0), 1.5);

        qtable.set(0, 1, 2.5);
        assert_eq!(qtable.best_action(0), 1);
        assert_eq!(qtable.max_q_value(0), 2.5);
    }

    #[test]
    fn test_q_learning_update() {
        let mut qtable = QTable::new(256);

        // Initial Q(s,a) = 0
        qtable.update(0, 0, 1.0, 1, 0.1, 0.9);

        // Q(s,a) should increase toward reward
        assert!(qtable.get(0, 0) > 0.0);
        assert!(qtable.get(0, 0) < 1.0); // Not fully converged yet
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..15 {
            buffer.push(Experience {
                state: RLState::default(),
                action: ForceCommand::NoOp,
                reward: i as f32,
                next_state: RLState::default(),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 10); // Capped at capacity
        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_epsilon_decay() {
        let config = RLConfig {
            epsilon_start: 0.3,
            epsilon_decay: 0.99,
            epsilon_min: 0.05,
            ..Default::default()
        };

        let mut controller = RLController::new(config, 256, 1024, 100, 200, true);

        assert_eq!(controller.epsilon(), 0.3);

        // Simulate 100 updates
        for _ in 0..100 {
            controller.update(
                RLState::default(),
                ForceCommand::NoOp,
                0.0,
                RLState::default(),
                false,
            );
        }

        // Epsilon should decay but not below min
        assert!(controller.epsilon() < 0.3);
        assert!(controller.epsilon() >= 0.05);
    }
}
