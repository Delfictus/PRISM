//! Reinforcement Learning for PRCT Parameter Optimization
//!
//! Q-learning implementation for adaptive parameter tuning based on
//! optimization performance feedback. Ported from CSF's ADP module.

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Reinforcement learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    /// Learning rate (how fast to update Q-values)
    pub alpha: f64,

    /// Discount factor (how much to value future rewards)
    pub gamma: f64,

    /// Initial exploration rate
    pub epsilon: f64,

    /// Epsilon decay per episode
    pub epsilon_decay: f64,

    /// Minimum exploration rate
    pub epsilon_min: f64,

    /// Q-table cleanup threshold
    pub cleanup_threshold: usize,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            alpha: 0.001,              // Small learning rate for stability
            gamma: 0.95,               // High discount for long-term planning
            epsilon: 1.0,              // Start with full exploration
            epsilon_decay: 0.995,      // Gradual shift to exploitation
            epsilon_min: 0.01,         // Always maintain 1% exploration
            cleanup_threshold: 10_000, // Prevent Q-table from growing unbounded
        }
    }
}

/// State representation for Q-learning
///
/// Discretized features representing current PRCT performance state
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// Discretized feature vector
    pub features: Vec<i32>,
}

impl State {
    /// Create state from continuous features
    pub fn from_features(features: &[f64], resolution: f64) -> Self {
        Self {
            features: features.iter().map(|&x| (x / resolution) as i32).collect(),
        }
    }

    /// Hamming distance between states (for similarity)
    pub fn distance(&self, other: &State) -> usize {
        self.features
            .iter()
            .zip(&other.features)
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// Actions for PRCT parameter adjustment
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Increase coupling strength
    IncreaseCoupling,

    /// Decrease coupling strength
    DecreaseCoupling,

    /// Increase evolution time
    IncreaseEvolutionTime,

    /// Decrease evolution time
    DecreaseEvolutionTime,

    /// Increase neuron count
    IncreaseNeuronCount,

    /// Decrease neuron count
    DecreaseNeuronCount,

    /// Increase noise level (exploration)
    IncreaseNoise,

    /// Decrease noise level (exploitation)
    DecreaseNoise,

    /// Keep current parameters
    MaintainCurrent,
}

impl Action {
    /// Get all possible actions
    pub fn all() -> Vec<Action> {
        vec![
            Action::IncreaseCoupling,
            Action::DecreaseCoupling,
            Action::IncreaseEvolutionTime,
            Action::DecreaseEvolutionTime,
            Action::IncreaseNeuronCount,
            Action::DecreaseNeuronCount,
            Action::IncreaseNoise,
            Action::DecreaseNoise,
            Action::MaintainCurrent,
        ]
    }

    /// Parameter adjustment magnitude
    pub fn adjustment_factor(&self) -> f64 {
        match self {
            Action::IncreaseCoupling
            | Action::IncreaseEvolutionTime
            | Action::IncreaseNeuronCount
            | Action::IncreaseNoise => 1.1,

            Action::DecreaseCoupling
            | Action::DecreaseEvolutionTime
            | Action::DecreaseNeuronCount
            | Action::DecreaseNoise => 0.9,

            Action::MaintainCurrent => 1.0,
        }
    }
}

/// Q-learning based reinforcement learner
pub struct ReinforcementLearner {
    pub config: RlConfig,

    /// Q-table: State → Action → Expected reward
    q_table: Arc<RwLock<HashMap<State, HashMap<Action, f64>>>>,

    /// Current exploration rate
    pub epsilon: Arc<RwLock<f64>>,

    /// Episode count for tracking
    episodes: Arc<RwLock<u64>>,

    /// Total updates performed
    updates: Arc<RwLock<u64>>,
}

impl ReinforcementLearner {
    /// Create new reinforcement learner
    pub fn new(config: RlConfig) -> Self {
        Self {
            epsilon: Arc::new(RwLock::new(config.epsilon)),
            config,
            q_table: Arc::new(RwLock::new(HashMap::new())),
            episodes: Arc::new(RwLock::new(0)),
            updates: Arc::new(RwLock::new(0)),
        }
    }

    /// Select action using ε-greedy policy
    pub fn select_action(&self, state: &State) -> Action {
        let epsilon = *self.epsilon.read();

        // Exploration: random action
        if rand::random::<f64>() < epsilon {
            use rand::seq::SliceRandom;
            *Action::all().choose(&mut rand::thread_rng()).unwrap()
        } else {
            // Exploitation: best known action
            let q_table = self.q_table.read();

            if let Some(actions) = q_table.get(state) {
                actions
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(action, _)| *action)
                    .unwrap_or(Action::MaintainCurrent)
            } else {
                // Unknown state: explore
                Action::MaintainCurrent
            }
        }
    }

    /// Update Q-value using Bellman equation
    pub fn update(&mut self, state: State, action: Action, reward: f64, next_state: State) {
        let mut q_table = self.q_table.write();

        // Find max Q-value for next state first (before mutable borrow)
        let q_next_max = q_table
            .get(&next_state)
            .and_then(|actions| {
                actions
                    .values()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            })
            .unwrap_or(0.0);

        // Initialize state-action if needed and get current Q-value
        let state_actions = q_table.entry(state.clone()).or_insert_with(HashMap::new);
        let q_current = *state_actions.entry(action).or_insert(0.0);

        // Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        let q_new =
            q_current + self.config.alpha * (reward + self.config.gamma * q_next_max - q_current);

        // Update Q-table
        state_actions.insert(action, q_new);

        // Increment update counter
        *self.updates.write() += 1;

        // Cleanup if Q-table too large
        let should_cleanup = q_table.len() > self.config.cleanup_threshold;
        if should_cleanup {
            self.cleanup_q_table(&mut q_table);
        }
    }

    /// Decay exploration rate
    pub fn decay_epsilon(&self) {
        let mut epsilon = self.epsilon.write();
        *epsilon = (*epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// End episode and decay exploration
    pub fn end_episode(&self) {
        *self.episodes.write() += 1;
        self.decay_epsilon();
    }

    /// Get current Q-value for state-action pair
    pub fn get_q_value(&self, state: &State, action: Action) -> f64 {
        let q_table = self.q_table.read();
        q_table
            .get(state)
            .and_then(|actions| actions.get(&action))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get best action for state (greedy)
    pub fn get_best_action(&self, state: &State) -> Option<Action> {
        let q_table = self.q_table.read();
        q_table.get(state).and_then(|actions| {
            actions
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(action, _)| *action)
        })
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> RlStats {
        RlStats {
            episodes: *self.episodes.read(),
            updates: *self.updates.read(),
            epsilon: *self.epsilon.read(),
            q_table_size: self.q_table.read().len(),
        }
    }

    /// Cleanup Q-table by removing low-value state-action pairs
    fn cleanup_q_table(&self, q_table: &mut HashMap<State, HashMap<Action, f64>>) {
        // Remove states with all low Q-values
        q_table.retain(|_, actions| {
            let max_q = actions
                .values()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(0.0);

            max_q.abs() > 0.01 // Keep if any action has meaningful value
        });
    }

    /// Save Q-table to file
    pub fn save_q_table(&self, path: &str) -> Result<()> {
        let q_table = self.q_table.read();
        let json = serde_json::to_string_pretty(&*q_table)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load Q-table from file
    pub fn load_q_table(&self, path: &str) -> Result<()> {
        let json = std::fs::read_to_string(path)?;
        let loaded: HashMap<State, HashMap<Action, f64>> = serde_json::from_str(&json)?;
        *self.q_table.write() = loaded;
        Ok(())
    }
}

/// Reinforcement learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStats {
    pub episodes: u64,
    pub updates: u64,
    pub epsilon: f64,
    pub q_table_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let features = vec![0.5, 1.2, 0.8];
        let state = State::from_features(&features, 0.1);
        assert_eq!(state.features, vec![5, 12, 8]);
    }

    #[test]
    fn test_q_learning_update() {
        let config = RlConfig::default();
        let mut learner = ReinforcementLearner::new(config);

        let state = State {
            features: vec![1, 2, 3],
        };
        let next_state = State {
            features: vec![1, 2, 4],
        };

        // Positive reward should increase Q-value
        learner.update(
            state.clone(),
            Action::IncreaseCoupling,
            1.0,
            next_state.clone(),
        );

        let q = learner.get_q_value(&state, Action::IncreaseCoupling);
        assert!(q > 0.0);
    }

    #[test]
    fn test_epsilon_decay() {
        let config = RlConfig::default();
        let learner = ReinforcementLearner::new(config);

        let initial_epsilon = *learner.epsilon.read();
        assert_eq!(initial_epsilon, 1.0);

        for _ in 0..100 {
            learner.decay_epsilon();
        }

        let final_epsilon = *learner.epsilon.read();
        assert!(final_epsilon < initial_epsilon);
        assert!(final_epsilon >= learner.config.epsilon_min);
    }

    #[test]
    fn test_action_selection() {
        let config = RlConfig {
            epsilon: 0.0,
            ..Default::default()
        }; // Pure exploitation
        let learner = ReinforcementLearner::new(config);

        let state = State {
            features: vec![1, 2],
        };

        // Should select action deterministically when epsilon=0
        let action1 = learner.select_action(&state);
        let action2 = learner.select_action(&state);

        // Both should be same (greedy)
        assert_eq!(action1, action2);
    }
}
