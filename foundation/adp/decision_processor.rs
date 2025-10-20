//! Adaptive Decision Processor
//!
//! High-level decision making for PRCT parameter optimization
//! combining reinforcement learning with performance feedback.

use super::reinforcement::{Action, ReinforcementLearner, RlConfig, State};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Decision made by ADP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Unique decision ID
    pub id: String,

    /// Selected action
    pub action: Action,

    /// Confidence score [0, 1]
    pub confidence: f64,

    /// Human-readable reasoning
    pub reasoning: String,

    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

/// Adaptive decision processor combining RL with performance tracking
pub struct AdaptiveDecisionProcessor {
    /// Q-learning agent
    rl_learner: ReinforcementLearner,

    /// Recent decision history for learning
    decision_history: Vec<(State, Action, u64)>, // (state, action, timestamp)

    /// Performance metrics history
    performance_history: Vec<f64>,

    /// Total decisions made
    decisions_made: u64,

    /// Feature discretization resolution
    resolution: f64,
}

impl AdaptiveDecisionProcessor {
    /// Create new adaptive decision processor
    pub fn new(config: RlConfig) -> Self {
        Self {
            rl_learner: ReinforcementLearner::new(config),
            decision_history: Vec::new(),
            performance_history: Vec::new(),
            decisions_made: 0,
            resolution: 0.1, // Discretization resolution for state
        }
    }

    /// Make decision based on current system features
    ///
    /// # Arguments
    /// * `features` - Current performance features:
    ///   [pattern_strength, coherence, energy, processing_time, ...]
    ///
    /// # Returns
    /// Decision with action, confidence, and reasoning
    pub fn make_decision(&mut self, features: &[f64]) -> Result<Decision> {
        // Convert features to discrete state
        let state = State::from_features(features, self.resolution);

        // Select action using ε-greedy policy
        let action = self.rl_learner.select_action(&state);

        // Calculate confidence based on Q-value variance
        let confidence = self.calculate_confidence(&state, action);

        // Generate reasoning
        let reasoning = self.generate_reasoning(&state, action, features);

        // Create decision
        let decision = Decision {
            id: format!("adp-{:08x}", self.decisions_made),
            action,
            confidence,
            reasoning,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        // Record decision
        self.decision_history
            .push((state, action, decision.timestamp_ns));
        self.decisions_made += 1;

        // Cleanup old history
        if self.decision_history.len() > 1000 {
            self.decision_history.drain(0..500);
        }

        Ok(decision)
    }

    /// Learn from performance feedback
    ///
    /// # Arguments
    /// * `performance` - Performance metric (higher is better)
    ///   Examples: solution quality, speedup, convergence rate
    pub fn learn_from_feedback(&mut self, performance: f64) -> Result<()> {
        // Store performance
        self.performance_history.push(performance);

        // Need at least 2 decisions to compute reward
        if self.decision_history.len() < 2 {
            return Ok(());
        }

        // Get previous and current decisions
        let (prev_state, prev_action, _) =
            self.decision_history[self.decision_history.len() - 2].clone();
        let (current_state, _, _) = self.decision_history[self.decision_history.len() - 1].clone();

        // Calculate reward as performance improvement
        let prev_performance = if self.performance_history.len() >= 2 {
            self.performance_history[self.performance_history.len() - 2]
        } else {
            0.0
        };

        let reward = performance - prev_performance;

        // Update Q-values
        self.rl_learner
            .update(prev_state, prev_action, reward, current_state);

        Ok(())
    }

    /// End episode (decay exploration)
    pub fn end_episode(&self) {
        self.rl_learner.end_episode();
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> AdpStats {
        let rl_stats = self.rl_learner.get_stats();

        AdpStats {
            decisions_made: self.decisions_made,
            episodes: rl_stats.episodes,
            updates: rl_stats.updates,
            epsilon: rl_stats.epsilon,
            q_table_size: rl_stats.q_table_size,
            avg_performance: if self.performance_history.is_empty() {
                0.0
            } else {
                self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
            },
        }
    }

    /// Calculate decision confidence
    fn calculate_confidence(&self, state: &State, action: Action) -> f64 {
        let q_value = self.rl_learner.get_q_value(state, action);

        // Get Q-values for all actions
        let all_actions = Action::all();
        let q_values: Vec<f64> = all_actions
            .iter()
            .map(|&a| self.rl_learner.get_q_value(state, a))
            .collect();

        // Confidence based on how much better this action is
        let max_q = q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_q = q_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if (max_q - min_q).abs() < 1e-6 {
            0.5 // Uncertain - all actions equally valued
        } else {
            ((q_value - min_q) / (max_q - min_q)).clamp(0.0, 1.0)
        }
    }

    /// Generate human-readable reasoning
    fn generate_reasoning(&self, state: &State, action: Action, features: &[f64]) -> String {
        let feature_names = vec!["pattern_strength", "coherence", "energy", "processing_time"];

        let mut reasoning = format!("Action: {:?}\n", action);
        reasoning.push_str("Context:\n");

        for (i, &value) in features.iter().enumerate().take(feature_names.len()) {
            let name = feature_names.get(i).unwrap_or(&"feature");
            reasoning.push_str(&format!("  {}: {:.3}\n", name, value));
        }

        let epsilon = *self.rl_learner.epsilon.read();
        if epsilon > 0.5 {
            reasoning.push_str("Strategy: Exploration (learning)\n");
        } else {
            reasoning.push_str("Strategy: Exploitation (optimized)\n");
        }

        reasoning
    }

    /// Reset learning (start fresh)
    pub fn reset(&mut self) {
        self.rl_learner = ReinforcementLearner::new(self.rl_learner.config.clone());
        self.decision_history.clear();
        self.performance_history.clear();
        self.decisions_made = 0;
    }

    /// Save learned policy
    pub fn save_policy(&self, path: &str) -> Result<()> {
        self.rl_learner.save_q_table(path)
    }

    /// Load learned policy
    pub fn load_policy(&self, path: &str) -> Result<()> {
        self.rl_learner.load_q_table(path)
    }
}

/// ADP statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdpStats {
    pub decisions_made: u64,
    pub episodes: u64,
    pub updates: u64,
    pub epsilon: f64,
    pub q_table_size: usize,
    pub avg_performance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_making() {
        let config = RlConfig::default();
        let mut adp = AdaptiveDecisionProcessor::new(config);

        let features = vec![0.8, 0.9, 50.0, 10.0]; // Good performance
        let decision = adp.make_decision(&features).unwrap();

        assert!(!decision.id.is_empty());
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    }

    #[test]
    fn test_learning_cycle() {
        let config = RlConfig::default();
        let mut adp = AdaptiveDecisionProcessor::new(config);

        // Simulate multiple decision-feedback cycles
        for i in 0..10 {
            let features = vec![0.5 + i as f64 * 0.01, 0.7, 50.0, 10.0];
            let _decision = adp.make_decision(&features).unwrap();

            // Provide positive feedback
            adp.learn_from_feedback(0.1).unwrap();
        }

        let stats = adp.get_stats();
        assert!(stats.decisions_made == 10);
        assert!(stats.updates > 0); // Should have learned
    }

    #[test]
    fn test_epsilon_decay() {
        let config = RlConfig::default();
        let mut adp = AdaptiveDecisionProcessor::new(config);

        let initial_epsilon = adp.get_stats().epsilon;

        // Make many decisions to trigger decay
        for _ in 0..50 {
            let features = vec![0.5, 0.7, 50.0];
            let _decision = adp.make_decision(&features).unwrap();
            adp.learn_from_feedback(0.1).unwrap();
            adp.end_episode();
        }

        let final_epsilon = adp.get_stats().epsilon;
        assert!(final_epsilon < initial_epsilon);
    }
}
