//! FluxNet Multi-Phase RL Controller - Q-Learning Agent
//!
//! Implements tabular Q-learning for adaptive parameter control across ALL 4 phases:
//! - Phase 0 (Reservoir): Spectral radius, leak rate, input scaling
//! - Phase 1 (Transfer Entropy): TE weights, geodesic weights, batch sizes
//! - Phase 2 (Thermodynamic): Forces, temps, steps, replicas
//! - Phase 3 (Quantum): Iterations, beta, temperature, QUBO settings
//!
//! # Architecture
//!
//! ```text
//! Per Phase Completion:
//! ┌─────────────────┐
//! │  Telemetry      │ (chromatic, conflicts, gpu_util, phase_metrics)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  UnifiedRLState │ Discretize to state index
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  QTable         │ Q(s, a) lookup (37 actions)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Epsilon-Greedy  │ Explore vs Exploit
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ FluxNetAction   │ Selected action (phase-filtered)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ WorldRecordConfig│ Apply action (modify parameters)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Next Phase      │ Execute with updated config
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Reward          │ Δchromatic, Δconflicts
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Q-Update        │ Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
//! └─────────────────┘
//! ```
//!
//! # GPU-First Compliance
//!
//! - ✅ All state/reward metrics from GPU telemetry
//! - ✅ Actions modify GPU kernel parameters only
//! - ✅ No CPU-specific Q-table updates
//! - ❌ NO fallback to CPU RL

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use crate::errors::*;

use super::command::FluxNetAction;
use super::config::RLConfig;
use super::unified_state::UnifiedRLState;
use super::reward::compute_reward;
use super::adaptive_index::AdaptiveStateIndexer;
use crate::telemetry::PhaseName;

/// Q-Table for tabular Q-learning
///
/// Stores Q-values Q(s, a) for state-action pairs.
/// Uses HashMap for sparse representation (only visited states stored).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values: (state_idx, action_idx) -> Q-value
    q_values: HashMap<(usize, usize), f64>,

    /// Table size (max state index)
    table_size: usize,

    /// Number of actions
    num_actions: usize,

    /// Visit counts: (state_idx, action_idx) -> count
    visit_counts: HashMap<(usize, usize), usize>,
}

impl QTable {
    /// Create a new Q-table with specified size
    pub fn new(table_size: usize) -> Self {
        Self {
            q_values: HashMap::new(),
            table_size,
            num_actions: FluxNetAction::ACTION_SPACE_SIZE,
            visit_counts: HashMap::new(),
        }
    }

    /// Get Q-value for state-action pair (default 0.0 if unvisited)
    pub fn get(&self, state_idx: usize, action_idx: usize) -> f64 {
        self.q_values
            .get(&(state_idx, action_idx))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get Q-value for state-action pair (alias for get, used by telemetry)
    pub fn get_q_value(&self, state_idx: usize, action_idx: usize) -> f32 {
        self.get(state_idx, action_idx) as f32
    }

    /// Set Q-value for state-action pair
    pub fn set(&mut self, state_idx: usize, action_idx: usize, value: f64) {
        if state_idx < self.table_size && action_idx < self.num_actions {
            self.q_values.insert((state_idx, action_idx), value);
            *self.visit_counts.entry((state_idx, action_idx)).or_insert(0) += 1;
        }
    }

    /// Get best action for state (argmax Q(s, a))
    pub fn best_action(&self, state_idx: usize) -> usize {
        if state_idx >= self.table_size {
            return FluxNetAction::NoOp.to_index();
        }

        (0..self.num_actions)
            .max_by(|&a, &b| {
                let qa = self.get(state_idx, a);
                let qb = self.get(state_idx, b);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(FluxNetAction::NoOp.to_index())
    }

    /// Get max Q-value for state (max_a Q(s, a))
    pub fn max_q_value(&self, state_idx: usize) -> f64 {
        if state_idx >= self.table_size {
            return 0.0;
        }

        (0..self.num_actions)
            .map(|a| self.get(state_idx, a))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0) // Default to 0 if no actions visited
    }

    /// Update Q-value using Q-learning rule
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    pub fn update(
        &mut self,
        state_idx: usize,
        action_idx: usize,
        reward: f64,
        next_state_idx: usize,
        alpha: f64,
        gamma: f64,
    ) {
        let current_q = self.get(state_idx, action_idx);
        let max_next_q = self.max_q_value(next_state_idx);

        let td_target = reward + gamma * max_next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + alpha * td_error;

        self.set(state_idx, action_idx, new_q);
    }

    /// Get visit count for state-action pair
    pub fn visit_count(&self, state_idx: usize, action_idx: usize) -> usize {
        self.visit_counts
            .get(&(state_idx, action_idx))
            .copied()
            .unwrap_or(0)
    }

    /// Get total number of visited state-action pairs
    pub fn num_visited(&self) -> usize {
        self.q_values.len()
    }

    /// Save Q-table to binary file
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to serialize Q-table: {}", e))
        })?;
        std::fs::write(path, bytes).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to write Q-table file: {}", e))
        })?;
        Ok(())
    }

    /// Load Q-table from binary file
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to read Q-table file: {}", e))
        })?;
        let qtable: QTable = bincode::deserialize(&bytes).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to deserialize Q-table: {}", e))
        })?;
        Ok(qtable)
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: UnifiedRLState,
    pub action: FluxNetAction,
    pub reward: f64,
    pub next_state: UnifiedRLState,
    pub done: bool,
}

/// Experience replay buffer with prioritized sampling
///
/// Supports both uniform sampling (legacy) and priority-based sampling.
/// Priorities are stored alongside experiences and updated after Q-learning updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Experience buffer (FIFO for eviction)
    buffer: VecDeque<Experience>,

    /// Priority buffer (parallel to experience buffer)
    priorities: VecDeque<f64>,

    /// Buffer capacity
    capacity: usize,

    /// Running maximum priority (for new experiences)
    max_priority: f64,

    /// Sum of all priorities (for normalization)
    priority_sum: f64,
}

impl ReplayBuffer {
    /// Create new replay buffer with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            max_priority: 1.0, // Default priority for first experience
            priority_sum: 0.0,
        }
    }

    /// Add experience to buffer with default priority
    pub fn push(&mut self, experience: Experience) {
        self.push_priority(experience, self.max_priority);
    }

    /// Add experience to buffer with explicit priority
    ///
    /// Evicts oldest experience if buffer is full (FIFO policy)
    pub fn push_priority(&mut self, experience: Experience, priority: f64) {
        let priority = priority.max(0.0); // Ensure non-negative

        if self.buffer.len() >= self.capacity {
            // Evict oldest
            self.buffer.pop_front();
            if let Some(old_priority) = self.priorities.pop_front() {
                self.priority_sum -= old_priority;
            }
        }

        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
        self.priority_sum += priority;

        // Track max priority for new experiences
        if priority > self.max_priority {
            self.max_priority = priority;
        }
    }

    /// Sample uniform random batch from buffer (legacy)
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

    /// Sample weighted batch from buffer based on priorities
    ///
    /// Uses priority-proportional sampling: P(i) = priority[i] / sum(priorities)
    ///
    /// # Arguments
    /// - `batch_size`: Number of samples to draw
    /// - `priority_eps`: Small constant for numerical stability (default 0.01)
    ///
    /// # Returns
    /// (experiences, indices, importance_weights)
    pub fn sample_weighted(
        &self,
        batch_size: usize,
        priority_eps: f64,
        beta: f64,
    ) -> (Vec<Experience>, Vec<usize>, Vec<f64>) {
        use rand::Rng;

        if self.buffer.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let batch_size = batch_size.min(self.buffer.len());
        let mut rng = rand::thread_rng();

        // Normalize priorities to probabilities
        let total_priority = self.priority_sum.max(priority_eps);
        let mut probabilities: Vec<f64> = self
            .priorities
            .iter()
            .map(|&p| (p + priority_eps) / total_priority)
            .collect();

        // Compute importance sampling weights: w_i = (N * P(i))^(-beta)
        let n = self.buffer.len() as f64;
        let max_weight = (n * probabilities.iter().cloned().fold(f64::INFINITY, f64::min)).powf(-beta);

        let mut samples = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Sample index based on priority
            let rand_val: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut sampled_idx = 0;

            for (i, &prob) in probabilities.iter().enumerate() {
                cumsum += prob;
                if rand_val <= cumsum {
                    sampled_idx = i;
                    break;
                }
            }

            // Compute importance sampling weight
            let prob = probabilities[sampled_idx];
            let weight = ((n * prob).powf(-beta) / max_weight).min(1.0);

            samples.push(self.buffer[sampled_idx].clone());
            indices.push(sampled_idx);
            weights.push(weight);

            // Zero out probability to avoid resampling (without replacement)
            probabilities[sampled_idx] = 0.0;
            let new_total: f64 = probabilities.iter().sum();
            if new_total > 0.0 {
                for p in &mut probabilities {
                    *p /= new_total;
                }
            } else {
                break; // All probabilities exhausted
            }
        }

        (samples, indices, weights)
    }

    /// Update priority for a specific index
    ///
    /// Called after Q-learning update to adjust priority based on TD-error
    pub fn update_priority(&mut self, index: usize, new_priority: f64) {
        if index < self.priorities.len() {
            let old_priority = self.priorities[index];
            self.priorities[index] = new_priority.max(0.0);
            self.priority_sum += new_priority - old_priority;

            if new_priority > self.max_priority {
                self.max_priority = new_priority;
            }
        }
    }

    /// Get priority statistics for observability
    pub fn priority_stats(&self) -> ReplayBufferStats {
        if self.priorities.is_empty() {
            return ReplayBufferStats {
                mean_priority: 0.0,
                max_priority: 0.0,
                min_priority: 0.0,
                total_priority: 0.0,
            };
        }

        let mean = self.priority_sum / self.priorities.len() as f64;
        let max = self.priorities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = self.priorities.iter().cloned().fold(f64::INFINITY, f64::min);

        ReplayBufferStats {
            mean_priority: mean,
            max_priority: max,
            min_priority: min,
            total_priority: self.priority_sum,
        }
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.priorities.clear();
        self.priority_sum = 0.0;
    }
}

/// Replay buffer priority statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBufferStats {
    pub mean_priority: f64,
    pub max_priority: f64,
    pub min_priority: f64,
    pub total_priority: f64,
}

/// Multi-Phase RL Controller
///
/// Q-learning agent for adaptive parameter control across all 4 phases.
/// Replaces single-phase RLController with unified multi-phase control.
pub struct MultiPhaseRLController {
    /// Q-table for state-action values
    qtable: QTable,

    /// Experience replay buffer
    replay_buffer: ReplayBuffer,

    /// Adaptive state indexer (percentile-based quantization)
    adaptive_indexer: AdaptiveStateIndexer,

    /// RL hyperparameters
    config: RLConfig,

    /// Current epsilon (decays over time)
    epsilon: f64,

    /// Table size for state indexing
    table_size: usize,

    /// Maximum chromatic number (for normalization)
    max_chromatic: usize,

    /// Phase-specific reward tracking
    phase_rewards: HashMap<PhaseName, VecDeque<f64>>,

    /// Action frequency tracking (phase, action_idx) -> count
    action_counts: HashMap<(PhaseName, usize), usize>,

    /// Total updates performed
    total_updates: usize,

    /// Verbose logging
    verbose: bool,
}

impl MultiPhaseRLController {
    /// Create new multi-phase RL controller
    ///
    /// # Arguments
    /// - `config`: RL hyperparameters
    /// - `table_size`: Q-table state space size (typically 8192)
    /// - `replay_capacity`: Experience replay buffer capacity
    /// - `max_chromatic`: Maximum expected chromatic number (for normalization)
    /// - `verbose`: Enable verbose logging
    pub fn new(
        config: RLConfig,
        table_size: usize,
        replay_capacity: usize,
        max_chromatic: usize,
        verbose: bool,
    ) -> Self {
        Self {
            qtable: QTable::new(table_size),
            replay_buffer: ReplayBuffer::new(replay_capacity),
            adaptive_indexer: AdaptiveStateIndexer::new(),
            epsilon: config.epsilon_start as f64,
            config,
            table_size,
            max_chromatic,
            phase_rewards: HashMap::new(),
            action_counts: HashMap::new(),
            total_updates: 0,
            verbose,
        }
    }

    /// Select action using epsilon-greedy policy
    ///
    /// # Arguments
    /// - `state`: Current state
    /// - `phase`: Current phase (for action filtering)
    ///
    /// # Returns
    /// Selected action
    pub fn choose_action(&mut self, state: &UnifiedRLState, phase: PhaseName) -> FluxNetAction {
        use rand::Rng;

        // Use adaptive indexing if enabled and ready, otherwise fall back to hash-based
        let state_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
            state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
        } else {
            state.to_index(self.table_size)
        };

        // Get valid actions for this phase
        let valid_actions = FluxNetAction::actions_for_phase(phase);
        if valid_actions.is_empty() {
            return FluxNetAction::NoOp;
        }

        // Epsilon-greedy
        if rand::thread_rng().gen::<f64>() < self.epsilon {
            // Explore: random action from valid set
            let idx = rand::thread_rng().gen_range(0..valid_actions.len());
            valid_actions[idx]
        } else {
            // Exploit: best action from Q-table among valid actions
            let best_action_idx = valid_actions
                .iter()
                .map(|a| (a.to_index(), self.qtable.get(state_idx, a.to_index())))
                .max_by(|(_, qa), (_, qb)| qa.partial_cmp(qb).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(FluxNetAction::NoOp.to_index());

            FluxNetAction::from_index(best_action_idx)
        }
    }

    /// Update Q-table from experience
    ///
    /// # Arguments
    /// - `state`: Previous state
    /// - `action`: Action taken
    /// - `reward`: Observed reward
    /// - `next_state`: Resulting state
    /// - `done`: Episode terminated
    pub fn update(
        &mut self,
        state: UnifiedRLState,
        action: FluxNetAction,
        reward: f64,
        next_state: UnifiedRLState,
        done: bool,
    ) {
        // Observe state features for adaptive indexer (always, for histogram building)
        self.adaptive_indexer.observe(
            state.chromatic_bin,
            state.conflicts_bin,
            state.iteration_bin,
            state.gpu_util_bin,
            [
                state.phase0_difficulty_quality,
                state.phase1_te_centrality,
                state.phase1_ai_uncertainty,
                state.phase2_temp_band,
                state.phase2_escape_rate,
                state.phase3_qubo_quality,
            ],
        );

        // Use adaptive indexing if enabled and ready, otherwise fall back to hash-based
        let state_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
            state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
        } else {
            state.to_index(self.table_size)
        };

        let action_idx = action.to_index();

        let next_state_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
            next_state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
        } else {
            next_state.to_index(self.table_size)
        };

        // Compute experience priority based on reward magnitude
        // priority = |reward|^alpha, with 2x boost for high rewards
        let priority = if reward.abs() > self.config.high_reward_cutoff as f64 {
            2.0 * reward.abs().max(1.0).powf(self.config.priority_alpha as f64)
        } else {
            reward.abs().max(1.0).powf(self.config.priority_alpha as f64)
        };

        // Store experience in replay buffer with priority
        self.replay_buffer.push_priority(
            Experience {
                state,
                action,
                reward,
                next_state,
                done,
            },
            priority,
        );

        // Q-learning update
        let gamma = if done {
            0.0 // No future reward if episode ended
        } else {
            self.config.discount_factor as f64
        };

        self.qtable.update(
            state_idx,
            action_idx,
            reward,
            next_state_idx,
            self.config.learning_rate as f64,
            gamma,
        );

        // Track metrics
        self.phase_rewards
            .entry(state.phase())
            .or_insert_with(|| VecDeque::with_capacity(100))
            .push_back(reward);

        *self
            .action_counts
            .entry((state.phase(), action_idx))
            .or_insert(0) += 1;

        self.total_updates += 1;

        // Decay epsilon
        self.epsilon *= self.config.epsilon_decay as f64;
        self.epsilon = self.epsilon.max(self.config.epsilon_min as f64);

        if self.verbose {
            let adaptive_status = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
                "adaptive"
            } else {
                "hashed"
            };
            println!(
                "[FLUXNET] Update #{}: Q({},{}) -> {:.2}, reward={:.2}, priority={:.2}, ε={:.3} [{}]",
                self.total_updates,
                state_idx,
                action_idx,
                self.qtable.get(state_idx, action_idx),
                reward,
                priority,
                self.epsilon,
                adaptive_status
            );
        }
    }

    /// Load pre-trained Q-table
    pub fn load_qtable(&mut self, path: &Path) -> Result<()> {
        self.qtable = QTable::load(path)?;
        Ok(())
    }

    /// Save Q-table
    pub fn save_qtable(&self, path: &Path) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                PRCTError::ConfigError(format!("Failed to create directory: {}", e))
            })?;
        }
        self.qtable.save(path)?;
        Ok(())
    }

    /// Save both Q-table and adaptive indexer
    ///
    /// Ensures state indexing consistency by saving both components.
    /// Automatically generates indexer path from qtable path.
    ///
    /// # Example
    /// ```ignore
    /// controller.save_with_indexer("target/fluxnet_cache/qtable_final.bin")?;
    /// // Creates:
    /// // - target/fluxnet_cache/qtable_final.bin
    /// // - target/fluxnet_cache/adaptive_indexer_final.bin
    /// ```
    pub fn save_with_indexer(&self, qtable_path: &Path) -> Result<()> {
        // Save Q-table
        self.save_qtable(qtable_path)?;

        // Generate indexer path (same directory, replace "qtable" with "adaptive_indexer")
        let indexer_path = if let Some(filename) = qtable_path.file_name() {
            let filename_str = filename.to_string_lossy();
            let indexer_filename = filename_str.replace("qtable", "adaptive_indexer");
            qtable_path.with_file_name(indexer_filename)
        } else {
            return Err(PRCTError::ConfigError("Invalid qtable path".to_string()));
        };

        // Save adaptive indexer
        self.adaptive_indexer.save(&indexer_path).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to save adaptive indexer: {}", e))
        })?;

        Ok(())
    }

    /// Load both Q-table and adaptive indexer
    ///
    /// Restores both components to ensure consistent state indexing.
    /// Automatically generates indexer path from qtable path.
    ///
    /// # Example
    /// ```ignore
    /// controller.load_with_indexer("target/fluxnet_cache/qtable_final.bin")?;
    /// // Loads:
    /// // - target/fluxnet_cache/qtable_final.bin
    /// // - target/fluxnet_cache/adaptive_indexer_final.bin
    /// ```
    pub fn load_with_indexer(&mut self, qtable_path: &Path) -> Result<()> {
        // Load Q-table
        self.load_qtable(qtable_path)?;

        // Generate indexer path
        let indexer_path = if let Some(filename) = qtable_path.file_name() {
            let filename_str = filename.to_string_lossy();
            let indexer_filename = filename_str.replace("qtable", "adaptive_indexer");
            qtable_path.with_file_name(indexer_filename)
        } else {
            return Err(PRCTError::ConfigError("Invalid qtable path".to_string()));
        };

        // Load adaptive indexer if it exists
        if indexer_path.exists() {
            self.adaptive_indexer = AdaptiveStateIndexer::load(&indexer_path).map_err(|e| {
                PRCTError::ConfigError(format!("Failed to load adaptive indexer: {}", e))
            })?;
        } else {
            // If indexer doesn't exist (legacy Q-table), start fresh
            self.adaptive_indexer = AdaptiveStateIndexer::new();
        }

        Ok(())
    }

    /// Get current epsilon
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get current replay buffer size
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// Get total updates performed
    pub fn total_updates(&self) -> usize {
        self.total_updates
    }

    /// Get number of visited state-action pairs
    pub fn num_visited_states(&self) -> usize {
        self.qtable.num_visited()
    }

    /// Get Q-table size (for state indexing)
    pub fn table_size(&self) -> usize {
        self.table_size
    }

    /// Get Q-value for state-action pair
    pub fn get_q_value(&self, state_idx: usize, action_idx: usize) -> f64 {
        self.qtable.get(state_idx, action_idx)
    }

    /// Get average reward for a phase (last 100 episodes)
    pub fn avg_reward(&self, phase: PhaseName) -> Option<f64> {
        self.phase_rewards.get(&phase).and_then(|rewards| {
            if rewards.is_empty() {
                None
            } else {
                let sum: f64 = rewards.iter().sum();
                Some(sum / rewards.len() as f64)
            }
        })
    }

    /// Get action frequency for a phase
    pub fn action_frequency(&self, phase: PhaseName) -> HashMap<usize, usize> {
        self.action_counts
            .iter()
            .filter_map(|((p, action_idx), count)| {
                if *p == phase {
                    Some((*action_idx, *count))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get adaptive indexer statistics (for telemetry)
    pub fn adaptive_indexer_stats(&self) -> super::adaptive_index::AdaptiveIndexStats {
        self.adaptive_indexer.stats()
    }

    /// Get replay buffer priority statistics (for telemetry)
    pub fn replay_buffer_stats(&self) -> ReplayBufferStats {
        self.replay_buffer.priority_stats()
    }

    /// Check if adaptive indexing is ready
    pub fn is_adaptive_ready(&self) -> bool {
        self.adaptive_indexer.is_ready()
    }

    /// Print statistics (for debugging)
    pub fn print_stats(&self) {
        println!("\n[FLUXNET] Multi-Phase RL Controller Statistics");
        println!("================================================");
        println!("Total updates: {}", self.total_updates);
        println!("Visited state-action pairs: {}", self.num_visited_states());
        println!("Current epsilon: {:.4}", self.epsilon);
        println!("Replay buffer size: {}/{}", self.replay_buffer.len(), self.replay_buffer.capacity);

        // Adaptive indexer stats
        if self.config.use_adaptive_index {
            let indexer_stats = self.adaptive_indexer.stats();
            println!("\nAdaptive Indexer:");
            println!("  Status: {}", if indexer_stats.adaptive_ready { "Ready" } else { "Learning" });
            println!("  Total samples: {}", indexer_stats.total_samples);
            println!("  Histogram size: {} (chromatic), {} (conflicts)",
                indexer_stats.chromatic_hist_size, indexer_stats.conflicts_hist_size);
            println!("  Next recompute in: {} samples", indexer_stats.samples_until_recompute);
        }

        // Priority stats
        let priority_stats = self.replay_buffer.priority_stats();
        println!("\nPrioritized Replay:");
        println!("  Mean priority: {:.2}", priority_stats.mean_priority);
        println!("  Max priority: {:.2}", priority_stats.max_priority);
        println!("  Min priority: {:.2}", priority_stats.min_priority);

        println!("\nPhase-wise Average Rewards:");
        for phase in &[
            PhaseName::Reservoir,
            PhaseName::TransferEntropy,
            PhaseName::Thermodynamic,
            PhaseName::Quantum,
        ] {
            if let Some(avg_reward) = self.avg_reward(*phase) {
                println!("  {:?}: {:.2}", phase, avg_reward);
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qtable_operations() {
        let mut qtable = QTable::new(1024);

        qtable.set(0, 0, 1.5);
        assert_eq!(qtable.get(0, 0), 1.5);

        qtable.set(0, 1, 2.5);
        assert_eq!(qtable.best_action(0), 1);
        assert_eq!(qtable.max_q_value(0), 2.5);
    }

    #[test]
    fn test_q_learning_update() {
        let mut qtable = QTable::new(1024);

        // Initial Q(s,a) = 0
        qtable.update(0, 0, 10.0, 1, 0.1, 0.9);

        // Q(s,a) should increase toward reward
        assert!(qtable.get(0, 0) > 0.0);
        assert!(qtable.get(0, 0) < 10.0); // Not fully converged yet
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..15 {
            buffer.push(Experience {
                state: UnifiedRLState::default(),
                action: FluxNetAction::NoOp,
                reward: i as f64,
                next_state: UnifiedRLState::default(),
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
            epsilon_start: 1.0,
            epsilon_decay: 0.99,
            epsilon_min: 0.05,
            learning_rate: 0.1,
            discount_factor: 0.95,
            ..Default::default()
        };

        let mut controller = MultiPhaseRLController::new(config, 1024, 1024, 200, false);

        assert_eq!(controller.epsilon(), 1.0);

        // Simulate 100 updates
        for _ in 0..100 {
            controller.update(
                UnifiedRLState::default(),
                FluxNetAction::NoOp,
                0.0,
                UnifiedRLState::default(),
                false,
            );
        }

        // Epsilon should decay but not below min
        assert!(controller.epsilon() < 1.0);
        assert!(controller.epsilon() >= 0.05);
    }

    #[test]
    fn test_multi_phase_action_selection() {
        let config = RLConfig::default();
        let mut controller = MultiPhaseRLController::new(config, 1024, 1024, 200, false);

        let state = UnifiedRLState::default();

        // Test action selection for each phase
        let reservoir_action = controller.choose_action(&state, PhaseName::Reservoir);
        assert_eq!(reservoir_action.target_phase(), Some(PhaseName::Reservoir));

        let te_action = controller.choose_action(&state, PhaseName::TransferEntropy);
        assert_eq!(te_action.target_phase(), Some(PhaseName::TransferEntropy));

        let thermo_action = controller.choose_action(&state, PhaseName::Thermodynamic);
        assert_eq!(thermo_action.target_phase(), Some(PhaseName::Thermodynamic));

        let quantum_action = controller.choose_action(&state, PhaseName::Quantum);
        assert_eq!(quantum_action.target_phase(), Some(PhaseName::Quantum));
    }

    #[test]
    fn test_visit_counts() {
        let mut qtable = QTable::new(1024);

        qtable.set(0, 0, 1.0);
        qtable.set(0, 0, 2.0);
        qtable.set(0, 1, 3.0);

        assert_eq!(qtable.visit_count(0, 0), 2);
        assert_eq!(qtable.visit_count(0, 1), 1);
        assert_eq!(qtable.visit_count(1, 0), 0);
    }
}
