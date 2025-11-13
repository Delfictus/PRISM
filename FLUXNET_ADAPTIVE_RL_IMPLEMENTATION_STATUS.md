# FluxNet Adaptive RL Enhancement - Implementation Status

## Overview

Major enhancements to FluxNet RL system implementing adaptive state indexing and prioritized experience replay as specified in the enhancement request.

**Branch:** `claude/fluxnet-rl-gpu-implementation-011CUzcMsiXXsaMumMmenNtx`
**Status:** 5/8 tasks complete - Core infrastructure done, integration pending

---

## ✅ Completed Components

### 1. Adaptive State Indexer Module
**File:** `foundation/prct-core/src/fluxnet/adaptive_index.rs` (new)

**Features:**
- `AdaptiveStateIndexer` struct with sliding histograms (2000-sample window)
- Percentile-based quantization (maps raw bins → 4-bit buckets 0-15)
- Lazy breakpoint recomputation (every 50 samples)
- Per-feature quantizers: chromatic, conflicts, iteration, GPU util, 6 phase metrics
- 100-sample threshold before adaptive indexing activates
- Full serialization support for persistence

**Key APIs:**
```rust
pub fn observe(&mut self, chromatic_bin: u8, conflicts_bin: u8, ...)
pub fn quantize_chromatic(&self, value: u8) -> u8
pub fn quantize_conflicts(&self, value: u8) -> u8
pub fn is_ready(&self) -> bool  // Returns true after 100 samples
pub fn learned_bins(&self) -> AdaptiveBinsSnapshot
pub fn stats(&self) -> AdaptiveIndexStats
```

**Tests:** 5 unit tests covering histogram eviction, percentile computation, breakpoint distribution, fallback behavior

---

### 2. UnifiedRLState Extension
**File:** `foundation/prct-core/src/fluxnet/unified_state.rs:304-399`

**Added:**
- `to_index_adaptive(&self, table_size, indexer)` method
- Packs 10 features into 64-bit key: [chromatic:4][conflicts:4][iteration:4][gpu_util:4][phase0-5:4 each]
- Falls back to `to_index()` (hash-based) if indexer not ready (<100 samples)
- Supports 16K table size (16,384 states)

**Backward Compatibility:** Original `to_index()` method preserved

---

### 3. Prioritized Replay Buffer
**File:** `foundation/prct-core/src/fluxnet/controller.rs:222-440`

**Features:**
- `ReplayBuffer` now stores priorities alongside experiences
- `push_priority(&mut self, experience, priority)` - explicit priority
- `push(&mut self, experience)` - uses max_priority (backward compatible)
- `sample_weighted(&self, batch_size, priority_eps, beta)` - priority-proportional sampling
  - Returns `(experiences, indices, importance_weights)`
  - Implements importance sampling correction: `w_i = (N * P(i))^(-beta)`
- `update_priority(&mut self, index, new_priority)` - update after TD-error
- `priority_stats()` - observability (mean/max/min/total priority)

**FIFO Eviction:** Maintains FIFO policy for capacity limits
**Priority Tracking:** Running sum for O(1) normalization

**Data Structure:**
```rust
struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    priorities: VecDeque<f64>,  // Parallel to buffer
    capacity: usize,
    max_priority: f64,          // For new experiences
    priority_sum: f64,          // For normalization
}
```

---

### 4. RLConfig Extension
**File:** `foundation/prct-core/src/fluxnet/config.rs:177-213`

**New Parameters:**
```toml
[fluxnet.rl]
use_adaptive_index = true       # Enable adaptive indexing (default: true)
priority_alpha = 0.6            # Priority exponent (default: 0.6)
priority_beta = 0.4             # IS correction (default: 0.4)
priority_eps = 0.01             # Numerical stability (default: 0.01)
high_reward_cutoff = 10.0       # Threshold for priority boost (default: 10.0)
```

**Default Q-table Size:**
- `use_adaptive_index=true` → 16,384 states (16K)
- `use_adaptive_index=false` → 4K/1K (memory tier default)

**Updated Method:**
```rust
pub fn get_qtable_states(&self, memory_tier: MemoryTier) -> usize {
    self.qtable_states.unwrap_or_else(|| {
        if self.use_adaptive_index {
            16384  // 16K for adaptive
        } else {
            memory_tier.qtable_states()  // Legacy
        }
    })
}
```

---

### 5. Module Integration
**File:** `foundation/prct-core/src/fluxnet/mod.rs:54,66`

**Exports:**
```rust
pub mod adaptive_index;
pub use adaptive_index::{AdaptiveStateIndexer, AdaptiveBinsSnapshot, AdaptiveIndexStats};
```

---

## ⏳ Pending Tasks

### 6. MultiPhaseRLController Integration
**File:** `foundation/prct-core/src/fluxnet/controller.rs:442+`

**Required Changes:**

#### A. Add Indexer Field
```rust
pub struct MultiPhaseRLController {
    qtable: QTable,
    replay_buffer: ReplayBuffer,
    config: RLConfig,
    epsilon: f64,
    table_size: usize,
    max_chromatic: usize,

    // NEW: Adaptive indexer
    adaptive_indexer: AdaptiveStateIndexer,

    phase_rewards: HashMap<PhaseName, VecDeque<f64>>,
    action_counts: HashMap<(PhaseName, usize), usize>,
    total_updates: usize,
    verbose: bool,
}
```

#### B. Update Constructor
```rust
pub fn new(...) -> Self {
    Self {
        // ... existing fields ...
        adaptive_indexer: AdaptiveStateIndexer::new(),
    }
}
```

#### C. Update `choose_action()` (line ~474)
```rust
pub fn choose_action(&mut self, state: &UnifiedRLState, phase: PhaseName) -> FluxNetAction {
    // Use adaptive indexing if enabled
    let state_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
        state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
    } else {
        state.to_index(self.table_size)
    };

    // ... rest of epsilon-greedy logic ...
}
```

#### D. Update `update()` (line ~503)
```rust
pub fn update(
    &mut self,
    state: UnifiedRLState,
    action: FluxNetAction,
    reward: f64,
    next_state: UnifiedRLState,
    done: bool,
) {
    // Observe state for adaptive indexer
    if self.config.use_adaptive_index {
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
    }

    // Use adaptive indexing if ready
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

    // Compute priority: priority = |reward|.max(1.0).powf(alpha)
    let priority = reward.abs().max(1.0).powf(self.config.priority_alpha as f64);
    let high_reward_priority = if reward.abs() > self.config.high_reward_cutoff as f64 {
        priority * 2.0  // Boost high-reward experiences
    } else {
        priority
    };

    // Store experience with priority
    self.replay_buffer.push_priority(
        Experience { state, action, reward, next_state, done },
        high_reward_priority,
    );

    // Q-learning update (immediate online update)
    let gamma = if done { 0.0 } else { self.config.discount_factor as f64 };
    self.qtable.update(state_idx, action_idx, reward, next_state_idx,
                      self.config.learning_rate as f64, gamma);

    // Optional: Prioritized batch replay (if batch_size configured)
    if let Some(batch_size) = self.config.batch_size {
        if self.replay_buffer.len() >= batch_size {
            let (experiences, indices, weights) = self.replay_buffer.sample_weighted(
                batch_size,
                self.config.priority_eps as f64,
                self.config.priority_beta as f64,
            );

            for (i, (exp, weight)) in experiences.iter().zip(weights.iter()).enumerate() {
                let s_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
                    exp.state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
                } else {
                    exp.state.to_index(self.table_size)
                };

                let a_idx = exp.action.to_index();

                let ns_idx = if self.config.use_adaptive_index && self.adaptive_indexer.is_ready() {
                    exp.next_state.to_index_adaptive(self.table_size, &self.adaptive_indexer)
                } else {
                    exp.next_state.to_index(self.table_size)
                };

                let g = if exp.done { 0.0 } else { self.config.discount_factor as f64 };

                // Weighted Q-update
                let current_q = self.qtable.get(s_idx, a_idx);
                let target_q = exp.reward + g * self.qtable.max_q_value(ns_idx);
                let td_error = target_q - current_q;
                let weighted_update = self.config.learning_rate as f64 * weight * td_error;
                self.qtable.set(s_idx, a_idx, current_q + weighted_update);

                // Update priority based on TD-error
                let new_priority = td_error.abs().max(self.config.priority_eps as f64)
                                           .powf(self.config.priority_alpha as f64);
                self.replay_buffer.update_priority(indices[i], new_priority);
            }
        }
    }

    // ... rest of update logic (epsilon decay, metrics tracking) ...
}
```

---

### 7. Telemetry Extension
**File:** `foundation/prct-core/src/fluxnet/telemetry.rs`

**Required:** Add fields to `RLDecisionTelemetry`:
```rust
pub struct RLDecisionTelemetry {
    pub temp_index: usize,
    pub state: RLStateTelemetry,
    pub action: FluxNetAction,
    pub q_value: f32,
    pub epsilon: f32,
    pub was_exploration: bool,

    // NEW: Adaptive indexing telemetry
    pub state_index_hashed: Option<usize>,    // Legacy hash-based index
    pub state_index_adaptive: Option<usize>,  // Adaptive percentile-based index
    pub indexer_ready: bool,                  // Adaptive indexer has enough samples
    pub indexer_samples: usize,               // Total samples in indexer

    // NEW: Prioritized replay telemetry
    pub experience_priority: Option<f64>,     // Priority assigned to this experience
    pub replay_buffer_mean_priority: Option<f64>,
    pub replay_buffer_max_priority: Option<f64>,
}
```

**Logging:** Update world_record_pipeline.rs Phase 1 RL logging to capture both indices

---

### 8. Unit Tests
**Files:**
- `foundation/prct-core/src/fluxnet/unified_state.rs` (test module)
- `foundation/prct-core/src/fluxnet/controller.rs` (test module)

**Required Tests:**

#### A. Adaptive Indexing Tests
```rust
#[test]
fn test_adaptive_index_stability() {
    // Given identical histograms, same state should map to same index
}

#[test]
fn test_adaptive_index_distribution() {
    // Given synthetic percentiles, indices should distribute across 16K buckets
}

#[test]
fn test_adaptive_index_fallback() {
    // Before 100 samples, should fall back to hashed indexing
}
```

#### B. Prioritized Replay Tests
```rust
#[test]
fn test_priority_ordering() {
    // High-reward experience (reward=20) should be sampled more frequently
}

#[test]
fn test_priority_eviction() {
    // High-priority experiences should survive FIFO eviction
}

#[test]
fn test_importance_sampling_weights() {
    // Weights should correct for sampling bias
}
```

---

### 9. Documentation & Migration
**Files:**
- `FLUXNET_INTEGRATION_REFERENCE.md` (update)
- `FLUXNET_IMPLEMENTATION_CHECKLIST.md` (update)
- `foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml` (example)

**Required Sections:**

#### A. Configuration Guide
```toml
[fluxnet.rl]
# Adaptive indexing (16K Q-table, percentile-based)
use_adaptive_index = true
qtable_states = 16384  # Override if needed

# Prioritized experience replay
priority_alpha = 0.6   # Higher = more aggressive prioritization
priority_beta = 0.4    # Importance sampling correction
priority_eps = 0.01    # Numerical stability
high_reward_cutoff = 10.0  # Threshold for 2x priority boost

# Batch replay (optional, disables if unset)
batch_size = 64
```

#### B. Migration Helper
Create `tools/migrate_qtable.rs` or script:
```bash
#!/bin/bash
# Migrate existing 8K Q-table to 16K adaptive Q-table
# Usage: ./migrate_qtable.sh old_qtable.bin new_qtable.bin

# Step 1: Load old 8K table
# Step 2: For each (state_idx, action_idx, q_value):
#   - Recompute state_idx using adaptive indexer (zero-initialized)
#   - Insert into new 16K table
# Step 3: Save new table
```

**Warning:** Old 8K Q-tables are incompatible with 16K adaptive tables - discard or migrate

---

## Testing Strategy

### Phase 1: Unit Tests (30 min)
```bash
cargo test --package prct-core fluxnet::adaptive_index
cargo test --package prct-core fluxnet::unified_state::test_adaptive
cargo test --package prct-core fluxnet::controller::test_priority
```

### Phase 2: Integration Test (10 min)
```bash
# Small graph with adaptive indexing enabled
cargo run --release --features cuda --example world_record_dsjc1000 \
    benchmarks/dimacs/DSJC125.5.col \
    --max-runtime-hours 0.1 \
    --config-override "fluxnet.rl.use_adaptive_index=true"
```

**Success Criteria:**
- Indexer reaches 100 samples and activates adaptive mode
- Both indices logged in telemetry (hashed vs adaptive)
- Priority stats show non-uniform distribution (mean < max)
- No crashes, Q-table saves/loads correctly

### Phase 3: DSJC250.5 Smoke Test (30 min)
```bash
# Train on medium graph to validate learning
cargo run --release --features cuda --example world_record_dsjc1000 \
    benchmarks/dimacs/DSJC250.5.col \
    --target-chromatic 28 \
    --max-runtime-hours 0.5
```

**Expected:**
- Adaptive indexer stabilizes after ~2-3 restarts
- High-reward experiences get 2x priority boost
- Batch replay (if enabled) samples weighted experiences
- Chromatic improvement comparable to baseline

### Phase 4: DSJC1000.5 Full Run (48 hours)
Use updated `wr_sweep_D_aggr_seed_9001.v1.1.toml` with adaptive RL

---

## Performance Impact

**Memory:**
- Adaptive indexer: ~120 KB (2000 samples × 10 features × 6 bytes)
- Prioritized replay: +8 bytes per experience (f64 priority)
- Total overhead: <200 KB (negligible for 16GB system)

**Compute:**
- Percentile quantization: ~10 µs per state (10 features × 1 µs lookup)
- Priority sampling: ~50 µs for 64-batch (cumulative sum + importance weights)
- Breakpoint recomputation: ~5 ms every 50 samples (lazy, amortized)

**Expected:** <1% overhead on 48-hour runs

---

## Next Steps

1. **Complete MultiPhaseRLController integration** (~1 hour)
   - Add adaptive_indexer field
   - Update choose_action and update methods
   - Wire prioritized batch replay

2. **Add telemetry logging** (~30 min)
   - Extend RLDecisionTelemetry
   - Log both indices + priority stats

3. **Write unit tests** (~1 hour)
   - Adaptive indexing stability
   - Priority ordering and eviction
   - Fallback behavior

4. **Update documentation** (~30 min)
   - Configuration guide
   - Migration notes
   - Example configs

5. **Run integration tests** (~1 hour)
   - DSJC125.5 smoke test
   - DSJC250.5 validation
   - Check telemetry logs

6. **Commit and push** (~15 min)
   - Descriptive commit message
   - Reference enhancement request

**Total Estimated Time:** 4-5 hours to complete

---

## Files Modified

### New Files:
- `foundation/prct-core/src/fluxnet/adaptive_index.rs`
- `FLUXNET_ADAPTIVE_RL_IMPLEMENTATION_STATUS.md` (this file)

### Modified Files:
- `foundation/prct-core/src/fluxnet/mod.rs`
- `foundation/prct-core/src/fluxnet/unified_state.rs`
- `foundation/prct-core/src/fluxnet/controller.rs`
- `foundation/prct-core/src/fluxnet/config.rs`

### Pending Modifications:
- `foundation/prct-core/src/fluxnet/telemetry.rs`
- `foundation/prct-core/src/world_record_pipeline.rs` (Phase 1 RL logging)
- Test files (add new test functions)
- Documentation files

---

## Rollback Plan

If issues arise:

1. **Disable adaptive indexing:**
   ```toml
   [fluxnet.rl]
   use_adaptive_index = false  # Falls back to 8K hash-based
   ```

2. **Disable prioritized replay:**
   ```toml
   # Remove or comment out batch_size
   # batch_size = 64
   ```

3. **Revert to baseline:**
   ```bash
   git checkout b84e50b  # Before adaptive RL changes
   ```

All changes are backward-compatible - existing configs continue to work.

---

## Summary

**Status:** Core infrastructure complete (5/8 tasks)

**Remaining:** Controller integration, telemetry, tests, docs (~4-5 hours)

**Risk:** Low - backward compatible, incremental deployment

**Benefit:** Improved Q-table distribution + better experience replay → faster convergence

**Ready for:** Integration testing after controller wiring completed
