# FluxNet RL GPU Testing Guide

## ✅ Implementation Status

**Phase E - Telemetry Integration: COMPLETE**

All FluxNet components are implemented and integrated:
- ✅ Phase A.1: ForceProfile (GPU-accelerated force bands)
- ✅ Phase A.2: ForceCommand (RL actions)
- ✅ Phase A.3: FluxNetConfig (TOML configuration)
- ✅ Phase D: RLController (Q-learning agent)
- ✅ Phase C: RL integration with Phase 2 thermodynamic loop
- ✅ Phase C.2: CUDA kernel force buffer integration
- ✅ **Phase E: Telemetry integration** (JUST COMPLETED)

## Prerequisites

### GPU Hardware Requirements
- NVIDIA GPU with CUDA support
- 8GB+ VRAM (Compact mode)
- 24GB+ VRAM (Extended mode, recommended)

### Software Requirements
```bash
# CUDA Toolkit 11.x or 12.x
nvidia-smi  # Verify GPU availability
nvcc --version  # Verify CUDA compiler

# Rust toolchain
cargo --version  # Should be 1.70+
rustc --version
```

## Build Process

### 1. Clean Build
```bash
cd /path/to/PRISM
cargo clean
cargo build --release --features cuda
```

**Expected Output:**
- Compiles cudarc dependencies
- Links CUDA libraries
- Builds PTX kernels from `foundation/kernels/thermodynamic.cu`
- Generates `target/ptx/thermodynamic.ptx`

### 2. Verify Build
```bash
# Check that CUDA kernels are present
ls -lh target/ptx/thermodynamic.ptx

# Check prct-core library
ls -lh target/release/libprct_core.rlib
```

## Configuration

### FluxNet Configuration (TOML)

Create `foundation/prct-core/configs/fluxnet_test.toml`:

```toml
[graph]
path = "data/graphs/DSJC250.5.col"

[world_record]
target_chromatic = 28  # Known chromatic for DSJC250.5
max_runtime_minutes = 5

[thermodynamic]
t_min = 0.01
t_max = 1.0
num_temps = 24
steps_per_temp = 1000

[fluxnet]
enabled = true
memory_tier = "compact"  # or "extended" for 24GB+ GPU

[fluxnet.force_profile]
strong_multiplier = 1.5
weak_multiplier = 0.7
neutral_multiplier = 1.0

[fluxnet.rl]
learning_rate = 0.001
discount_factor = 0.95
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
reward_conflict_weight = 1.0
reward_color_weight = 0.5
reward_compaction_weight = 0.3

[fluxnet.persistence]
save_dir = "target/fluxnet_cache"
save_interval_temps = 5
load_pretrained = false
# pretrained_path = "target/fluxnet_cache/qtable_dsjc250.bin"
```

## Test Phases

### Phase 1: Smoke Test (2 minutes)
**Goal:** Verify FluxNet loads and runs without crashes

```bash
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/fluxnet_test.toml --max-minutes 2
```

**Expected Output:**
```
[FLUXNET] Initializing FluxNet RL controller
[FLUXNET] Memory tier: Compact
[FLUXNET] Learning rate: 0.001
[FLUXNET] Epsilon: 1.0 → 0.01
[FLUXNET] Initializing ForceProfile from reservoir difficulty scores
[FLUXNET][T=0] State: conflicts=X, chromatic=Y, compaction=Z
[FLUXNET][T=0] Action: BoostStrong(1.2) (Q=0.000, ε=1.000, explore=true)
[FLUXNET][T=0] Applied BS: mean_force 1.000 → 1.050
[THERMO-GPU] Starting GPU thermodynamic equilibration
...
[FLUXNET][T=0] Reward: 0.123, Q: 0.000 → 0.000 (Δ=0.000), ε=0.995
```

**Verification:**
- [ ] No crashes or panics
- [ ] FluxNet initialization messages appear
- [ ] RL actions logged per temperature
- [ ] Telemetry file created: `target/run_artifacts/live_metrics_*.jsonl`

### Phase 2: Pre-Training (5-10 minutes)
**Goal:** Train Q-table on smaller graph (DSJC250.5)

```bash
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/fluxnet_test.toml --max-minutes 10
```

**Expected Outcomes:**
- [ ] Q-table checkpoints saved every 5 temps: `target/fluxnet_cache/qtable_checkpoint_temp5.bin`
- [ ] Final Q-table: `target/fluxnet_cache/qtable_final.bin`
- [ ] Epsilon decays: 1.0 → ~0.01
- [ ] Q-values diverge from 0.0 (learning signal)

**Telemetry Checks:**
```bash
# View telemetry
cat target/run_artifacts/live_metrics_*.jsonl | jq '.parameters.fluxnet'
```

**Expected JSON Structure:**
```json
{
  "force_bands": {
    "strong_fraction": 0.2,
    "neutral_fraction": 0.6,
    "weak_fraction": 0.2,
    "mean_force": 1.0,
    "min_force": 0.7,
    "max_force": 1.5,
    "force_stddev": 0.2
  },
  "rl_decision": {
    "temp_index": 5,
    "state": {
      "conflicts": 100,
      "chromatic_number": 95,
      "compaction_ratio": 0.75,
      "state_index": 42
    },
    "action": {"BoostStrong": 1.2},
    "q_value": 0.234,
    "epsilon": 0.95,
    "was_exploration": false
  },
  "config": {
    "memory_tier": "Compact",
    "qtable_states": 256,
    "learning_rate": 0.001,
    ...
  }
}
```

### Phase 3: DSJC1000 Run (60 minutes)
**Goal:** Full world-record attempt with pre-trained Q-table

```toml
# Update fluxnet_test.toml
[graph]
path = "data/graphs/DSJC1000.5.col"

[world_record]
target_chromatic = 83  # World record

[thermodynamic]
num_temps = 48
steps_per_temp = 2000

[fluxnet.persistence]
load_pretrained = true
pretrained_path = "target/fluxnet_cache/qtable_final.bin"
```

```bash
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/fluxnet_test.toml --max-minutes 60
```

**Success Criteria:**
- [ ] **No mid-temp collapse:** Temps 7-34 maintain >20 colors
- [ ] **Compaction ratio >0.6:** Healthy convergence
- [ ] **Conflict reduction:** Decreases over time
- [ ] **Final chromatic ≤95:** Competitive result
- [ ] **🏆 Final chromatic ≤83:** World record!

## Telemetry Analysis

### Extract FluxNet Metrics
```bash
# Force band evolution
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq -r '[.parameters.temp_index, .parameters.fluxnet.force_bands.strong_fraction] | @csv'

# RL exploration rate
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq -r '[.parameters.temp_index, .parameters.fluxnet.rl_decision.epsilon] | @csv'

# Q-value progression
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq -r '[.parameters.temp_index, .parameters.fluxnet.rl_decision.q_value] | @csv'

# Actions taken
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq '.parameters.fluxnet.rl_decision.action'
```

### Performance Metrics
```bash
# Chromatic number trajectory
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq -r '[.parameters.temp_index, .chromatic_number, .conflicts] | @csv'

# Compaction health
cat target/run_artifacts/live_metrics_*.jsonl | \
  jq -r '[.parameters.temp_index, .parameters.compaction_ratio] | @csv'
```

## Troubleshooting

### CUDA Errors

**Error:** `Unable to find include/cuda.h`
```bash
# Set CUDA_ROOT explicitly
export CUDA_ROOT=/usr/local/cuda
cargo build --release --features cuda
```

**Error:** `CUDA out of memory`
- Use `memory_tier = "compact"` in config
- Reduce `num_temps` or `steps_per_temp`
- Check GPU memory: `nvidia-smi`

### FluxNet Issues

**No FluxNet telemetry in output:**
- Verify `fluxnet.enabled = true` in config
- Check logs for "[FLUXNET]" messages
- Ensure `telemetry_handle` is passed to `equilibrate_thermodynamic_gpu()`

**Q-values stuck at 0.0:**
- Reward signal may be too weak
- Increase `reward_conflict_weight` or `reward_color_weight`
- Check that conflicts are changing between temperatures

**All actions are NoOp:**
- Epsilon may be stuck at 0.0 (check decay rate)
- Q-table may have all-zero values (check initialization)

## Next Steps After Testing

### 1. Analyze Results
- Plot force band evolution over temperatures
- Analyze Q-value convergence
- Compare with/without FluxNet runs

### 2. Hyperparameter Tuning
- Adjust learning rate, epsilon decay
- Tune reward weights for better signal
- Experiment with force multipliers

### 3. World Record Push
- Extended run (24 hours): `--max-minutes 1440`
- Memory tier "extended" with 24GB+ GPU
- Pre-train on multiple problem instances

## Validation Checklist

- [ ] Code compiles with `--features cuda`
- [ ] Smoke test runs without crashes
- [ ] FluxNet initialization logs appear
- [ ] Telemetry contains "fluxnet" field
- [ ] Force band statistics vary per temperature
- [ ] RL actions change over time (not all NoOp)
- [ ] Q-values update (diverge from 0.0)
- [ ] Epsilon decays from 1.0 to min value
- [ ] Q-table checkpoints saved
- [ ] Pre-trained Q-table loads successfully
- [ ] DSJC1000 run completes without errors
- [ ] Compaction ratio stays >0.6
- [ ] No mid-temp collapse (temps 7-34 >20 colors)

## Contact / Support

If you encounter issues:
1. Check `GPU_MANDATE.md` - Never disable GPU, always fix bugs
2. Review `PRISM_GPU_ORCHESTRATOR_GUIDE.md` for debugging
3. Inspect telemetry: `target/run_artifacts/live_metrics_*.jsonl`
4. Check CUDA errors: `dmesg | grep -i cuda`

---

**FluxNet RL is ready for GPU testing! 🚀**

All code is implemented, integrated, and telemetry-enabled. The system awaits GPU hardware for validation and world-record attempts.
