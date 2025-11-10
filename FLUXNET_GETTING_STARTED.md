# FluxNet RL Implementation - Getting Started Guide

## Welcome to the FluxNet Worktree!

This is a **separate git worktree** for implementing the FluxNet RL force profile system. You can work here independently without affecting the main branch.

## Directory Structure

```
worktrees/fluxnet-rl/               # Your isolated development environment
├── FLUX-NET-PLAN.txt               # Complete implementation plan
├── FLUXNET_GETTING_STARTED.md      # This file
├── FLUXNET_IMPLEMENTATION_CHECKLIST.md  # Step-by-step task list
├── FLUXNET_INTEGRATION_REFERENCE.md     # Quick reference for integration points
├── foundation/prct-core/           # Core graph coloring pipeline
│   ├── src/
│   │   ├── world_record_pipeline_gpu.rs  # Main pipeline (Phase 0-6)
│   │   ├── gpu_thermodynamic.rs          # Phase 2 thermodynamic kernel (MAIN INTEGRATION)
│   │   ├── telemetry/                    # Telemetry system (per-temp logging)
│   │   └── fluxnet/                      # NEW: FluxNet RL module (you will create)
│   ├── configs/                    # Configuration files
│   │   ├── wr_sweep_D.v1.1.toml         # Base WR config
│   │   └── dsjc250.5.toml               # Pre-training config (DSJC250)
│   └── examples/
│       └── world_record_dsjc1000.rs     # WR benchmark entry point
└── foundation/neuromorphic/        # Neuromorphic reservoir (Phase 0)
```

## Branch Information

- **Branch:** `feature/fluxnet-rl`
- **Based on:** `main` at commit `debdbad`
- **Status:** Clean slate, ready for FluxNet implementation

## Quick Start

### 1. Verify Environment

```bash
# Check CUDA availability
nvidia-smi

# Check Rust toolchain
cargo --version
rustc --version

# Build current code (should compile cleanly)
cargo build --release --features cuda
```

### 2. Understand the Current Pipeline

The PRISM pipeline has 7 phases:

- **Phase 0:** GPU Neuromorphic Reservoir (conflict prediction) → outputs `difficulty_scores`
- **Phase 1:** Active Inference (policy selection) → outputs `ai_uncertainty`
- **Phase 2:** GPU Thermodynamic Equilibration → **MAIN FLUXNET INTEGRATION POINT**
- **Phase 3:** Quantum-Classical Hybrid
- **Phase 4:** Memetic Algorithm
- **Phase 5:** TSP Refinement
- **Phase 6:** TDA (Topological Data Analysis)

### 3. FluxNet Implementation Overview

FluxNet adds:

1. **ForceProfile** system - Classify vertices into Strong/Neutral/Weak bands
2. **RL Controller** - Q-learning agent that adjusts force bands per temperature
3. **Force Commands** - Band-specific multipliers injected into Phase 2 thermodynamic kernel
4. **Pre-training** - Warm-start Q-table on DSJC250.5 before DSJC1000 run
5. **Per-temp telemetry** - Track RL decisions at each of 48 temperature steps

## Key Integration Points

### Phase 0 → FluxNet
```rust
// After reservoir training, export difficulty scores
let difficulty_scores = reservoir_predictor.conflict_scores;
// Feed into ForceProfile initialization
```

### Phase 1 → FluxNet
```rust
// After Active Inference, export uncertainty
let ai_uncertainty = policy_selector.uncertainty_scores;
// Update ForceProfile with AI uncertainty
```

### Phase 2 (Main Integration)
```rust
// CURRENT: gpu_thermodynamic.rs equilibrate_thermodynamic_gpu()
// NEW: FluxNet adds ForceCommand processing per temperature

for temp_idx in 0..num_temps {
    // RL controller observes telemetry state
    let state = rl_controller.observe(temp_idx, telemetry);

    // RL controller issues ForceCommand
    let command = rl_controller.select_action(state);

    // Apply command to thermodynamic kernel
    apply_force_command(command, force_profile);

    // Run thermodynamic evolution at this temperature
    evolve_oscillators_kernel(...);

    // Compute reward for RL training
    let reward = compute_reward(telemetry_before, telemetry_after);
    rl_controller.update(state, command, reward);
}
```

## Configuration Files

### Base Config: `foundation/prct-core/configs/wr_sweep_D.v1.1.toml`
- Target: 83 colors (world record)
- Quantum depth: 8
- Thermo temps: 56
- Runtime: 24 hours

### Pre-training Config: `foundation/prct-core/configs/dsjc250.5.toml`
- Smaller graph (~250 vertices)
- Fast training (~3-5 minutes)
- Builds initial Q-table

## Telemetry System

Located: `foundation/prct-core/src/telemetry/`

Current telemetry includes:
- `temp_index` - Temperature step (0-47)
- `chromatic_before/after` - Number of colors
- `conflicts` - Constraint violations
- `unique_buckets` - Color diversity metric
- `compaction_ratio` - Convergence health (target >0.6)

**FluxNet additions needed:**
- `force_band_stats` - {strong_fraction, weak_fraction, mean}
- `force_command_applied` - Which band was adjusted
- `rl_action` - Action taken by RL agent
- `rl_reward` - Reward computed
- `rl_q_delta` - Q-value update magnitude

## Build Commands

```bash
# Build with CUDA support
cargo build --release --features cuda

# Run quick test
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/quick_test.toml --max-minutes 2

# Pre-train FluxNet on DSJC250 (after implementation)
cargo run --release --features cuda --example world_record_dsjc250 \
    foundation/prct-core/configs/dsjc250.5.toml --max-minutes 5

# Run WR attempt with FluxNet (after implementation)
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D.v1.1.toml --max-minutes 60
```

## Testing Strategy

1. **Unit tests** - Test ForceProfile blending, RL state/action encoding
2. **Integration test** - DSJC250 pre-training (verify Q-table saved)
3. **Smoke test** - DSJC1000 5-minute run (verify no crashes, telemetry present)
4. **Performance test** - DSJC1000 60-minute run (target: no mid-temp collapse, compaction_ratio >0.6)

## Memory Constraints

- **Compact mode** (8GB GPU): 1k replay buffer, 256-state Q-table
- **Extended mode** (24GB+ GPU): 16k replay buffer, 1k×16 Q-table

Configure via TOML:
```toml
[fluxnet]
enabled = true
memory_tier = "compact"  # or "extended"
replay_capacity = 1024
qtable_states = 256
persistence_path = "target/fluxnet_cache"
```

## Success Criteria

**Phase 1: Basic Implementation**
- [ ] ForceProfile data structures compile
- [ ] RL controller compiles
- [ ] Phase 2 integration compiles
- [ ] Pre-training runs on DSJC250

**Phase 2: Functional**
- [ ] DSJC250 pre-training produces Q-table file
- [ ] DSJC1000 loads Q-table and runs
- [ ] Telemetry shows RL actions per temperature
- [ ] No crashes during 60-minute run

**Phase 3: Effective**
- [ ] Temperatures 7-34 maintain >20 colors (no collapse)
- [ ] Compaction ratio stays >0.6
- [ ] Conflict count decreases over time
- [ ] Final chromatic ≤95 colors

**Phase 4: World Record**
- [ ] Final chromatic ≤83 colors (matches or beats WR)

## Next Steps

1. Read `FLUX-NET-PLAN.txt` for complete implementation details
2. Read `FLUXNET_IMPLEMENTATION_CHECKLIST.md` for step-by-step tasks
3. Read `FLUXNET_INTEGRATION_REFERENCE.md` for code snippets
4. Start with Task A: Core Data Types (ForceProfile, ForceCommand, FluxNetConfig)

## Git Workflow

This worktree is on branch `feature/fluxnet-rl`. When ready to merge:

```bash
# Commit your work in the worktree
cd /path/to/worktrees/fluxnet-rl
git add .
git commit -m "feat: FluxNet RL implementation"

# Switch to main and merge
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
git checkout main
git merge feature/fluxnet-rl
```

## Resources

- **🚨 GPU Mandate:** `GPU_MANDATE.md` - **CRITICAL: Never disable GPU, always fix bugs properly**
- **FluxNet Plan:** `FLUX-NET-PLAN.txt`
- **Implementation Checklist:** `FLUXNET_IMPLEMENTATION_CHECKLIST.md`
- **Integration Reference:** `FLUXNET_INTEGRATION_REFERENCE.md`
- **GPU Orchestrator Guide:** `PRISM_GPU_ORCHESTRATOR_GUIDE.md`
- **Phase 2 Kernel:** `foundation/prct-core/src/gpu_thermodynamic.rs`
- **Telemetry System:** `foundation/prct-core/src/telemetry/`
- **World Record Pipeline:** `foundation/prct-core/src/world_record_pipeline_gpu.rs`

## Questions?

This worktree is isolated - experiment freely! If something breaks, you can always:
1. Reset to the starting commit: `git reset --hard debdbad`
2. Delete and recreate the worktree

Happy coding!
