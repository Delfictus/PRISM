# 🚀 FluxNet RL Worktree - START HERE

Welcome to the isolated FluxNet RL implementation worktree!

## Quick Links

🎯 **[CLAUDE_CODE_STARTUP.md](CLAUDE_CODE_STARTUP.md)** - **START HERE: Complete startup sequence**
🚨 **[GPU_MANDATE.md](GPU_MANDATE.md)** - **READ FIRST: NEVER DISABLE GPU - FIX BUGS PROPERLY**
📖 **[FLUXNET_GETTING_STARTED.md](FLUXNET_GETTING_STARTED.md)** - Comprehensive guide
✅ **[FLUXNET_IMPLEMENTATION_CHECKLIST.md](FLUXNET_IMPLEMENTATION_CHECKLIST.md)** - Step-by-step tasks
🔧 **[FLUXNET_INTEGRATION_REFERENCE.md](FLUXNET_INTEGRATION_REFERENCE.md)** - Code snippets
📋 **[FLUX-NET-PLAN.txt](FLUX-NET-PLAN.txt)** - Complete implementation plan
🤖 **[PRISM_GPU_ORCHESTRATOR_GUIDE.md](PRISM_GPU_ORCHESTRATOR_GUIDE.md)** - GPU orchestrator agent usage

## What is FluxNet?

FluxNet adds reinforcement learning to PRISM's thermodynamic Phase 2:
- Classifies vertices into **Strong/Neutral/Weak** force bands based on difficulty
- RL controller adjusts forces **per temperature step** (48 times per pass)
- Prevents **mid-temp collapse** (temps 7-34 maintaining >20 colors)
- Targets **world record** chromatic number (≤83 colors on DSJC1000)

## Architecture Overview

```
Phase 0: Reservoir → difficulty_scores → ForceProfile init
Phase 1: AI Inference → ai_uncertainty → ForceProfile update
Phase 2: Thermodynamic (per temp):
    RL observes → selects action → issues ForceCommand
    → Phase 2 applies forces → GPU kernel evolves
    → RL computes reward → updates Q-table
```

## Where to Start

### Option 1: Read First (Recommended)
1. Read `FLUXNET_GETTING_STARTED.md` (15 min)
2. Read `FLUX-NET-PLAN.txt` sections A-D (20 min)
3. Skim `FLUXNET_INTEGRATION_REFERENCE.md` (10 min)
4. Start implementing from `FLUXNET_IMPLEMENTATION_CHECKLIST.md`

### Option 2: Dive In
1. Open `FLUXNET_IMPLEMENTATION_CHECKLIST.md`
2. Start with **Phase A: Core Data Structures**
3. Refer to `FLUXNET_INTEGRATION_REFERENCE.md` for code snippets

## Quick Test Commands

```bash
# Build (should compile cleanly)
cargo build --release --features cuda

# Quick smoke test (2 min)
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/quick_test.toml --max-minutes 2
```

## Success Criteria

✅ **Phase 1:** Compiles, pre-training runs
✅ **Phase 2:** DSJC1000 runs without crashes, telemetry shows RL
✅ **Phase 3:** Temps 7-34 maintain >20 colors, compaction >0.6
🏆 **Phase 4:** Final chromatic ≤83 (world record!)

## Time Estimate

**Total: 33-46 hours** (~1 week full-time, 2-3 weeks part-time)

## Claude Code Usage

### Web Interface (https://claude.ai/code)
When starting a new session, provide this path:
```
/home/diddy/Desktop/PRISM-FINNAL-PUSH/worktrees/fluxnet-rl
```

## Module Structure Created

```
foundation/prct-core/src/fluxnet/
├── mod.rs        ✅ Created (placeholder with docs)
└── README.md     ✅ Created
```

**Ready to implement? Start with Phase A in `FLUXNET_IMPLEMENTATION_CHECKLIST.md`!** 🎯
