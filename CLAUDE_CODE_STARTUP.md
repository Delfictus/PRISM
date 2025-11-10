# Claude Code Research Preview - Startup Commands

## Initial Setup Instructions

Copy and paste this **entire section** to Claude Code Research Preview when starting a new session:

---

**FLUXNET RL IMPLEMENTATION - STARTUP SEQUENCE**

I'm working on the FluxNet RL implementation for PRISM. Follow these commands in order:

### Step 1: Pull Latest Changes

```bash
git fetch origin
git checkout feature/fluxnet-rl
git pull origin feature/fluxnet-rl
```

**Verify you're on the correct branch:**
```bash
git branch --show-current
```
Expected output: `feature/fluxnet-rl`

### Step 2: Read Critical Documentation (IN ORDER)

```bash
# 🚨 CRITICAL: Read GPU mandate first
cat GPU_MANDATE.md | head -100

# Quick start guide
cat START_HERE.md

# Comprehensive getting started
cat FLUXNET_GETTING_STARTED.md | head -150

# GPU Orchestrator standards
cat PRISM_GPU_ORCHESTRATOR_GUIDE.md | head -150
```

**Key takeaways from GPU_MANDATE.md:**
- ❌ NEVER disable CUDA features
- ❌ NEVER add CPU fallbacks
- ✅ ALWAYS fix GPU bugs properly
- ✅ GPU-FIRST, GPU-ONLY, GPU-ALWAYS

### Step 3: Verify Environment

```bash
# Check CUDA availability
nvidia-smi

# Verify Rust toolchain
cargo --version
rustc --version

# Check current codebase compiles
cargo build --release --features cuda
```

**Expected:** Build should complete successfully with CUDA features enabled.

### Step 4: Run Baseline Test

```bash
# Quick smoke test (2 minutes)
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/quick_test.toml --max-minutes 2
```

**Expected:** Should run without errors, showing GPU initialization and phases executing.

### Step 5: Review Implementation Plan

```bash
# View the checklist
cat FLUXNET_IMPLEMENTATION_CHECKLIST.md | head -200
```

**Starting point:** Phase A - Core Data Structures

**First tasks:**
- A.1: Create `foundation/prct-core/src/fluxnet/profile.rs` (ForceProfile, ForceBand)
- A.2: Create `foundation/prct-core/src/fluxnet/command.rs` (ForceCommand)
- A.3: Add FluxNetConfig to `foundation/shared-types/src/lib.rs`

### Step 6: Confirm Ready to Start

Respond with:
- ✅ Confirmed on `feature/fluxnet-rl` branch
- ✅ Latest changes pulled
- ✅ GPU mandate read and understood
- ✅ Build passes with CUDA
- ✅ Smoke test passes
- ✅ Ready to implement Phase A

---

## Once Setup is Complete

After Claude confirms it's ready, tell it:

**"Begin implementing Phase A.1 from FLUXNET_IMPLEMENTATION_CHECKLIST.md. Create `foundation/prct-core/src/fluxnet/profile.rs` with ForceProfile, ForceBand, and ForceBandStats. Use code snippets from FLUXNET_INTEGRATION_REFERENCE.md section 1. Remember: GPU device buffers are MANDATORY (device_f_strong, device_f_weak as CudaSlice<f32>)."**

---

## Reference Documents Available

- 🚨 **GPU_MANDATE.md** - Critical GPU-first policy (READ FIRST)
- 📖 **FLUXNET_GETTING_STARTED.md** - Comprehensive guide
- ✅ **FLUXNET_IMPLEMENTATION_CHECKLIST.md** - Step-by-step tasks
- 🔧 **FLUXNET_INTEGRATION_REFERENCE.md** - Copy-paste code snippets
- 🤖 **PRISM_GPU_ORCHESTRATOR_GUIDE.md** - GPU standards
- 📋 **FLUX-NET-PLAN.txt** - Original detailed plan
- 📄 **START_HERE.md** - Quick overview

## Critical Reminders

**GPU Standards:**
- Use `Arc<CudaDevice>` for context sharing (Article V)
- PTX kernels compiled in `build.rs` (Article VII)
- Device buffers: `CudaSlice<T>` for all GPU data
- Synchronization: `cuda_device.synchronize()` before reading results

**Policy Checks:**
```bash
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh
SUB=stubs ./tools/mcp_policy_checks.sh
SUB=cuda_gates ./tools/mcp_policy_checks.sh
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh
```

**FluxNet Architecture:**
```
Phase 0: Reservoir → difficulty_scores → ForceProfile init
Phase 1: AI → ai_uncertainty → ForceProfile update
Phase 2: RL Controller → ForceCommand → GPU Thermodynamic Kernel
```

## Progress Tracking

After completing each phase:

```bash
# Check what changed
git status
git diff

# Test compilation
cargo build --release --features cuda

# Run policy checks
SUB=cuda_gates ./tools/mcp_policy_checks.sh

# Commit progress
git add <files>
git commit -m "feat: Phase X - <description>"
git push
```

---

**Repository:** https://github.com/Delfictus/PRISM.git
**Branch:** feature/fluxnet-rl
**Goal:** Implement FluxNet RL to achieve ≤83 colors on DSJC1000 (world record)

🚀 **Ready to build world-record graph coloring with RL!**
