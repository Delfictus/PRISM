# PRISM Command Reference

**Working Directory**: `/home/diddy/Desktop/PRISM-FINNAL-PUSH/`

---

## Git Status

### Current Branch
```bash
git branch --show-current
# Output: chore/gitignore-artifact-cleanup
```

### View Changes
```bash
# See what was untracked
git show --stat

# See detailed .gitignore changes
git show HEAD -- .gitignore
```

### Switch Between Branches
```bash
# Go back to main GPU branch
git checkout gpu-quantum-acceleration

# Return to cleanup branch
git checkout chore/gitignore-artifact-cleanup
```

---

## WR Seed Probe Commands

### Directory
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH/
```

### Validate Configs
```bash
./tools/validate_wr_sweep.sh
```

### Run Seed Probe (90-minute timeout per config)
```bash
# Default (90 minutes per config, ~13.5 hours total for 9 configs)
./tools/run_wr_seed_probe.sh

# Custom timeout (60 minutes per config, ~9 hours total)
MAX_MINUTES=60 ./tools/run_wr_seed_probe.sh

# Quick test (5 minutes per config, ~45 minutes total)
MAX_MINUTES=5 ./tools/run_wr_seed_probe.sh
```

### View Results
```bash
# JSONL summary
cat results/dsjc1000_seed_probe.jsonl

# Individual run logs
ls -lh results/logs/wr_sweep_*

# Watch live output during run
tail -f results/logs/wr_sweep_D_seed_42_<timestamp>.log
```

### Run Single Config Manually
```bash
# Config D with seed 42
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_seed_42.v1.1.toml

# Config F with seed 1337
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_F_seed_1337.v1.1.toml

# Aggressive D with seed 9001
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml
```

---

## Build and Test Commands

### Basic Builds
```bash
# Check compilation (debug mode)
cargo check

# Release build with CUDA
cargo build --release --features cuda

# World record example (CUDA required)
cd foundation/prct-core
cargo build --release --features cuda --example world_record_dsjc1000
```

### GPU Policy Checks
```bash
# Cargo CUDA check
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh

# GPU info
SUB=gpu_info ./tools/mcp_policy_checks.sh

# Check for stubs
SUB=stubs ./tools/mcp_policy_checks.sh

# CUDA gates check
SUB=cuda_gates ./tools/mcp_policy_checks.sh

# GPU reservoir check
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh
```

### Run Tests
```bash
# All tests (with 60-second timeout)
timeout 60 cargo test

# CUDA-specific tests
timeout 60 cargo test --features cuda

# Specific test
cargo test test_name --features cuda
```

---

## Git History Cleanup (Optional)

⚠️ **WARNING**: Only run these commands if you have typed **"APPROVED: history purge"**

### Check What Would Be Removed
```bash
# Show large files in history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '$1 == "blob" && $3 >= 50000000' | \
  sort -rn -k3 | \
  head -20
```

### Create History Purge Branch (After Approval)
```bash
# Install git-filter-repo (if needed)
pip3 install git-filter-repo

# Create branch
git checkout -b maintenance/purge-blobs

# Remove large blobs from entire history
git filter-repo --path deps --path venv --path lib --path libonnxruntime_providers_cuda.so --invert-paths

# Force push (destructive!)
git push -u origin maintenance/purge-blobs --force

# All collaborators must re-clone after this
```

---

## Merge Local Changes

### Option 1: Merge Cleanup to Main
```bash
# Go to your main branch
git checkout gpu-quantum-acceleration

# Merge the cleanup
git merge chore/gitignore-artifact-cleanup

# Future commits won't include build artifacts
```

### Option 2: Cherry-Pick Just .gitignore
```bash
# Go to your main branch
git checkout gpu-quantum-acceleration

# Get just the .gitignore change
git checkout chore/gitignore-artifact-cleanup -- .gitignore
git commit -m "chore: Add comprehensive .gitignore for build artifacts"
```

---

## Seed Probe Configuration Files

### Location
```bash
ls -1 foundation/prct-core/configs/wr_sweep_*seed*.toml
```

### Config D Variants (Quantum-deeper)
- `wr_sweep_D_seed_42.v1.1.toml`
- `wr_sweep_D_seed_1337.v1.1.toml`
- `wr_sweep_D_seed_9001.v1.1.toml`

### Config D Aggressive Variants (depth=7, attempts=224, gens=900)
- `wr_sweep_D_aggr_seed_42.v1.1.toml`
- `wr_sweep_D_aggr_seed_1337.v1.1.toml`
- `wr_sweep_D_aggr_seed_9001.v1.1.toml`

### Config F Variants (Thermo/PIMC-heavy)
- `wr_sweep_F_seed_42.v1.1.toml`
- `wr_sweep_F_seed_1337.v1.1.toml`
- `wr_sweep_F_seed_9001.v1.1.toml`

---

## Key Documentation Files

```bash
# Seed probe delivery summary
cat SEED_PROBE_DELIVERY.md

# WR sweep quickstart
cat WR_SWEEP_QUICKSTART.md

# WR sweep strategy
cat WR_SWEEP_STRATEGY.md

# Config verification report
cat CONFIG_V1.1_VERIFICATION_REPORT.md

# Git cleanup PR notes
cat GITIGNORE_CLEANUP_PR.md

# This file
cat COMMAND_REFERENCE.md
```

---

## Cleanup Commands

### Remove Build Artifacts (Local)
```bash
# Remove all build outputs
cargo clean

# Remove specific targets
rm -rf target/debug
rm -rf target/release

# Remove CUDA PTX outputs
rm -rf target/ptx

# Remove results and logs (if needed)
rm -rf results/
mkdir -p results/logs
```

### Check Disk Usage
```bash
# Size of build directories
du -sh target/ deps/ results/ logs/ venv/

# Total project size
du -sh .

# Git repository size
du -sh .git/
```

---

## Quick Reference Card

```bash
# Working Directory
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH/

# Run Seed Probe
MAX_MINUTES=90 ./tools/run_wr_seed_probe.sh

# View Results
cat results/dsjc1000_seed_probe.jsonl

# Build CUDA Example
cd foundation/prct-core && \
  cargo build --release --features cuda --example world_record_dsjc1000

# Check Git Status
git status --short

# Switch Branches
git checkout gpu-quantum-acceleration          # Main GPU work
git checkout chore/gitignore-artifact-cleanup   # Cleanup changes

# Merge Cleanup Locally
git checkout gpu-quantum-acceleration && \
  git merge chore/gitignore-artifact-cleanup
```

---

**Last Updated**: 2025-11-02
**Branch**: `chore/gitignore-artifact-cleanup`
**Commit**: `9420ff4`
