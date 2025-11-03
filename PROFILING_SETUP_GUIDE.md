# Complete Profiling Setup for PRISM-AI
## Using Docker Image + Nsight Systems + Nsight Compute

**Docker Image:** `delfictus/prism-ai-world-record:latest`
**Target Hardware:** RTX 5070 (sm_89)
**Goal:** Baseline unfused kernel performance before implementing fusion

---

## Quick Start

```bash
# 1. Pull the profiling scripts
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# 2. Run automated profiling
./scripts/profile_all_pipelines.sh

# 3. Wait for completion (~15-30 minutes)

# 4. View results
./scripts/analyze_profiles.sh
```

---

## Prerequisites

### 1. Install Nsight Tools (If Not Already Installed)

```bash
# Check if already installed
which nsys
which ncu

# If not installed, download from NVIDIA
# CUDA Toolkit 12.0+ includes both tools
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run --toolkit --silent
```

### 2. Setup Reports Directory

```bash
mkdir -p reports/{nsys,ncu,analysis,csv}
mkdir -p reports/nsys/{active_inference,transfer_entropy,linear_algebra,quantum,neuromorphic}
mkdir -p reports/ncu/{active_inference,transfer_entropy,linear_algebra,quantum,neuromorphic}
```

### 3. Extract Binary from Docker

Since you're using the Docker image, we need to extract the binary for local profiling:

```bash
# Extract the world_record binary
docker create --name tmp delfictus/prism-ai-world-record:latest
docker cp tmp:/usr/local/bin/world_record ./target/release/prism-ai
docker rm tmp

# Make executable
chmod +x ./target/release/prism-ai

# Verify
./target/release/prism-ai --help
```

**Alternative:** Build locally with release profile:
```bash
cargo build --release --features cuda,profiling
```

---

## Profiling Strategy

### Phase 1: Timeline Analysis (Nsight Systems)

**Purpose:** Identify hot paths, kernel sequences, CPU-GPU gaps

**Command template:**
```bash
nsys profile \
    --stats=true \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --trace=cuda,nvtx,osrt \
    -o reports/nsys/<pipeline>/<pipeline> \
    ./target/release/prism-ai --pipeline <pipeline>
```

### Phase 2: Kernel-Level Analysis (Nsight Compute)

**Purpose:** Memory bandwidth, occupancy, tensor core usage

**Command template:**
```bash
ncu \
    --set full \
    --kernel-name-base demangled \
    --target-processes all \
    --details-all \
    --csv \
    --log-file reports/ncu/<pipeline>/ncu_<pipeline>.csv \
    ./target/release/prism-ai --pipeline <pipeline>
```

---

## Implementation

### Profiling Scripts

I'll create automated scripts for all pipelines.
