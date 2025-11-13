# PRISM Quick Run Commands

## 🚀 Quick Reference

### Local Machine (RTX 5070 or similar)

#### Build and Run Locally
```bash
# Navigate to PRISM directory
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Build with CUDA support
cargo build --release --features cuda

# Quick test (5-10 minutes)
PRISM_DEVICE_PROFILE=rtx5070 \
./target/release/examples/world_record_dsjc1000 \
  foundation/prct-core/configs/quick_test.v1.1.toml

# World record attempt (60+ minutes)
PRISM_DEVICE_PROFILE=rtx5070 \
./target/release/examples/world_record_dsjc1000 \
  foundation/prct-core/configs/world_record.v1.toml

# With custom graph file
PRISM_DEVICE_PROFILE=rtx5070 \
./target/release/examples/world_record_dsjc1000 \
  foundation/prct-core/configs/world_record.v1.toml \
  benchmarks/dimacs/DSJC1000.5.col
```

#### Using the Multi-GPU Launcher Script
```bash
# Auto-detects GPU count and selects profile
./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml

# Specify graph file
./run-prism-gpu.sh \
  foundation/prct-core/configs/world_record.v1.toml \
  benchmarks/dimacs/DSJC1000.5.col

# Override profile
PRISM_DEVICE_PROFILE=rtx5070 \
./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml
```

---

## 🐳 Docker (Local or RunPod)

### Pull Image
```bash
docker pull delfictus/prism-ai-world-record:latest
```

### Run with Docker

#### Quick Test (5-10 minutes)
```bash
docker run --gpus all -it --rm \
  delfictus/prism-ai-world-record:latest \
  ./target/release/examples/world_record_dsjc1000 \
  foundation/prct-core/configs/quick_test.v1.1.toml
```

#### With Volume Mounts (Recommended)
```bash
# Create local directories
mkdir -p qtables results

# Run with persistence
docker run --gpus all -it --rm \
  -v $(pwd)/qtables:/app/fluxnet_cache \
  -v $(pwd)/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml
```

#### Interactive Mode (Explore and Run)
```bash
docker run --gpus all -it \
  -v $(pwd)/qtables:/app/fluxnet_cache \
  -v $(pwd)/results:/app/results \
  delfictus/prism-ai-world-record:latest

# Inside container, you'll see GPU info and available commands
# Then run:
./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml
```

---

## ☁️ RunPod Deployment

### Step 1: SSH into RunPod Pod
```bash
# Get SSH command from RunPod console (looks like):
ssh root@X.X.X.X -p XXXXX -i ~/.ssh/runpod_key
```

### Step 2: Pull Docker Image
```bash
docker pull delfictus/prism-ai-world-record:latest
```

### Step 3: Run PRISM

#### Quick Test (8× B200)
```bash
docker run --gpus all -it --rm \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/quick_test.v1.1.toml
```

#### World Record Attempt (72h, 8× B200)
```bash
docker run --gpus all -it \
  --name prism-wr \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/runpod_8gpu.v1.1.toml
```

#### Adaptive RL with Persistence (48h)
```bash
docker run --gpus all -it \
  --name prism-adaptive \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/wr_adaptive_rl.v1.1.toml
```

#### Using RunPod Launcher Script
```bash
docker run --gpus all -it \
  --name prism-training \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./runpod-launch.sh \
    benchmarks/dimacs/DSJC1000.5.col \
    foundation/prct-core/configs/runpod_8gpu.v1.1.toml
```

---

## 📊 Monitoring Commands

### GPU Utilization

#### Real-time Monitoring
```bash
# Watch all GPUs
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet -c 100

# Inside Docker container
./monitor_gpus.sh
```

#### GPU Summary
```bash
# List all GPUs
nvidia-smi --list-gpus

# Query specific info
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# Per-GPU memory usage
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

### PRISM Telemetry

#### View Live Telemetry
```bash
# All telemetry
tail -f results/telemetry.jsonl

# Pretty print with jq
tail -f results/telemetry.jsonl | jq '.'

# Multi-GPU specific
tail -f results/telemetry.jsonl | jq -c 'select(.replica_id)'

# Filter by phase
tail -f results/telemetry.jsonl | jq 'select(.phase == "2")'

# FluxNet metrics only
tail -f results/telemetry.jsonl | jq 'select(.fluxnet)'
```

#### Check Multi-GPU Summary
```bash
# View summary file (created after run)
cat results/multi_gpu_summary.json | jq '.'

# Check device assignments
cat results/multi_gpu_summary.json | jq '.replica_assignments'

# View coordination metrics
cat results/multi_gpu_summary.json | jq '.coordination_metrics'
```

### Logs

#### View Logs
```bash
# Latest log
tail -f results/logs/run_*.log

# Search for errors
grep -i error results/logs/run_*.log

# Search for best results
grep -E "BEST|chromatic.*[0-9]+" results/logs/run_*.log | tail -20

# Phase transitions
grep "PHASE" results/logs/run_*.log
```

---

## 🔧 Configuration Options

### Select Different GPU Profiles

```bash
# Single GPU (RTX 5070)
PRISM_DEVICE_PROFILE=rtx5070 ./run-prism-gpu.sh <config>

# 2× B200
PRISM_DEVICE_PROFILE=runpod_b200_2gpu ./run-prism-gpu.sh <config>

# 4× B200
PRISM_DEVICE_PROFILE=runpod_b200_4gpu ./run-prism-gpu.sh <config>

# 8× B200
PRISM_DEVICE_PROFILE=runpod_b200 ./run-prism-gpu.sh <config>
```

### Specify GPU Devices

```bash
# Use only GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./run-prism-gpu.sh <config>

# Use only GPU 3
CUDA_VISIBLE_DEVICES=3 ./run-prism-gpu.sh <config>

# All GPUs (default)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run-prism-gpu.sh <config>
```

### Thread Configuration

```bash
# Custom thread counts
OMP_NUM_THREADS=64 RAYON_NUM_THREADS=64 ./run-prism-gpu.sh <config>

# Auto-detect (default)
./run-prism-gpu.sh <config>
```

---

## 📁 Available Configurations

### Quick Tests (5-10 minutes)
```bash
foundation/prct-core/configs/quick_test.v1.1.toml
```

### World Record Attempts
```bash
# Single GPU optimized
foundation/prct-core/configs/world_record.v1.toml

# 8× GPU optimized (RunPod)
foundation/prct-core/configs/runpod_8gpu.v1.1.toml
```

### FluxNet Adaptive RL
```bash
# With Q-table persistence
foundation/prct-core/configs/wr_adaptive_rl.v1.1.toml
```

### Benchmark Graphs
```bash
# DSJC1000.5 (most common, world record target: 83 colors)
benchmarks/dimacs/DSJC1000.5.col

# Other graphs
benchmarks/dimacs/DSJC500.5.col
benchmarks/dimacs/DSJC250.5.col
```

---

## 🎯 Complete Example Workflows

### Workflow 1: Local Quick Test
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Build
cargo build --release --features cuda

# Run quick test
PRISM_DEVICE_PROFILE=rtx5070 \
./target/release/examples/world_record_dsjc1000 \
  foundation/prct-core/configs/quick_test.v1.1.toml

# Monitor (in another terminal)
watch -n 1 nvidia-smi
```

### Workflow 2: Local World Record Attempt
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Run with auto-detection
./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml

# Monitor telemetry
tail -f results/telemetry.jsonl | jq -c '{phase:.phase, colors:.chromatic_number, conflicts:.conflicts}'

# Check progress
grep "chromatic" results/logs/run_*.log | tail -20
```

### Workflow 3: RunPod 8× B200 Training (Long Run)
```bash
# SSH into RunPod
ssh root@X.X.X.X -p XXXXX

# Pull image
docker pull delfictus/prism-ai-world-record:latest

# Start long-running training in screen session
screen -S prism-training

# Run 72-hour world record attempt
docker run --gpus all -it \
  --name prism-wr-72h \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/runpod_8gpu.v1.1.toml

# Detach from screen: Ctrl+A, then D

# Reattach later
screen -r prism-training

# Monitor in another SSH session
docker exec prism-wr-72h tail -f /app/results/telemetry.jsonl | jq -c 'select(.replica_id)'
```

### Workflow 4: Docker Interactive Exploration
```bash
# Start container interactively
docker run --gpus all -it \
  -v $(pwd)/qtables:/app/fluxnet_cache \
  -v $(pwd)/results:/app/results \
  delfictus/prism-ai-world-record:latest

# Inside container, you see GPU detection and available commands
# Explore:
ls foundation/prct-core/configs/
cat foundation/prct-core/configs/device_profiles.toml
nvidia-smi

# Run test
./run-prism-gpu.sh foundation/prct-core/configs/quick_test.v1.1.toml

# Exit when done
exit
```

---

## 🛠️ Troubleshooting Commands

### Verify GPU Detection
```bash
# Check CUDA
nvcc --version

# List GPUs
nvidia-smi --list-gpus

# Test CUDA availability
nvidia-smi

# Check CUDA visible devices
echo $CUDA_VISIBLE_DEVICES
```

### Verify PRISM Setup
```bash
# Check binary exists
ls -lh target/release/examples/world_record_dsjc1000

# Test device profile loading
cat foundation/prct-core/configs/device_profiles.toml

# Verify current profile
echo $PRISM_DEVICE_PROFILE
```

### Check Build
```bash
# Build with verbose output
cargo build --release --features cuda --verbose

# Check for CUDA feature
cargo build --release --features cuda 2>&1 | grep -i cuda

# Run tests
cargo test --release --features cuda
```

### Docker Troubleshooting
```bash
# List running containers
docker ps

# Check container logs
docker logs <container-id>

# Attach to running container
docker exec -it <container-id> bash

# Stop container
docker stop <container-id>

# Remove all stopped containers
docker container prune
```

---

## 📥 Download Results from RunPod

### Method 1: SCP (Recommended)
```bash
# From your local machine
scp -P <port> -r root@<runpod-ip>:/workspace/qtables ./trained-qtables
scp -P <port> -r root@<runpod-ip>:/workspace/results ./results
```

### Method 2: Tar and Download
```bash
# On RunPod, create archive
cd /workspace
tar -czf prism-results.tar.gz qtables/ results/

# Download from local machine
scp -P <port> root@<runpod-ip>:/workspace/prism-results.tar.gz ./

# Extract locally
tar -xzf prism-results.tar.gz
```

### Method 3: Docker Copy
```bash
# Copy from running container
docker cp <container-id>:/app/fluxnet_cache ./qtables
docker cp <container-id>:/app/results ./results
```

---

## 🎮 Quick Command Cheat Sheet

```bash
# LOCAL: Quick test
./run-prism-gpu.sh foundation/prct-core/configs/quick_test.v1.1.toml

# LOCAL: World record
./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml

# DOCKER: Quick test
docker run --gpus all -it --rm delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/quick_test.v1.1.toml

# DOCKER: World record with persistence
docker run --gpus all -it \
  -v $(pwd)/qtables:/app/fluxnet_cache \
  -v $(pwd)/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/world_record.v1.toml

# RUNPOD: 8× B200 world record
docker run --gpus all -it \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest \
  ./run-prism-gpu.sh foundation/prct-core/configs/runpod_8gpu.v1.1.toml

# MONITOR: GPU utilization
watch -n 1 nvidia-smi

# MONITOR: Live telemetry
tail -f results/telemetry.jsonl | jq -c 'select(.replica_id)'

# MONITOR: Best results
grep -E "BEST|chromatic" results/logs/run_*.log | tail -20
```

---

## 📚 More Information

- **Multi-GPU Guide**: `README_MULTI_GPU.md`
- **Quick Start**: `MULTI_GPU_QUICK_START.md`
- **RunPod Access**: `RUNPOD_ACCESS_GUIDE.md`
- **Deployment Status**: `FINAL_DEPLOYMENT_STATUS.md`
- **Docker Deployment**: `DOCKER_DEPLOYMENT_COMPLETE.md`

---

**Need help?** All scripts include `--help` option:
```bash
./run-prism-gpu.sh --help
./runpod-launch.sh --help
```
