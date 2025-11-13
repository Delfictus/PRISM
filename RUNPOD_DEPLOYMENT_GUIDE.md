# 🚀 PRISM RunPod Deployment Guide - 8x B200 GPU Optimized

Complete guide to deploying PRISM on RunPod with 8x NVIDIA B200 GPUs (1440 GB VRAM).

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Building the Docker Image](#building-the-docker-image)
4. [Deploying to RunPod](#deploying-to-runpod)
5. [Running World Record Attempts](#running-world-record-attempts)
6. [Monitoring & Results](#monitoring--results)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Prerequisites

### Local Machine (for building)
- Docker installed and running
- Docker Hub account
- Git (to clone repository)
- 50+ GB free disk space for build

### RunPod Account
- RunPod account with credits
- Access to 8x B200 GPU pod (1440 GB VRAM total)
- Minimum 100 GB disk space
- CUDA 12.6+ runtime

---

## Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/PRISM.git
cd PRISM
git checkout claude/fluxnet-rl-gpu-implementation-011CUzcMsiXXsaMumMmenNtx
```

### 2️⃣ Build Docker Image
```bash
# Set your Docker Hub username
export DOCKER_USER=yourusername

# Build and push (takes 30-60 minutes)
chmod +x docker/build-runpod.sh
./docker/build-runpod.sh
```

### 3️⃣ Deploy to RunPod

**Option A: RunPod Web UI**
1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" → "Custom Template"
3. Configure:
   - **Container Image**: `yourusername/prism-runpod:1.1.0`
   - **GPU Type**: 8x B200 (or 8x H100)
   - **GPU Count**: 8
   - **Container Disk**: 100 GB
   - **Volume Disk**: 200 GB (for results)
   - **Ports**: 8080 (TCP)
4. Click "Deploy"

**Option B: RunPod CLI**
```bash
# Install RunPod CLI
pip install runpod

# Configure API key
runpod config

# Deploy pod
runpod create pod \
  --name "PRISM-WR-8xB200" \
  --image "yourusername/prism-runpod:1.1.0" \
  --gpu-type "NVIDIA B200" \
  --gpu-count 8 \
  --container-disk-size 100 \
  --volume-size 200
```

---

## Building the Docker Image

### Manual Build (alternative to script)

```bash
# Navigate to project root
cd PRISM

# Build for B200 (sm_100 architecture)
docker build \
  -f Dockerfile.runpod \
  -t yourusername/prism-runpod:1.1.0 \
  --build-arg CUDA_ARCH=sm_100 \
  --build-arg NUM_BUILD_JOBS=64 \
  .

# Tag as latest
docker tag yourusername/prism-runpod:1.1.0 yourusername/prism-runpod:latest

# Login to Docker Hub
docker login

# Push to registry
docker push yourusername/prism-runpod:1.1.0
docker push yourusername/prism-runpod:latest
```

### Verify Build
```bash
# Check image size
docker images yourusername/prism-runpod:1.1.0

# Test locally (if you have GPU)
docker run --rm --gpus all yourusername/prism-runpod:1.1.0 nvidia-smi
```

---

## Deploying to RunPod

### Using RunPod Template

1. **Create Template**
   - Name: `PRISM 8xB200 World Record`
   - Container Image: `yourusername/prism-runpod:1.1.0`
   - Docker Command: `/app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml`
   - Container Disk: `100 GB`
   - Volume Mount: `/workspace`
   - Expose Ports: `8080`

2. **Pod Configuration**
   - GPU: `8x NVIDIA B200`
   - vCPU: `288`
   - RAM: `2264 GB`
   - Persistent Volume: `200 GB` (for results/cache)

3. **Environment Variables**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   OMP_NUM_THREADS=288
   RAYON_NUM_THREADS=288
   RUST_LOG=info
   RUST_BACKTRACE=1
   ```

### SSH Access (for monitoring)

```bash
# Get pod SSH credentials from RunPod console
ssh root@X.X.X.X -p XXXXX -i ~/.ssh/runpod_key

# Once connected
cd /workspace
docker ps  # Check running containers
nvidia-smi  # Check GPU usage
```

---

## Running World Record Attempts

### Default Run (72 hours max)

The container starts automatically with the optimized config:
```bash
/app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml
```

### Custom Run

SSH into pod and run custom config:

```bash
# Copy custom config to volume
docker cp my_config.toml prism-container:/app/configs/

# Run with custom config
docker exec prism-container /app/bin/world_record_dsjc1000 /app/configs/my_config.toml
```

### Resume from Checkpoint

FluxNet automatically loads saved Q-tables:
```bash
# Check saved files
ls -lh /workspace/prism-cache/

# Files auto-loaded on next run:
# - qtable_final.bin
# - adaptive_indexer_final.bin
```

---

## Monitoring & Results

### Real-time Monitoring

**Option 1: SSH + nvidia-smi**
```bash
ssh root@pod -p port

# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor all 8 GPUs
nvidia-smi dmon -s u -c 1000
```

**Option 2: Docker Logs**
```bash
docker logs -f prism-container

# Look for:
# - [FLUXNET] messages (RL progress)
# - [PHASE X][GPU] (pipeline execution)
# - {"event":"final_result",...} (JSON results)
```

**Option 3: RunPod Web Console**
- Navigate to pod → Logs
- View real-time stdout/stderr

### Results Location

All results saved to `/workspace/prism-results/`:
```bash
/workspace/
├── prism-results/       # Coloring results, metrics
├── prism-cache/         # FluxNet Q-tables, indexers
└── prism-logs/          # Execution logs
```

### Download Results

```bash
# From RunPod SSH
cd /workspace/prism-results
tar -czf results.tar.gz *

# Download via RunPod web console or scp
scp -P port root@pod:/workspace/prism-results/results.tar.gz ./
```

---

## Troubleshooting

### Issue: Out of Disk Space

**Symptom**: Build fails with "no space left on device"

**Solution**:
```bash
# Clean Docker cache
docker system prune -a --volumes

# Increase RunPod disk allocation to 150 GB
```

### Issue: Only 1 GPU Detected

**Symptom**: nvidia-smi shows only 1 GPU

**Solution**:
```bash
# Verify all GPUs visible
nvidia-smi --list-gpus

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES  # Should be: 0,1,2,3,4,5,6,7

# Restart container with --gpus all
docker run --rm --gpus all ...
```

### Issue: CUDA Out of Memory

**Symptom**: "CUDA error: out of memory"

**Solution**:
```bash
# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Reduce batch sizes in config:
# [gpu]
# batch_size = 4096  # Lower if needed

# [thermo]
# batch_size = 2048  # Lower if needed
```

### Issue: Slow Performance

**Symptom**: GPU utilization < 50%

**Solution**:
```bash
# Check CPU bottleneck
htop

# Verify thread count
echo $OMP_NUM_THREADS  # Should be 288
echo $RAYON_NUM_THREADS  # Should be 288

# Check GPU streams
# Ensure config has:
# [gpu]
# streams = 8  # Per GPU = 64 total
```

---

## Performance Optimization

### Expected Performance (8x B200)

| Metric | Value |
|--------|-------|
| Total VRAM | 1440 GB |
| VRAM per GPU | 180 GB |
| CPU Threads | 288 vCPUs |
| GPU Streams | 64 (8 per GPU) |
| Q-table Size | 65K states |
| Replay Buffer | 8K experiences |
| Quantum Depth | 10 |
| Memetic Population | 2048 |
| Thermodynamic Replicas | 96 (12 per GPU) |

### GPU Utilization Targets

- **Phase 0 (Reservoir)**: 60-80% (neuromorphic computation)
- **Phase 1 (Transfer Entropy)**: 70-90% (TE + Active Inference)
- **Phase 2 (Thermodynamic)**: 80-95% (parallel tempering)
- **Phase 3 (Quantum)**: 85-98% (QUBO solving)

### Cost Estimation

RunPod pricing (approximate):
- **8x B200**: ~$40-60/hour
- **72-hour run**: ~$2,880 - $4,320
- **Recommended**: Start with 6-hour test run (~$300)

### Resource Allocation Strategy

```toml
# Distribute 8 GPUs across phases:
# GPU 0: Orchestrator + Phase 0 (Reservoir)
# GPU 1-2: Phase 1 (TE + Active Inference)
# GPU 3-6: Phase 2 (Thermodynamic - 48 replicas)
# GPU 7: Phase 3 (Quantum)
```

---

## Advanced Configuration

### Multi-Instance Training

Run 4 independent instances (2 GPUs each):

```bash
# Instance 1 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 /app/bin/world_record_dsjc1000 config1.toml &

# Instance 2 (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 /app/bin/world_record_dsjc1000 config2.toml &

# Instance 3 (GPUs 4-5)
CUDA_VISIBLE_DEVICES=4,5 /app/bin/world_record_dsjc1000 config3.toml &

# Instance 4 (GPUs 6-7)
CUDA_VISIBLE_DEVICES=6,7 /app/bin/world_record_dsjc1000 config4.toml &
```

### Persistent Q-Table Training

Train Q-table across multiple runs:

```bash
# Run 1: Initial training (24h)
docker run ... /app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml

# Run 2: Continue training with saved Q-table (24h)
# Q-table automatically loaded from /app/fluxnet_cache/qtable_final.bin
docker run ... /app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml

# Run 3: Exploitation phase (epsilon_min=0.0 in config)
# Uses learned Q-table for pure exploitation
```

---

## Support & Resources

- **GitHub Issues**: https://github.com/yourusername/PRISM/issues
- **RunPod Docs**: https://docs.runpod.io/
- **CUDA Docs**: https://docs.nvidia.com/cuda/

---

## License

PRISM is licensed under [LICENSE]. See LICENSE file for details.

---

**🎯 Target**: DSJC1000.5 world record (<83 colors)
**⚡ Hardware**: 8x NVIDIA B200 (1440 GB VRAM)
**🧠 RL**: Adaptive percentile indexing + prioritized replay
**💾 Persistence**: Auto-save every 5 temps + final checkpoint

Good luck with your world record attempt! 🚀
