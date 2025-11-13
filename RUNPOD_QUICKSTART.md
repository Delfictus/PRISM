# 🚀 PRISM RunPod Quick Start - 8x B200

Get PRISM running on RunPod in < 5 minutes.

---

## 📦 Step 1: Build & Push Image (One Time)

```bash
# Login to Docker Hub
docker login

# Build and push (30-60 min)
./docker/build-runpod.sh

# Image will be pushed to: delfictus/prism-ai-world-record:latest
```

---

## 🎯 Step 2: Deploy on RunPod

### Option A: Web UI (Easiest)

1. Go to https://www.runpod.io/console/pods
2. Click **"+ GPU Pod"**
3. Select:
   - GPU: **8x B200** (or 8x H100 if B200 unavailable)
   - Template: **RunPod PyTorch 2.4.0**
4. Click **"Customize Deployment"**
5. Change:
   - **Container Image**: `delfictus/prism-ai-world-record:latest`
   - **Docker Command**: Leave empty for terminal mode
   - **Container Disk**: `100 GB`
   - **Volume**: `200 GB`
   - **Expose HTTP Port**: `8080`
6. Click **"Deploy"**
7. Wait 2-3 minutes for pod to start
8. Click **"Connect"** → **"Start Web Terminal"** or **"Connect via SSH"**

### Option B: RunPod CLI

```bash
pip install runpod

runpod config  # Enter API key

runpod create pod \
  --name "PRISM-WR" \
  --image "delfictus/prism-ai-world-record:latest" \
  --gpu-type "NVIDIA B200" \
  --gpu-count 8 \
  --container-disk-size 100 \
  --volume-size 200 \
  --ports "8080/http"
```

---

## 🖥️ Step 3: Use Terminal (Quick Testing)

Once pod is running, open terminal (Web Terminal or SSH):

```bash
# You'll see welcome screen with GPU info

# Quick test (5-10 minutes):
prism-quick

# World record attempt (72h max):
prism-wr

# Adaptive RL with persistence:
prism-adaptive

# Check GPUs:
gpus

# Watch GPU utilization:
gpu-watch

# Custom config:
/app/bin/world_record_dsjc1000 /app/configs/YOUR_CONFIG.toml
```

---

## 📊 Available Commands

| Command | Description | Time |
|---------|-------------|------|
| `prism-quick` | Quick test run | 5-10 min |
| `prism-wr` | Full world record (8GPU config) | 72h max |
| `prism-adaptive` | Adaptive RL with FluxNet | 48h max |
| `prism-results` | List results | - |
| `prism-cache` | List RL cache (Q-tables) | - |
| `gpus` | Show GPU status | - |
| `gpu-watch` | Live GPU monitoring | - |

---

## 📁 File Locations

```
/app/
├── bin/
│   └── world_record_dsjc1000          # Main binary
├── configs/
│   ├── runpod_8gpu.v1.1.toml          # 8-GPU optimized
│   ├── wr_adaptive_rl.v1.1.toml       # Adaptive RL
│   └── quick.v1.1.toml                # Quick test
├── results/                            # Coloring results
├── fluxnet_cache/                      # Q-tables & indexers
│   ├── qtable_final.bin
│   └── adaptive_indexer_final.bin
└── logs/                               # Execution logs
```

---

## 🔄 Resume from Checkpoint

Q-tables auto-load on next run:

```bash
# First run (trains Q-table)
prism-adaptive

# Second run (loads saved Q-table)
prism-adaptive  # Continues learning!
```

---

## 💰 Cost Estimate

| Duration | 8x B200 Cost |
|----------|--------------|
| 1 hour | ~$50 |
| 6 hours | ~$300 |
| 24 hours | ~$1,200 |
| 72 hours | ~$3,600 |

**Tip**: Start with `prism-quick` (< $5) to verify setup.

---

## 🛠️ Troubleshooting

**Issue**: Only 1 GPU showing
```bash
nvidia-smi --list-gpus  # Should show 8
echo $CUDA_VISIBLE_DEVICES  # Should be 0,1,2,3,4,5,6,7
```

**Issue**: Command not found
```bash
source /root/.bashrc  # Reload aliases
```

**Issue**: Out of memory
```bash
# Edit config: reduce batch_size
vim /app/configs/runpod_8gpu.v1.1.toml
```

---

## 📈 Monitor Progress

```bash
# Watch logs
tail -f /app/logs/prism.log

# Check results
ls -lh /app/results/

# View FluxNet stats
ls -lh /app/fluxnet_cache/
```

---

## 📥 Download Results

```bash
# From terminal in pod:
cd /app/results
tar -czf results.tar.gz *

# Download via RunPod web UI:
# Files → /app/results/results.tar.gz → Download
```

---

## 🎯 Expected Results

With 8x B200:
- **Target**: DSJC1000.5 < 83 colors
- **GPU Util**: 80-95% across all 8 GPUs
- **Throughput**: ~10-20x faster than single GPU
- **Memory**: ~50-100 GB VRAM per GPU

---

## 🆘 Quick Help

```bash
# All configs
ls /app/configs/

# All binaries
ls /app/bin/

# Check CUDA version
nvcc --version

# Full help
/app/bin/world_record_dsjc1000 --help
```

---

**🎯 Goal**: DSJC1000.5 world record
**⚡ Hardware**: 8x B200 (1440 GB VRAM)
**🧠 Method**: Multi-phase GPU pipeline + Adaptive RL
**💾 Image**: `delfictus/prism-ai-world-record:latest`

Ready to break the world record! 🚀
