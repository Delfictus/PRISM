# Docker Build & Push Instructions

## Current Status

**Implementation**: ✅ Complete (Option E - 8x B200 Multi-GPU)
**Docker Build**: 🔄 In progress
**Push to Hub**: ⏳ Pending build completion

---

## What's Being Built

**Image**: `delfictus/prism-ai-world-record:8xb200-v1.0`
**Base**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
**Rust**: 1.90.0
**CUDA**: 12.8.1
**Target**: 8x NVIDIA B200 GPUs

---

## Build Progress

The Docker build is currently running and will take approximately:
- **Base image download**: ~5-8 minutes
- **Rust installation**: ~2 minutes
- **PRISM compilation**: ~10-15 minutes
- **Total**: ~15-25 minutes

---

## After Build Completes

### 1. Verify Image

```bash
docker images | grep prism-ai-world-record
```

Expected output:
```
delfictus/prism-ai-world-record   8xb200-v1.0    <image_id>   <timestamp>   ~15GB
delfictus/prism-ai-world-record   latest         <image_id>   <timestamp>   ~15GB
```

### 2. Test Locally (Optional)

```bash
docker run --gpus all delfictus/prism-ai-world-record:8xb200-v1.0
```

Should show help menu with available commands.

### 3. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push versioned tag
docker push delfictus/prism-ai-world-record:8xb200-v1.0

# Push latest tag
docker push delfictus/prism-ai-world-record:latest
```

---

## Image Contents

### Binaries
- `/workspace/prism/target/release/examples/world_record_dsjc1000` - Main pipeline
- `/workspace/prism/target/release/libprct_core.so` - Core library

### Scripts
- `/workspace/prism/run_8gpu_world_record.sh` - Launch 8-GPU run
- `/workspace/prism/monitor_gpus.sh` - GPU monitoring
- `/workspace/prism/entrypoint.sh` - Container entrypoint

### Configuration
- `/workspace/prism/foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml` - 8x B200 config
- All other configs available in `configs/` directory

### Graph Data
- `/workspace/prism/benchmarks/dimacs/DSJC1000.5.col` - Target graph

---

## RunPod Deployment

### 1. Create Instance
- GPU: 8x NVIDIA B200
- VRAM: 1440GB (180GB × 8)
- RAM: 3000GB+
- vCPUs: 192+
- Disk: 100GB+

### 2. Pull Image
```bash
docker pull delfictus/prism-ai-world-record:8xb200-v1.0
```

### 3. Launch World Record Attempt
```bash
docker run --gpus all \
  -v $(pwd)/results:/workspace/prism/results \
  delfictus/prism-ai-world-record:8xb200-v1.0 \
  ./run_8gpu_world_record.sh
```

### 4. Monitor Progress
```bash
# Get container ID
docker ps

# Monitor GPUs
docker exec -it <container_id> ./monitor_gpus.sh

# Tail logs
docker logs -f <container_id>
```

---

## Expected Output

### Startup
```
╔══════════════════════════════════════════════════════════════════╗
║   PRISM AI World Record - 8x B200 Multi-GPU Execution           ║
╚══════════════════════════════════════════════════════════════════╝

🔍 Detecting GPUs...
   Found: 8 GPUs

📊 GPU Configuration:
   GPU 0: NVIDIA B200 | VRAM: 184320 MiB | Compute: 9.0
   GPU 1: NVIDIA B200 | VRAM: 184320 MiB | Compute: 9.0
   ...
   GPU 7: NVIDIA B200 | VRAM: 184320 MiB | Compute: 9.0

🎯 Configuration:
   Config: foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml
   Target: 83 colors (world record)
```

### During Execution
```
[PIPELINE][INIT] Multi-GPU pool: 8 devices
[PHASE 2][MULTI-GPU] Distributing 10000 replicas across 8 GPUs
[THERMO-GPU-0] Starting 1250 replicas, 250 temps
[THERMO-GPU-1] Starting 1250 replicas, 250 temps
...
[THERMO-GPU-7] Starting 1250 replicas, 250 temps
```

### GPU Utilization
```
GPU 0 | Util: 98% | Mem: 108 / 180 GB | Temp: 65C | Power: 450W
GPU 1 | Util: 97% | Mem: 110 / 180 GB | Temp: 66C | Power: 455W
...
GPU 7 | Util: 99% | Mem: 112 / 180 GB | Temp: 64C | Power: 448W
```

---

## Troubleshooting

### Build Failed
Check full log:
```bash
cat /tmp/docker_build_final.log
```

### Image Too Large
Expected size: ~15-20GB (includes Rust toolchain + PRISM + CUDA libs)

If size is concern:
- Use multi-stage build (builder + runtime)
- Strip debug symbols
- Remove unnecessary dependencies

### CUDA Version Mismatch
RunPod base has CUDA 12.8.1, ensure B200 drivers support it.

---

## Cost Estimation

**RunPod 8x B200**: ~$60-80/hour

### Single World Record Attempt
- Duration: 7.5 hours (3 iterative passes)
- Cost: ~$450-600

### Expected Total (5-10 attempts)
- Duration: 37-75 hours
- Cost: ~$2,220-6,000

Budget accordingly and use spot instances if available (50-70% discount).

---

## Files in This Release

- `Dockerfile` - RunPod deployment image
- `.dockerignore` - Faster builds
- `RUNPOD_8XB200_DEPLOYMENT_GUIDE.md` - Full deployment guide
- `OPTION_E_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `DOCKER_BUILD_INSTRUCTIONS.md` - This file

---

## Support

**Build Issues**: Check `/tmp/docker_build_final.log`
**Runtime Issues**: Check container logs with `docker logs <container_id>`
**GPU Issues**: Run `nvidia-smi` inside container

**GitHub**: https://github.com/Delfictus/PRISM
**Docker Hub**: https://hub.docker.com/r/delfictus/prism-ai-world-record
