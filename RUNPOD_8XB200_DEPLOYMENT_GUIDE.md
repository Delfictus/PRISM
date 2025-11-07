# RunPod 8x B200 Deployment Guide

## Quick Start

### 1. Pull the Docker Image on RunPod

```bash
docker pull delfictus/prism-ai-world-record:8xb200-v1.0
```

### 2. Run World Record Attempt

```bash
docker run --gpus all \
  -v $(pwd)/results:/workspace/prism/results \
  delfictus/prism-ai-world-record:8xb200-v1.0 \
  ./run_8gpu_world_record.sh
```

### 3. Monitor GPU Utilization (In Another Terminal)

```bash
docker exec -it <container_id> ./monitor_gpus.sh
```

---

## What's Included

### Pre-Built Binaries
- `world_record_dsjc1000` - Main pipeline executable
- `prism_config_cli` - Configuration and monitoring tool

### Configuration
- `wr_ultra_8xb200.v1.0.toml` - Ultra-massive 8-GPU config
  - 10,000 thermodynamic replicas (1,250 per GPU)
  - 2,000 temperature points (250 per GPU)
  - 80,000 quantum attempts (10,000 per GPU)
  - 8,000 memetic population (1,000 per GPU)

### Helper Scripts
- `run_8gpu_world_record.sh` - Launch world record attempt
- `monitor_gpus.sh` - Real-time GPU monitoring
- `entrypoint.sh` - Container entrypoint with help

---

## RunPod Instance Specs

**Required**:
- Instance: 8x NVIDIA B200
- VRAM: 1440GB total (180GB per GPU)
- RAM: 3000GB+
- vCPUs: 192+

**Template**: Community Cloud → GPU Instances → 8x B200

---

## Expected Performance

### Parameter Scale
| Metric | Single GPU | 8x B200 Multi-GPU | Scaling |
|--------|------------|-------------------|---------|
| Thermo replicas | 56 | 10,000 | 178x |
| Thermo temps | 56 | 2,000 | 35x |
| Thermo steps | 25M | 400B | 16,000x |
| Quantum attempts | 512 | 80,000 | 156x |
| Quantum depth | 8 | 20 | 2.5x |
| QUBO solves | 23K | 4M | 173x |
| Memetic population | 320 | 8,000 | 25x |

### Wall-Clock Time
- **Single pass**: ~2.5 hours
- **3 iterative passes**: ~7.5 hours
- **Expected result**: 85-92 colors (target: 83)

### VRAM Usage
- **Per GPU**: ~110GB / 180GB (61%)
- **Total**: ~880GB / 1440GB (61%)
- **All 8 GPUs active simultaneously**

---

## Usage Examples

### Basic Run
```bash
docker run --gpus all delfictus/prism-ai-world-record:latest
```
Shows help and available commands.

### World Record Attempt
```bash
docker run --gpus all \
  -v $(pwd)/results:/workspace/prism/results \
  delfictus/prism-ai-world-record:latest \
  ./run_8gpu_world_record.sh
```

### Custom Config
```bash
docker run --gpus all \
  -v $(pwd)/custom_config.toml:/workspace/prism/custom.toml \
  -v $(pwd)/results:/workspace/prism/results \
  delfictus/prism-ai-world-record:latest \
  ./target/release/examples/world_record_dsjc1000 custom.toml
```

### Interactive Shell
```bash
docker run --gpus all -it \
  delfictus/prism-ai-world-record:latest \
  bash
```

### Monitor Running Container
```bash
# Get container ID
docker ps

# Monitor GPUs
docker exec -it <container_id> ./monitor_gpus.sh

# Tail logs
docker exec -it <container_id> tail -f results/logs/*.log
```

---

## Troubleshooting

### Issue: "No CUDA devices found"
**Solution**: Ensure `--gpus all` flag is used and RunPod instance has GPUs allocated.

### Issue: "VRAM allocation failed"
**Solution**: Check GPU memory is not being used by other processes:
```bash
docker exec -it <container_id> nvidia-smi
```

### Issue: Build takes too long
**Solution**: Use pre-built image from Docker Hub instead of building locally.

### Issue: Want to see detailed build progress
```bash
docker logs -f <container_id>
```

---

## Cost Estimation

**RunPod 8x B200 Pricing**: ~$50-100/hour (varies)

### Single Run Cost
- **Duration**: 7.5 hours (3 iterative passes)
- **Cost**: ~$375-750 per attempt
- **Expected attempts to WR**: 5-10 runs
- **Total**: $1,875-7,500

### Optimization
- Start with 1-2 test runs (1-2 hours each) to verify setup
- Then launch full 48-hour run if promising
- Use spot instances if available (50-70% discount)

---

## Monitoring & Logs

### Telemetry Output
Located in: `/workspace/prism/target/run_artifacts/live_metrics_*.jsonl`

### Log Files
Located in: `/workspace/prism/results/logs/run_*.log`

### Checkpoints
Located in: `/workspace/prism/checkpoints/` (saved every 30 minutes)

---

## Next Steps After Deployment

1. **Launch RunPod instance** (8x B200)
2. **Pull Docker image** (`docker pull delfictus/prism-ai-world-record:8xb200-v1.0`)
3. **Run test** (1-hour timeout to verify setup)
4. **Monitor GPUs** (ensure all 8 at 95-100% utilization)
5. **Launch full run** (48-hour maximum)
6. **Collect results** (copy from `/workspace/prism/results/`)

---

## Expected World Record Timeline

**Scenario 1 (Lucky)**: 1-2 runs → 83 colors found → 15-20 hours total

**Scenario 2 (Typical)**: 5-8 runs → 83 colors found → 40-60 hours total

**Scenario 3 (Unlucky)**: 10-15 runs → 83 colors found → 75-112 hours total

**Budget accordingly**: Reserve $500-1000 for RunPod credits.

---

## Support

**Issues**: Check logs in `results/logs/` and `target/run_artifacts/`

**GPU Status**: Run `nvidia-smi` inside container

**Build Logs**: `docker logs <container_id>`

---

## Version Info

- **PRISM Version**: 1.0.0 (Option E Multi-GPU)
- **Docker Image**: delfictus/prism-ai-world-record:8xb200-v1.0
- **CUDA**: 12.8.1
- **Base**: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
- **Target**: 8x NVIDIA B200 (1440GB VRAM)
