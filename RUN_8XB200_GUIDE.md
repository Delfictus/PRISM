# Quick Start Guide: Running PRISM on 8x B200 GPUs

## Prerequisites

- RunPod instance with 8x NVIDIA B200 GPUs (180GB VRAM each)
- CUDA 12.8+ installed
- PRISM repository cloned

## Step 1: Verify GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

**Expected output**:
```
0, NVIDIA B200, 180000 MiB
1, NVIDIA B200, 180000 MiB
2, NVIDIA B200, 180000 MiB
3, NVIDIA B200, 180000 MiB
4, NVIDIA B200, 180000 MiB
5, NVIDIA B200, 180000 MiB
6, NVIDIA B200, 180000 MiB
7, NVIDIA B200, 180000 MiB
```

## Step 2: Build PRISM with CUDA

```bash
cd /workspace/PRISM-FINNAL-PUSH
cargo build --release --features cuda
```

**Build time**: ~10-15 minutes

## Step 3: Verify Multi-GPU Config

```bash
cat foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml | grep -A 6 "\[multi_gpu\]"
```

**Expected output**:
```toml
[multi_gpu]
enabled = true
num_gpus = 8
devices = [0, 1, 2, 3, 4, 5, 6, 7]
enable_peer_access = true
enable_nccl = false
strategy = "distributed_phases"
```

## Step 4: Run World Record Pipeline

```bash
./target/release/prism-runner \
  --graph graphs/DSJC1000.5.col \
  --config foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml \
  --output results/dsjc1000_8xb200_$(date +%Y%m%d_%H%M%S).json \
  --telemetry results/telemetry_8xb200_$(date +%Y%m%d_%H%M%S).json
```

## Step 5: Monitor Execution

### Watch GPU Utilization

In a separate terminal:
```bash
watch -n 1 nvidia-smi
```

**Expected during Phase 2 (Thermodynamic)**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54       Driver Version: 535.54       CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA B200         On   | 00000000:00:04.0 Off |                    0 |
| 45%   65C    P2   450W / 600W |  45678MiB / 180000MiB |     98%      Default |
|   1  NVIDIA B200         On   | 00000000:00:05.0 Off |                    0 |
| 46%   66C    P2   455W / 600W |  45892MiB / 180000MiB |     99%      Default |
...
```

### Watch Log Output

```bash
tail -f results/dsjc1000_8xb200_*.log
```

**Key milestones to watch for**:

```
[PIPELINE][INIT] Multi-GPU mode enabled: 8 devices
[MULTI-GPU][INIT] ✅ Device pool ready with 8 GPUs

[PHASE 0] Initial coloring: 127 colors

[PHASE 1] Transfer Entropy ordering complete

[PHASE 2][MULTI-GPU] Using 8 GPUs for distributed thermodynamic
[THERMO-MULTI-GPU] Distributing 10000 replicas across 8 GPUs
[THERMO-GPU-0] Starting 1250 replicas, 250 temps
...
[THERMO-MULTI-GPU] ✅ Gathered 2000 total solutions from 8 GPUs
[PHASE 2] 🎯 Thermodynamic improvement: 127 → 94 colors

[PHASE 3] Quantum-Classical Hybrid Feedback Loop
[PHASE 3] 🎯 Quantum breakthrough: 94 → 88 colors

[PIPELINE] 🎯 FINAL RESULT: 86 colors (conflicts=0)
```

## Step 6: Verify Results

```bash
cat results/dsjc1000_8xb200_*.json | jq '.final_result'
```

**Expected output**:
```json
{
  "chromatic_number": 86,
  "conflicts": 0,
  "quality_score": 0.95,
  "computation_time_ms": 9123456,
  "colors": [0, 1, 2, ..., 85]
}
```

## Step 7: Check Multi-GPU Telemetry

```bash
cat results/telemetry_8xb200_*.json | jq '.phase_gpu_status'
```

**Expected output**:
```json
{
  "phase0_gpu_used": true,
  "phase1_gpu_used": true,
  "phase2_gpu_used": true,
  "phase3_gpu_used": true,
  "phase0_fallback_reason": null,
  "phase1_fallback_reason": null,
  "phase2_fallback_reason": null,
  "phase3_fallback_reason": null
}
```

## Troubleshooting

### Issue: "GPU 0 already in use"

**Solution**: Set `CUDA_VISIBLE_DEVICES`:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./target/release/prism-runner ...
```

### Issue: "Out of memory"

**Check VRAM usage**:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**If > 160GB per GPU**, reduce scale in config:
```toml
[thermo]
replicas = 5000   # Was 10000
num_temps = 1000  # Was 2000
```

### Issue: "Multi-GPU pool initialization failed"

**Check GPU topology**:
```bash
nvidia-smi topo -m
```

**If NVLink not available**, disable peer access:
```toml
[multi_gpu]
enable_peer_access = false  # Uses CPU staging instead
```

### Issue: "Phase 2 fallback to CPU"

**Check logs** for specific error:
```bash
grep "FALLBACK" results/dsjc1000_8xb200_*.log
```

**Common causes**:
- Insufficient VRAM (check `nvidia-smi`)
- PTX kernel load failure (check `target/ptx/thermodynamic.ptx` exists)
- Device initialization failure (check CUDA driver version)

## Performance Tuning

### Increase Thermodynamic Exploration

```toml
[thermo]
replicas = 15000      # More replicas
num_temps = 3000      # Finer temperature ladder
steps_per_temp = 30000  # Longer equilibration
```

### Increase Quantum Attempts

```toml
[quantum]
attempts = 120000     # More attempts (15K per GPU)
depth = 25            # Deeper annealing
```

### Adjust CPU Threads

```toml
[cpu]
threads = 192  # Match your vCPU count
```

## Expected Runtime

- **Phase 0 (Initial)**: ~2 minutes
- **Phase 1 (Transfer Entropy)**: ~5 minutes
- **Phase 2 (Thermodynamic)**: ~60 minutes ← **Multi-GPU accelerated**
- **Phase 3 (Quantum)**: ~90 minutes ← **Multi-GPU ready**
- **Phase 4 (Memetic)**: ~30 minutes
- **Total**: ~3 hours per run

## Success Criteria

✅ **Multi-GPU active**: All 8 GPUs show 98-100% utilization during Phase 2
✅ **No CPU fallbacks**: `phase2_fallback_reason: null` in telemetry
✅ **Chromatic reduction**: Final result < 90 colors
✅ **Zero conflicts**: `conflicts: 0` in final result

## Next Steps

1. **If result ≥ 90 colors**: Run again with different seed
2. **If result 85-89 colors**: Run 5-10 more times for best
3. **If result ≤ 84 colors**: 🎉 **World record candidate!** Verify solution

## Support

**Documentation**: See `/MULTI_GPU_B200_IMPLEMENTATION_REPORT.md`
**Configuration**: See `/foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml`
**Source code**: See `/foundation/prct-core/src/gpu_*_multi.rs`

---

**Quick Reference Command**:
```bash
./target/release/prism-runner \
  --graph graphs/DSJC1000.5.col \
  --config foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml \
  --output results/run_$(date +%s).json \
  --telemetry results/telem_$(date +%s).json \
  2>&1 | tee results/log_$(date +%s).txt
```
