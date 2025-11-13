# PRISM Multi-GPU Execution Guide

This guide explains how to run PRISM on both single RTX 5070 workstations and RunPod instances with 1-8× NVIDIA B200 GPUs using the same binary.

## Quick Start

### Local RTX 5070 Workstation

```bash
# Set device profile (optional, auto-detected)
export PRISM_DEVICE_PROFILE=rtx5070

# Run with default config
./run-prism-gpu.sh --config foundation/prct-core/configs/world_record.v1.toml

# Or run directly
cargo run --release --features cuda -- \
  --config foundation/prct-core/configs/world_record.v1.toml \
  graphs/DSJC1000.5.col
```

### RunPod 8× B200 Instance

```bash
# Launch script automatically detects RunPod environment
./runpod-launch.sh graphs/DSJC1000.5.col foundation/prct-core/configs/runpod_8gpu.v1.1.toml

# Or manually specify profile
export PRISM_DEVICE_PROFILE=runpod_b200
./run-prism-gpu.sh --config foundation/prct-core/configs/runpod_8gpu.v1.1.toml
```

## Device Profiles

Device profiles are defined in `foundation/prct-core/configs/device_profiles.toml`.

### Available Profiles

| Profile | Mode | Replicas | Devices | Use Case |
|---------|------|----------|---------|----------|
| `rtx5070` | Local | 1 | cuda:0 | Single RTX 5070 workstation |
| `rtx5070_aggressive` | Local | 4 | cuda:0 | Multiple replicas on single GPU |
| `runpod_b200` | Cluster | Auto | cuda:* | RunPod 8× B200 (auto-detect) |
| `runpod_b200_fixed` | Cluster | 8 | cuda:0-7 | RunPod 8× B200 (explicit) |
| `runpod_b200_4gpu` | Cluster | 4 | cuda:0-3 | RunPod 4× B200 |
| `runpod_b200_2gpu` | Cluster | 2 | cuda:0-1 | RunPod 2× B200 |
| `debug` | Local | 1 | cuda:0 | Debug mode with verbose logging |
| `benchmark` | Local | 8 | cuda:0 | Benchmark single vs multi-GPU |

### Profile Selection Priority

1. CLI argument: `--device-profile <name>`
2. Environment variable: `PRISM_DEVICE_PROFILE=<name>`
3. Auto-detection:
   - If `RUNPOD_GPU_COUNT` set → RunPod profile
   - Otherwise → `rtx5070` (local)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRISM_DEVICE_PROFILE` | Device profile name | auto-detect |
| `PRISM_REPLICAS` | Override replica count | from profile |
| `PRISM_OUTPUT_DIR` | Output directory for results | `./output` |
| `PRISM_TELEMETRY_FILE` | Telemetry JSONL output | `./telemetry.jsonl` |
| `PRISM_MULTI_GPU_SUMMARY` | Multi-GPU summary JSON | `./multi_gpu_summary.json` |
| `RUNPOD_GPU_COUNT` | Number of GPUs (set by RunPod) | N/A |
| `RUNPOD_POD_ID` | Pod identifier (set by RunPod) | N/A |

## Multi-GPU Architecture

### Strategy: Embarrassingly Parallel Replicas

PRISM distributes replicas across GPUs using an embarrassingly parallel strategy:

- **Replica 0 (Primary)**: Runs ALL phases including reservoir/TE
- **Replicas 1-N (Secondary)**: Run only parallel phases (thermo, quantum, memetic)
- **Snapshot Sharing**: Global best solution broadcast every 100ms
- **No Graph Partitioning**: Full graph on each replica (future enhancement)

### Phase Distribution

| Phase | Primary (Replica 0) | Secondary (Replicas 1-N) |
|-------|---------------------|--------------------------|
| 0: Reservoir Prediction | Yes | No |
| 1: Transfer Entropy + AI | Yes | No |
| 2: Thermodynamic | Yes | Yes |
| 3: Quantum Annealing | Yes | Yes |
| 4: Memetic Search | Yes | Yes |

### Performance Scaling

Expected speedup for DSJC1000.5 (1000 vertices, 500k edges):

| GPUs | Replicas | Expected Speedup | Memory per GPU |
|------|----------|------------------|----------------|
| 1 | 1 | 1.0× | ~8 GB |
| 2 | 2 | ~1.7× | ~8 GB |
| 4 | 4 | ~3.2× | ~8 GB |
| 8 | 8 | ~5.5× | ~8 GB |

Sub-linear scaling due to:
- Reservoir/TE serialized on replica 0
- Snapshot sync overhead
- Phase imbalance

## Configuration

### Single GPU (RTX 5070)

```toml
# foundation/prct-core/configs/world_record.v1.toml
[gpu]
device_id = 0
streams = 4
enable_reservoir_gpu = true
enable_thermo_gpu = true
enable_quantum_gpu = true
```

### Multi-GPU (RunPod 8× B200)

```toml
# foundation/prct-core/configs/runpod_8gpu.v1.1.toml
[gpu]
device_id = 0  # Primary device
streams = 4
enable_reservoir_gpu = true
enable_thermo_gpu = true
enable_quantum_gpu = true

[multi_gpu]
enabled = true
num_gpus = 8
devices = [0, 1, 2, 3, 4, 5, 6, 7]
enable_peer_access = true
strategy = "distributed_replicas"
```

## Monitoring and Telemetry

### Real-Time Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor telemetry
tail -f telemetry.jsonl | jq '.'

# Check replica health
grep "REPLICA.*HEARTBEAT" telemetry.jsonl
```

### Multi-GPU Summary

After execution, inspect `multi_gpu_summary.json`:

```json
{
  "profile_name": "runpod_b200",
  "num_replicas": 8,
  "num_devices": 8,
  "global_best_chromatic": 83,
  "global_best_conflicts": 0,
  "total_runtime_ms": 45123,
  "parallel_speedup": 5.2,
  "replicas": [
    {
      "replica_id": 0,
      "device_id": 0,
      "phases_executed": 25,
      "best_chromatic": 83,
      "is_healthy": true
    },
    ...
  ]
}
```

## Troubleshooting

### Issue: Profile not found

```
Error: Device profile 'runpod_b200' not found
```

**Solution**: Check available profiles:

```bash
grep '^\[device_profiles\.' foundation/prct-core/configs/device_profiles.toml
```

### Issue: GPU not detected

```
Error: No CUDA devices available on this system
```

**Solution**: Verify CUDA installation:

```bash
nvidia-smi
nvcc --version
```

### Issue: Replica timeout

```
Warning: Replica 3 unhealthy (no heartbeat for 10s)
```

**Solution**: Check GPU memory and reduce replicas:

```bash
nvidia-smi
export PRISM_REPLICAS=4  # Reduce from 8
```

### Issue: Build failure with CUDA

```
Error: Failed to compile cudarc
```

**Solution**: Ensure CUDA toolkit installed:

```bash
# Check CUDA path
echo $CUDA_PATH
echo $LD_LIBRARY_PATH

# Install CUDA if missing (Ubuntu)
sudo apt install nvidia-cuda-toolkit
```

## Advanced Usage

### Custom Replica Count

Override profile replica count:

```bash
# Run 16 replicas on 8 GPUs (2 per GPU)
export PRISM_REPLICAS=16
./run-prism-gpu.sh --profile runpod_b200
```

### Benchmark Single vs Multi-GPU

```bash
# Single GPU baseline
PRISM_DEVICE_PROFILE=rtx5070 ./run-prism-gpu.sh --config test.toml

# Multi-GPU
PRISM_DEVICE_PROFILE=benchmark PRISM_REPLICAS=8 ./run-prism-gpu.sh --config test.toml
```

### Debug Mode

Enable verbose logging:

```bash
PRISM_DEVICE_PROFILE=debug RUST_LOG=debug ./run-prism-gpu.sh
```

## Future Enhancements

### Planned Features

1. **Graph Partitioning**: True distributed computation with edge-cut partitioning
2. **Multi-Node**: gRPC/WebSocket for multi-node RunPod clusters
3. **Dynamic Load Balancing**: Reassign phases based on runtime profiling
4. **P2P Memory**: Direct GPU-GPU transfers via NVLink
5. **Checkpoint/Resume**: Save/restore multi-GPU state

### Experimental Features

Currently behind feature flags:

- `--multi-gpu` flag (default off locally, on for RunPod)
- `PRISM_FORCE_SINGLE_GPU=1` to disable multi-GPU even with profile

## References

- [Device Profiles Config](foundation/prct-core/configs/device_profiles.toml)
- [RunPod Deployment Guide](RUNPOD_8XB200_DEPLOYMENT_GUIDE.md)
- [GPU Architecture](foundation/prct-core/src/gpu/README.md)
- [Telemetry Schema](foundation/prct-core/src/telemetry/README.md)
