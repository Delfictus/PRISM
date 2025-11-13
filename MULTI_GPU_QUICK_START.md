# PRISM Multi-GPU Quick Start

## Single RTX 5070 Workstation

```bash
# Clone repository
cd /path/to/PRISM

# Build with CUDA
cargo build --release --features cuda

# Run on single GPU (auto-detected)
./run-prism-gpu.sh --config foundation/prct-core/configs/world_record.v1.toml

# Or explicitly
PRISM_DEVICE_PROFILE=rtx5070 \
cargo run --release --features cuda -- \
  --config foundation/prct-core/configs/world_record.v1.toml \
  graphs/DSJC1000.5.col
```

## RunPod 8× B200 Instance

```bash
# SSH into RunPod pod
ssh root@your-pod-id.runpod.io

# Clone repository
cd /workspace
git clone https://github.com/your-org/PRISM.git
cd PRISM

# Build
cargo build --release --features cuda

# Run with automatic detection
./runpod-launch.sh graphs/DSJC1000.5.col foundation/prct-core/configs/runpod_8gpu.v1.1.toml

# Check results
cat multi_gpu_summary.json | jq '.'
```

## Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor telemetry
tail -f telemetry.jsonl | jq -c 'select(.replica_id)'

# Check replica health
grep "HEARTBEAT" telemetry.jsonl | tail -20
```

## Troubleshooting

### GPU not detected

```bash
nvidia-smi  # Verify CUDA available
nvcc --version  # Check CUDA toolkit
```

### Profile not found

```bash
# List available profiles
grep '^\[device_profiles\.' foundation/prct-core/configs/device_profiles.toml

# Use explicit profile
export PRISM_DEVICE_PROFILE=rtx5070
```

### Build failure

```bash
# Clean and rebuild
cargo clean
cargo build --release --features cuda
```

## Configuration

Edit `foundation/prct-core/configs/device_profiles.toml` to customize:

- Replica count
- Device filters
- Sync interval
- P2P memory access

See `README_MULTI_GPU.md` for full documentation.
