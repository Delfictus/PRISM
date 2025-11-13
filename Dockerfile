# PRISM AI World Record - Multi-GPU Docker Image
# Base: RunPod PyTorch with CUDA 12.8
# Target: 1-8× NVIDIA B200 GPUs (flexible multi-GPU support)

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Metadata
LABEL maintainer="PRISM AI Team"
LABEL description="PRISM World Record Graph Coloring - Multi-GPU Support (1-8× B200)"
LABEL version="1.1.0"
LABEL cuda.version="12.8.1"
LABEL gpu.target="1-8× NVIDIA B200 (flexible)"
LABEL features="FluxNet RL, Multi-GPU, Q-table persistence, Phase 2 hardening"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_VERSION=1.90.0
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/root/.cargo/bin:${PATH}
ENV RUST_BACKTRACE=1
ENV RUST_LOG=info
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    libssl-dev \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION}

# Verify CUDA installation
RUN nvcc --version

# Set working directory
WORKDIR /workspace/prism

# Copy entire PRISM codebase
COPY . .

# Build PRISM with CUDA support (release mode, optimized)
# Build only world_record_dsjc1000 example (skip broken examples)
RUN echo "🔨 Building PRISM with CUDA support..." && \
    cargo build --release --features cuda --example world_record_dsjc1000 && \
    cargo build --release --features cuda --lib && \
    echo "✅ Build complete!"

# Create necessary directories
RUN mkdir -p /workspace/prism/results/logs && \
    mkdir -p /workspace/prism/target/run_artifacts && \
    mkdir -p /workspace/prism/checkpoints

# Create orchestrator script for 8-GPU execution
RUN cat > /workspace/prism/run_8gpu_world_record.sh <<'SCRIPT'
#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   PRISM AI World Record - 8x B200 Multi-GPU Execution           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Verify 8 GPUs available
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🔍 Detecting GPUs..."
echo "   Found: ${NUM_GPUS} GPUs"
echo ""

if [ "${NUM_GPUS}" -lt 8 ]; then
    echo "⚠️  WARNING: Expected 8 GPUs, found ${NUM_GPUS}"
    echo "   The ultra-massive config is optimized for 8 GPUs"
    echo "   Performance may not be optimal with fewer GPUs"
    echo ""
fi

# Display GPU info
echo "📊 GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | \
    awk -F',' '{printf "   GPU %s: %s | VRAM: %s | Compute: %s\n", $1, $2, $3, $4}'
echo ""

# Display config info
CONFIG="${CONFIG_PATH:-foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml}"
GRAPH="${GRAPH_PATH:-benchmarks/dimacs/DSJC1000.5.col}"

echo "🎯 Configuration:"
echo "   Config: ${CONFIG}"
echo "   Graph: ${GRAPH}"
echo "   Target: 83 colors (world record)"
echo "   Max Runtime: 48 hours"
echo ""

# Check if graph file exists
if [ ! -f "${GRAPH}" ]; then
    echo "❌ ERROR: Graph file not found: ${GRAPH}"
    echo "   Available graphs:"
    ls -1 benchmarks/dimacs/*.col 2>/dev/null || echo "   No graphs found"
    exit 1
fi

# Start timestamp
START_TIME=$(date +%s)
LOG_FILE="results/logs/run_$(date +%Y%m%d_%H%M%S)_8xb200.log"

echo "📝 Logging to: ${LOG_FILE}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run PRISM world record attempt
./target/release/examples/world_record_dsjc1000 "${CONFIG}" 2>&1 | tee "${LOG_FILE}"

# End timestamp
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Run complete!"
echo "   Total time: ${HOURS}h ${MINUTES}m"
echo "   Log: ${LOG_FILE}"
echo ""

# Extract best result
echo "🏆 Best Result:"
grep -E "FINAL.*colors|BEST.*colors|chromatic.*[0-9]+" "${LOG_FILE}" | tail -5 || echo "   (Check log file for results)"
echo ""
SCRIPT

RUN chmod +x /workspace/prism/run_8gpu_world_record.sh

# Create monitoring script
RUN cat > /workspace/prism/monitor_gpus.sh <<'MONITOR'
#!/bin/bash
# Real-time GPU monitoring for PRISM run

echo "🔍 Monitoring GPU utilization (Ctrl+C to stop)..."
echo ""

watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | awk -F"," '\''{printf "GPU %s | Util: %s | Mem: %s / %s | Temp: %s | Power: %s\n", $1, $2, $4, $5, $6, $7}'\'
MONITOR

RUN chmod +x /workspace/prism/monitor_gpus.sh

# Create enhanced entrypoint script with multi-GPU support
RUN cat > /workspace/prism/entrypoint.sh <<'ENTRY'
#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          PRISM AI World Record Multi-GPU Container              ║"
echo "║        Flexible 1-8× GPU Support with Auto-Detection            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Detect GPU count
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

# Display system info
echo "📊 System Information:"
echo "   GPUs Detected: ${NUM_GPUS}"
echo "   CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | sed 's/,//' || echo 'N/A')"
echo "   RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "   CPUs: $(nproc)"
echo ""

# Auto-select device profile
if [ ${NUM_GPUS} -ge 8 ]; then
    PROFILE="runpod_b200"
    echo "🎯 Auto-selected profile: runpod_b200 (8 GPUs)"
elif [ ${NUM_GPUS} -ge 4 ]; then
    PROFILE="runpod_b200_4gpu"
    echo "🎯 Auto-selected profile: runpod_b200_4gpu (4 GPUs)"
elif [ ${NUM_GPUS} -ge 2 ]; then
    PROFILE="runpod_b200_2gpu"
    echo "🎯 Auto-selected profile: runpod_b200_2gpu (2 GPUs)"
elif [ ${NUM_GPUS} -eq 1 ]; then
    PROFILE="rtx5070"
    echo "🎯 Auto-selected profile: rtx5070 (1 GPU)"
else
    echo "⚠️  WARNING: No GPUs detected!"
    PROFILE="rtx5070"
fi

export PRISM_DEVICE_PROFILE=${PRISM_DEVICE_PROFILE:-$PROFILE}
echo "   Active profile: ${PRISM_DEVICE_PROFILE}"
echo ""

# Display GPU info if available
if [ ${NUM_GPUS} -gt 0 ]; then
    echo "📊 GPU Configuration:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | \
        awk -F',' '{printf "   GPU %s: %s | VRAM: %s | Compute: %s\n", $1, $2, $3, $4}'
    echo ""
fi

# Run command if provided, otherwise show help
if [ $# -eq 0 ]; then
    echo "🎯 Quick Commands:"
    echo ""
    echo "   ./run-prism-gpu.sh <config>      - Auto-detecting multi-GPU launcher"
    echo "   ./runpod-launch.sh <graph> <cfg> - RunPod-optimized launcher"
    echo "   ./run_8gpu_world_record.sh       - Legacy 8x B200 runner"
    echo "   ./monitor_gpus.sh                - Monitor GPU utilization"
    echo ""
    echo "📚 Key Files:"
    echo "   Device profiles: foundation/prct-core/configs/device_profiles.toml"
    echo "   Configs: foundation/prct-core/configs/*.toml"
    echo "   Binary: target/release/examples/world_record_dsjc1000"
    echo "   Logs: results/logs/"
    echo ""
    echo "🔧 Environment Variables:"
    echo "   PRISM_DEVICE_PROFILE=${PRISM_DEVICE_PROFILE}"
    echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
    echo ""
    exec bash
else
    exec "$@"
fi
ENTRY

RUN chmod +x /workspace/prism/entrypoint.sh

# Expose ports for monitoring/telemetry (optional)
EXPOSE 8080 8081

# Set entrypoint
ENTRYPOINT ["/workspace/prism/entrypoint.sh"]

# Default command: Show help
CMD []

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD nvidia-smi || exit 1

# Add labels for documentation
LABEL prism.version="1.1.0"
LABEL prism.features="FluxNet RL, Multi-GPU (1-8×), Q-table persistence, Phase 2 hardening"
LABEL prism.device_profiles="rtx5070, runpod_b200, runpod_b200_2gpu, runpod_b200_4gpu"
LABEL prism.gpu.min="1"
LABEL prism.gpu.max="8"
LABEL prism.gpu.optimal="8"
LABEL prism.target="DSJC1000.5 @ 83 colors"
LABEL prism.speedup.2gpu="1.7×"
LABEL prism.speedup.4gpu="3.2×"
LABEL prism.speedup.8gpu="5.5×"
