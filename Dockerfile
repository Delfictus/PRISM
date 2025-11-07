# PRISM AI World Record - 8x B200 Multi-GPU Docker Image
# Base: RunPod PyTorch with CUDA 12.8
# Target: 8x NVIDIA B200 GPUs (1440GB VRAM)

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Metadata
LABEL maintainer="PRISM AI Team"
LABEL description="PRISM World Record Graph Coloring - 8x B200 Multi-GPU"
LABEL version="1.0.0"
LABEL cuda.version="12.8.1"
LABEL gpu.target="8x NVIDIA B200"

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

# Create entrypoint script
RUN cat > /workspace/prism/entrypoint.sh <<'ENTRY'
#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              PRISM AI World Record Container                     ║"
echo "║                 8x B200 Multi-GPU Ready                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Display system info
echo "📊 System Information:"
echo "   GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "   CUDA: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "   RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "   CPUs: $(nproc)"
echo ""

# Run command if provided, otherwise show help
if [ $# -eq 0 ]; then
    echo "🎯 Available Commands:"
    echo ""
    echo "   ./run_8gpu_world_record.sh       - Run 8x B200 world record attempt"
    echo "   ./monitor_gpus.sh                - Monitor GPU utilization"
    echo "   bash                             - Interactive shell"
    echo ""
    echo "📚 Files:"
    echo "   Config: foundation/prct-core/configs/wr_ultra_8xb200.v1.0.toml"
    echo "   Binary: target/release/examples/world_record_dsjc1000"
    echo "   Logs: results/logs/"
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
LABEL prism.config="wr_ultra_8xb200.v1.0"
LABEL prism.gpus="8"
LABEL prism.vram="1440GB"
LABEL prism.target="DSJC1000.5 @ 83 colors"
LABEL prism.thermodynamic.replicas="10000"
LABEL prism.thermodynamic.temps="2000"
LABEL prism.quantum.attempts="80000"
LABEL prism.quantum.depth="20"
