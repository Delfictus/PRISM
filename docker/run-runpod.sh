#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# PRISM RunPod Launch Script
# ═══════════════════════════════════════════════════════════════════════════════
# Launches PRISM container on RunPod with 8x B200 GPUs
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Configuration
DOCKER_USER="${DOCKER_USER:-yourusername}"
IMAGE_NAME="prism-runpod"
VERSION="1.1.0"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${VERSION}"

# RunPod environment detection
if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "⚠ Warning: RUNPOD_POD_ID not set (running locally?)"
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PRISM RunPod Launcher (8x B200)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Check GPU availability
echo "→ Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"

if [ "${GPU_COUNT}" -ne 8 ]; then
    echo "⚠ Warning: Expected 8 GPUs, found ${GPU_COUNT}"
fi

# Create output directories
mkdir -p /workspace/prism-results
mkdir -p /workspace/prism-cache
mkdir -p /workspace/prism-logs

echo -e "${GREEN}✓ Created output directories${NC}"

# Run container
echo -e "${BLUE}→ Launching PRISM container...${NC}"

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /workspace/prism-results:/app/results \
    -v /workspace/prism-cache:/app/fluxnet_cache \
    -v /workspace/prism-logs:/app/logs \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e OMP_NUM_THREADS=288 \
    -e RAYON_NUM_THREADS=288 \
    -e RUST_LOG=info \
    -e RUST_BACKTRACE=1 \
    "${FULL_IMAGE}" \
    /app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml

echo -e "${GREEN}✓ Run complete${NC}"
echo ""
echo "Results saved to: /workspace/prism-results"
echo "FluxNet cache: /workspace/prism-cache"
echo "Logs: /workspace/prism-logs"
