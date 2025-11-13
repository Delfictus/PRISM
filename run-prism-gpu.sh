#!/bin/bash
# PRISM GPU Runtime Launcher
#
# Automatically detects environment (local RTX vs RunPod cluster) and
# configures device profile accordingly.
#
# Usage:
#   ./run-prism-gpu.sh [--profile PROFILE] [--config CONFIG] [--replicas N] [additional args...]
#
# Environment Variables:
#   PRISM_DEVICE_PROFILE - Override device profile (default: auto-detect)
#   PRISM_REPLICAS - Override replica count (default: from profile)
#   RUNPOD_GPU_COUNT - Set by RunPod, number of GPUs available
#   RUNPOD_POD_ID - Set by RunPod, unique pod identifier

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_CONFIG="foundation/prct-core/configs/world_record.v1.toml"
DEVICE_PROFILE=""
CONFIG_FILE=""
REPLICAS=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            DEVICE_PROFILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use default config if not specified
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PRISM GPU Runtime Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

# Auto-detect environment if profile not specified
if [[ -z "$DEVICE_PROFILE" ]]; then
    if [[ -n "${RUNPOD_GPU_COUNT:-}" ]]; then
        # RunPod environment detected
        echo -e "${GREEN}[ENV] RunPod environment detected${NC}"
        echo -e "${GREEN}[ENV] GPU Count: ${RUNPOD_GPU_COUNT}${NC}"
        echo -e "${GREEN}[ENV] Pod ID: ${RUNPOD_POD_ID:-unknown}${NC}"

        if [[ "$RUNPOD_GPU_COUNT" -eq 1 ]]; then
            DEVICE_PROFILE="runpod_b200"
        elif [[ "$RUNPOD_GPU_COUNT" -eq 2 ]]; then
            DEVICE_PROFILE="runpod_b200_2gpu"
        elif [[ "$RUNPOD_GPU_COUNT" -eq 4 ]]; then
            DEVICE_PROFILE="runpod_b200_4gpu"
        elif [[ "$RUNPOD_GPU_COUNT" -ge 8 ]]; then
            DEVICE_PROFILE="runpod_b200"
        else
            DEVICE_PROFILE="runpod_b200_fixed"
        fi

        # Auto-set replicas to match GPU count if not specified
        if [[ -z "$REPLICAS" ]]; then
            REPLICAS="$RUNPOD_GPU_COUNT"
        fi
    else
        # Local environment (RTX workstation)
        echo -e "${GREEN}[ENV] Local RTX workstation detected${NC}"
        DEVICE_PROFILE="rtx5070"

        if [[ -z "$REPLICAS" ]]; then
            REPLICAS="1"
        fi
    fi
fi

echo -e "${GREEN}[CONFIG] Device Profile: ${DEVICE_PROFILE}${NC}"
echo -e "${GREEN}[CONFIG] Config File: ${CONFIG_FILE}${NC}"
echo -e "${GREEN}[CONFIG] Replicas: ${REPLICAS}${NC}"

# Export environment variables
export PRISM_DEVICE_PROFILE="$DEVICE_PROFILE"
if [[ -n "$REPLICAS" ]]; then
    export PRISM_REPLICAS="$REPLICAS"
fi

# Verify CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[ERROR] nvidia-smi not found. CUDA may not be installed.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU Topology${NC}"
echo -e "${BLUE}========================================${NC}"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | while IFS=',' read -r idx name mem cap; do
    echo -e "${GREEN}GPU $idx: $name | $mem | Compute $cap${NC}"
done

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}[ERROR] Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building PRISM${NC}"
echo -e "${BLUE}========================================${NC}"

# Build with CUDA support
cargo build --release --features cuda

if [[ $? -ne 0 ]]; then
    echo -e "${RED}[ERROR] Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}[BUILD] Build successful${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Launching PRISM Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Launch the pipeline
cargo run --release --features cuda -- \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo -e "${BLUE}========================================${NC}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}[SUCCESS] Pipeline completed successfully${NC}"
else
    echo -e "${RED}[ERROR] Pipeline failed with exit code: ${EXIT_CODE}${NC}"
fi
echo -e "${BLUE}========================================${NC}"

exit $EXIT_CODE
