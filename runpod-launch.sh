#!/bin/bash
# RunPod-Specific Launch Script
#
# Wrapper for PRISM execution on RunPod instances.
# Reads RunPod metadata and configures environment.
#
# Usage:
#   ./runpod-launch.sh [graph_path] [config_path] [additional args...]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RunPod PRISM Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if running on RunPod
if [[ ! -f "/runpod.json" ]] && [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo -e "${YELLOW}[WARNING] Not running on RunPod (no /runpod.json or RUNPOD_POD_ID)${NC}"
    echo -e "${YELLOW}[WARNING] Proceeding anyway with local detection${NC}"
fi

# Parse RunPod metadata if available
if [[ -f "/runpod.json" ]]; then
    echo -e "${GREEN}[RUNPOD] Reading pod metadata from /runpod.json${NC}"

    POD_ID=$(jq -r '.id // "unknown"' /runpod.json)
    POD_NAME=$(jq -r '.name // "unknown"' /runpod.json)

    echo -e "${GREEN}[RUNPOD] Pod ID: ${POD_ID}${NC}"
    echo -e "${GREEN}[RUNPOD] Pod Name: ${POD_NAME}${NC}"

    export RUNPOD_POD_ID="$POD_ID"
    export RUNPOD_POD_NAME="$POD_NAME"
fi

# Detect GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
export RUNPOD_GPU_COUNT="$GPU_COUNT"

echo -e "${GREEN}[RUNPOD] Detected ${GPU_COUNT} GPU(s)${NC}"

# Enumerate GPUs
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU Enumeration${NC}"
echo -e "${BLUE}========================================${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id --format=csv,noheader | while IFS=',' read -r idx name total free cap pci; do
    echo -e "${GREEN}GPU $idx [$pci]:${NC}"
    echo -e "  Name: $name"
    echo -e "  Memory: $total (free: $free)"
    echo -e "  Compute: $cap"
done

# Set default graph and config if not provided
GRAPH_PATH="${1:-graphs/DSJC1000.5.col}"
CONFIG_PATH="${2:-foundation/prct-core/configs/runpod_8gpu.v1.1.toml}"

shift 2 2>/dev/null || true

# Check if graph exists
if [[ ! -f "$GRAPH_PATH" ]]; then
    echo -e "${RED}[ERROR] Graph file not found: ${GRAPH_PATH}${NC}"
    echo -e "${YELLOW}[INFO] Available graphs:${NC}"
    ls -lh graphs/*.col 2>/dev/null || echo "No graphs found in graphs/"
    exit 1
fi

# Check if config exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}[ERROR] Config file not found: ${CONFIG_PATH}${NC}"
    echo -e "${YELLOW}[INFO] Available configs:${NC}"
    ls -lh foundation/prct-core/configs/*.toml 2>/dev/null || echo "No configs found"
    exit 1
fi

echo -e "${GREEN}[CONFIG] Graph: ${GRAPH_PATH}${NC}"
echo -e "${GREEN}[CONFIG] Config: ${CONFIG_PATH}${NC}"

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runpod_output_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}[OUTPUT] Results will be saved to: ${OUTPUT_DIR}${NC}"

# Set output environment variables
export PRISM_OUTPUT_DIR="$OUTPUT_DIR"
export PRISM_TELEMETRY_FILE="$OUTPUT_DIR/telemetry.jsonl"
export PRISM_MULTI_GPU_SUMMARY="$OUTPUT_DIR/multi_gpu_summary.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Launching PRISM via run-prism-gpu.sh${NC}"
echo -e "${BLUE}========================================${NC}"

# Call the main launcher
./run-prism-gpu.sh \
    --config "$CONFIG_PATH" \
    "$GRAPH_PATH" \
    "$@"

EXIT_CODE=$?

echo -e "${BLUE}========================================${NC}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}[SUCCESS] Pipeline completed successfully${NC}"
    echo -e "${GREEN}[SUCCESS] Results saved to: ${OUTPUT_DIR}${NC}"

    # Display summary if available
    if [[ -f "$OUTPUT_DIR/multi_gpu_summary.json" ]]; then
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Multi-GPU Summary${NC}"
        echo -e "${BLUE}========================================${NC}"
        jq '.' "$OUTPUT_DIR/multi_gpu_summary.json"
    fi
else
    echo -e "${RED}[ERROR] Pipeline failed with exit code: ${EXIT_CODE}${NC}"
fi
echo -e "${BLUE}========================================${NC}"

exit $EXIT_CODE
