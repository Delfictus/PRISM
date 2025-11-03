#!/bin/bash

# Quick runner for PRISM GPU binary

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input-file> [attempts]"
    echo ""
    echo "Example:"
    echo "  $0 data/nipah/2VSM.mtx 1000"
    echo "  $0 data/nipah/2VSM.mtx 10000"
    echo ""
    exit 1
fi

INPUT_FILE="$1"
ATTEMPTS="${2:-1000}"
IMAGE_NAME="my-prism:gpu"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "❌ Docker image not found!"
    echo ""
    echo "Build it first:"
    echo "  ./build-and-run-gpu.sh"
    echo ""
    exit 1
fi

# Get absolute paths
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Running PRISM with GPU Acceleration                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Input: $INPUT_FILE"
echo "Attempts: $ATTEMPTS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Convert relative path to container path
CONTAINER_INPUT="/data/${INPUT_FILE#data/}"

echo "Starting GPU run..."
echo ""

# Run with GPU
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$DATA_DIR:/data" \
  -v "$OUTPUT_DIR:/output" \
  "$IMAGE_NAME" \
  --input "$CONTAINER_INPUT" \
  --attempts "$ATTEMPTS"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Run completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
else
    echo "❌ Run failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
