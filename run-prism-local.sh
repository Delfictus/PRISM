#!/bin/bash

# Run PRISM locally with GPU support (no Docker needed)

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

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if binary exists
BINARY="./target/release/prism-ai"
if [ ! -f "$BINARY" ]; then
    echo "❌ Binary not found: $BINARY"
    echo ""
    echo "Build it first:"
    echo "  cargo build --release"
    echo ""
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Running PRISM with Local GPU Acceleration                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Input: $INPUT_FILE"
echo "Attempts: $ATTEMPTS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Detection failed')"
echo ""
echo "Starting GPU run..."
echo ""

# Set CUDA library path to use CUDA 13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Run the binary
"$BINARY" --input "$INPUT_FILE" --attempts "$ATTEMPTS"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Run completed successfully!"
else
    echo "❌ Run failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
