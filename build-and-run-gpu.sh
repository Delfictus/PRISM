#!/bin/bash

# Build and run your PRISM project with full GPU support

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PRISM GPU Build & Run Script                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR=$(pwd)
IMAGE_NAME="my-prism:gpu"

echo "Step 1: Building Docker image with GPU support..."
echo "This may take several minutes on first build..."
echo ""

docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "HOW TO USE YOUR GPU BINARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run with your protein data:"
echo ""
echo "  docker run --rm --runtime=nvidia --gpus all \\"
echo "    -v \"\$(pwd)/data:/data\" \\"
echo "    -v \"\$(pwd)/output:/output\" \\"
echo "    $IMAGE_NAME \\"
echo "    --input /data/nipah/2VSM.mtx --attempts 1000"
echo ""
echo "Or use the helper script:"
echo "  ./run-prism-gpu.sh data/nipah/2VSM.mtx 1000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
