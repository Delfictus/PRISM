#!/bin/bash
# Quick build and push to Docker Hub
set -euo pipefail

echo "🚀 Building PRISM for RunPod (8x B200)..."
docker build -f Dockerfile.runpod -t delfictus/prism-ai-world-record:latest \
  --build-arg CUDA_ARCH=sm_100 --build-arg NUM_BUILD_JOBS=64 .

echo "📤 Pushing to Docker Hub..."
docker push delfictus/prism-ai-world-record:latest

echo "✅ Done! Image: delfictus/prism-ai-world-record:latest"
