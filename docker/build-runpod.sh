#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# PRISM RunPod Docker Build Script
# ═══════════════════════════════════════════════════════════════════════════════
# Builds and pushes PRISM Docker image optimized for 8x B200 GPUs
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USER="delfictus"
IMAGE_NAME="prism-ai-world-record"
VERSION="latest"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${VERSION}"
LATEST_IMAGE="${FULL_IMAGE}"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PRISM RunPod Docker Build (8x B200 Optimized)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check if logged in to Docker Hub
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}⚠ Not logged in to Docker Hub${NC}"
    echo -e "${BLUE}→ Run: docker login${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Logged in to Docker Hub${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."

# Verify Dockerfile exists
if [ ! -f "Dockerfile.runpod" ]; then
    echo -e "${RED}✗ Dockerfile.runpod not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found Dockerfile.runpod${NC}"

# Build the image
echo -e "${BLUE}→ Building Docker image: ${FULL_IMAGE}${NC}"
echo -e "${BLUE}  This may take 30-60 minutes...${NC}"

docker build \
    -f Dockerfile.runpod \
    -t "${FULL_IMAGE}" \
    -t "${LATEST_IMAGE}" \
    --build-arg CUDA_ARCH=sm_100 \
    --build-arg NUM_BUILD_JOBS=64 \
    --progress=plain \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"

# Show image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE}" --format "{{.Size}}")
echo -e "${BLUE}→ Image size: ${IMAGE_SIZE}${NC}"

# Verify CUDA is available in the image
echo -e "${BLUE}→ Verifying CUDA in container...${NC}"
docker run --rm --gpus all "${FULL_IMAGE}" nvidia-smi --query-gpu=name --format=csv,noheader || true

# Ask to push
echo ""
echo -e "${YELLOW}Push to Docker Hub? (y/n)${NC}"
read -r PUSH_CONFIRM

if [[ "${PUSH_CONFIRM}" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}→ Pushing ${FULL_IMAGE}${NC}"
    docker push "${FULL_IMAGE}"

    echo -e "${BLUE}→ Pushing ${LATEST_IMAGE}${NC}"
    docker push "${LATEST_IMAGE}"

    echo -e "${GREEN}✓ Push successful${NC}"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Image published to Docker Hub!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}Pull command:${NC}"
    echo -e "  docker pull ${FULL_IMAGE}"
    echo ""
    echo -e "${BLUE}RunPod template:${NC}"
    echo -e "  Image: ${FULL_IMAGE}"
    echo -e "  GPUs: 8x B200"
    echo -e "  Disk: 100 GB (minimum)"
    echo -e "  Ports: 8080 (monitoring)"
else
    echo -e "${YELLOW}⚠ Skipping push${NC}"
fi

echo ""
echo -e "${GREEN}✓ Build complete!${NC}"
echo ""
echo -e "${BLUE}Local test command:${NC}"
echo -e "  docker run --rm --gpus all ${FULL_IMAGE}"
echo ""
