#!/bin/bash
# PRISM-AI Docker GPU Fix Script
# Based on GPU-FIX-SOLUTION.md

set -e  # Exit on error

echo "========================================================================"
echo "PRISM-AI Docker GPU Initialization Fix"
echo "========================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check NVIDIA driver
echo "Step 1: Checking NVIDIA driver..."
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA driver not found. Install drivers first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ NVIDIA driver found${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Step 2: Check Docker snap
echo "Step 2: Checking Docker snap..."
if ! snap list docker &> /dev/null; then
    echo -e "${RED}❌ Docker snap not found. Install: sudo snap install docker${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker snap found${NC}"
snap list docker | grep docker
echo ""

# Step 3: Check NVIDIA Container Toolkit
echo "Step 3: Checking NVIDIA Container Toolkit..."
if ! which nvidia-container-toolkit &> /dev/null; then
    echo -e "${YELLOW}⚠️  NVIDIA Container Toolkit not found. Installing...${NC}"
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
fi
echo -e "${GREEN}✅ NVIDIA Container Toolkit found${NC}"
echo ""

# Step 4: Check current service status
echo "Step 4: Current Docker service status..."
snap services docker
echo ""

# Step 5: Start NVIDIA Container Toolkit service
echo "Step 5: Starting NVIDIA Container Toolkit service..."
if ! sudo snap start docker.nvidia-container-toolkit; then
    echo -e "${RED}❌ Failed to start nvidia-container-toolkit service${NC}"
    echo "Check logs: journalctl -u snap.docker.nvidia-container-toolkit.service -n 50"
    exit 1
fi
echo -e "${GREEN}✅ nvidia-container-toolkit service started${NC}"

# Wait for service to complete
echo "Waiting for service initialization..."
sleep 3

# Step 6: Restart Docker daemon
echo "Step 6: Restarting Docker daemon..."
if ! sudo snap restart docker.dockerd; then
    echo -e "${RED}❌ Failed to restart Docker daemon${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker daemon restarted${NC}"

# Wait for Docker to restart
echo "Waiting for Docker to restart..."
sleep 3

# Step 7: Verify service status
echo "Step 7: Verifying service status..."
snap services docker
echo ""

# Check if nvidia-container-toolkit is now active
if snap services docker | grep -q "docker.nvidia-container-toolkit.*active"; then
    echo -e "${GREEN}✅ nvidia-container-toolkit service is ACTIVE${NC}"
else
    echo -e "${YELLOW}⚠️  nvidia-container-toolkit service may not be active${NC}"
    echo "Status:"
    snap services docker | grep nvidia-container-toolkit
fi
echo ""

# Step 8: Verify runtime configuration
echo "Step 8: Verifying Docker runtime configuration..."
if cat /var/snap/docker/*/config/daemon.json | grep -q '"nvidia"'; then
    echo -e "${GREEN}✅ NVIDIA runtime configured${NC}"
    echo "Runtime config:"
    cat /var/snap/docker/*/config/daemon.json | head -20
else
    echo -e "${RED}❌ NVIDIA runtime not configured${NC}"
    exit 1
fi
echo ""

# Step 9: Test GPU access in Docker
echo "Step 9: Testing GPU access in Docker..."
echo "Running: docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi"
echo ""

if docker run --rm --runtime=nvidia --gpus all \
    nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi; then
    echo ""
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}✅ SUCCESS! Docker can now access your GPU!${NC}"
    echo -e "${GREEN}========================================================================${NC}"
else
    echo ""
    echo -e "${RED}========================================================================${NC}"
    echo -e "${RED}❌ FAILED: Docker cannot access GPU${NC}"
    echo -e "${RED}========================================================================${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check service logs:"
    echo "   journalctl -u snap.docker.nvidia-container-toolkit.service -n 100"
    echo ""
    echo "2. Verify NVIDIA libraries exist:"
    echo "   ls -la /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*"
    echo ""
    echo "3. Try restarting services again:"
    echo "   sudo snap restart docker.nvidia-container-toolkit"
    echo "   sudo snap restart docker.dockerd"
    echo ""
    exit 1
fi

echo ""
echo "========================================================================"
echo "Next Steps for PRISM-AI"
echo "========================================================================"
echo ""
echo "1. Test PRISM GPU initialization (native):"
echo "   cd /home/diddy/Desktop/PRISM-FINNAL-PUSH"
echo "   cargo test --test test_gpu_setup"
echo ""
echo "2. Build PRISM with GPU support:"
echo "   cargo build --release --features cuda"
echo ""
echo "3. Run PRISM natively (recommended for best performance):"
echo "   ./target/release/prism-ai"
echo ""
echo "4. Or run PRISM in Docker with GPU:"
echo "   docker build -f Dockerfile.prism-gpu -t prism-ai:gpu ."
echo "   docker run --rm --runtime=nvidia --gpus all prism-ai:gpu"
echo ""
echo "Note: Always use '--runtime=nvidia --gpus all' when running GPU containers"
echo ""
echo "========================================================================"
echo "GPU Fix Complete!"
echo "========================================================================"
