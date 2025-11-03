#!/bin/bash
# Docker-based profiling for PRISM-AI
# Runs profiling inside the Docker container

set -e

DOCKER_IMAGE="delfictus/prism-ai-world-record:latest"
REPORTS_DIR="./reports"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PRISM-AI Docker Profiling                                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check for nvidia-docker
if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: Docker GPU access not working${NC}"
    echo -e "${YELLOW}Install nvidia-container-toolkit:${NC}"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo -e "${GREEN}✓ Docker GPU access working${NC}"
echo

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Function to run profiling in Docker
profile_in_docker() {
    local pipeline=$1
    local tool=$2  # nsys or ncu

    echo -e "${YELLOW}Profiling $pipeline with $tool...${NC}"

    if [ "$tool" == "nsys" ]; then
        docker run --rm --gpus all \
            -v "$(pwd)/$REPORTS_DIR:/output" \
            -e NUM_GPUS=1 \
            -e ATTEMPTS_PER_GPU=1000 \
            "$DOCKER_IMAGE" \
            nsys profile \
                --stats=true \
                --force-overwrite=true \
                -o "/output/nsys_${pipeline}" \
                /usr/local/bin/world_record
    elif [ "$tool" == "ncu" ]; then
        docker run --rm --gpus all \
            -v "$(pwd)/$REPORTS_DIR:/output" \
            -e NUM_GPUS=1 \
            -e ATTEMPTS_PER_GPU=100 \
            "$DOCKER_IMAGE" \
            ncu \
                --set full \
                --csv \
                --log-file "/output/ncu_${pipeline}.csv" \
                /usr/local/bin/world_record
    fi

    echo -e "${GREEN}✓ $pipeline profiling complete${NC}"
    echo
}

# Profile graph coloring (main workload)
echo -e "${YELLOW}[1/2] Nsight Systems timeline profiling...${NC}"
profile_in_docker "graph_coloring" "nsys"

echo -e "${YELLOW}[2/2] Nsight Compute kernel profiling...${NC}"
profile_in_docker "graph_coloring" "ncu"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Docker Profiling Complete!                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${BLUE}Results saved to: $REPORTS_DIR${NC}"
echo -e "${BLUE}View timeline: nsys-ui $REPORTS_DIR/nsys_graph_coloring.nsys-rep${NC}"
echo
