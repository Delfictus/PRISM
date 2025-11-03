#!/bin/bash
# Profile a single pipeline with both Nsight Systems and Nsight Compute
# Usage: ./profile_single_pipeline.sh <pipeline-name>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <pipeline>"
    echo
    echo "Available pipelines:"
    echo "  active-inference     - Active Inference (evolution + EFE + belief)"
    echo "  transfer-entropy     - Transfer Entropy (KSG method)"
    echo "  linear-algebra       - Linear Algebra (GEMM, GEMV)"
    echo "  quantum              - Quantum Computing (gates + circuits)"
    echo "  neuromorphic         - Neuromorphic (reservoir + STDP)"
    echo "  graph-coloring       - Graph Coloring (DSJC1000-5)"
    exit 1
fi

PIPELINE=$1
BINARY="${BINARY:-./target/release/prism-ai}"
REPORTS_DIR="${REPORTS_DIR:-./reports}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Profiling Pipeline: $PIPELINE${NC}"
echo

# Create directories
mkdir -p "$REPORTS_DIR/nsys/$PIPELINE"
mkdir -p "$REPORTS_DIR/ncu/$PIPELINE"

# Nsight Systems
echo -e "${YELLOW}[1/2] Nsight Systems (timeline)...${NC}"
nsys profile \
    --stats=true \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --trace=cuda,nvtx,osrt \
    -o "$REPORTS_DIR/nsys/$PIPELINE/${PIPELINE}_timeline" \
    "$BINARY" --pipeline "$PIPELINE" --profile-mode

nsys stats "$REPORTS_DIR/nsys/$PIPELINE/${PIPELINE}_timeline.nsys-rep" \
    --report cuda_api_sum,cuda_gpu_kern_sum \
    --format csv \
    --output "$REPORTS_DIR/nsys/$PIPELINE/${PIPELINE}"

echo -e "${GREEN}✓ Timeline saved to: $REPORTS_DIR/nsys/$PIPELINE/${PIPELINE}_timeline.nsys-rep${NC}"
echo

# Nsight Compute
echo -e "${YELLOW}[2/2] Nsight Compute (kernel metrics)...${NC}"
ncu \
    --set full \
    --kernel-name-base demangled \
    --target-processes all \
    --csv \
    --log-file "$REPORTS_DIR/ncu/$PIPELINE/ncu_full.csv" \
    "$BINARY" --pipeline "$PIPELINE" --profile-mode

echo -e "${GREEN}✓ Kernel metrics saved to: $REPORTS_DIR/ncu/$PIPELINE/ncu_full.csv${NC}"
echo

echo -e "${BLUE}View Results:${NC}"
echo "  Timeline:  nsys-ui $REPORTS_DIR/nsys/$PIPELINE/${PIPELINE}_timeline.nsys-rep"
echo "  Metrics:   cat $REPORTS_DIR/ncu/$PIPELINE/ncu_full.csv"
echo
