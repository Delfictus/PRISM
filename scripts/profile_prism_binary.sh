#!/bin/bash
# Profile the actual PRISM-AI binary with a real workload
# This script profiles whatever your binary does when executed

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

BINARY="./target/release/prism-ai"
REPORTS_DIR="./reports/prism_binary"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Profile PRISM-AI Binary                                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Binary not found: $BINARY${NC}"
    echo "Build it with: cargo build --release"
    exit 1
fi

echo -e "${GREEN}✓ Binary found: $BINARY${NC}"
echo

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Check what the binary does
echo -e "${YELLOW}Testing binary execution...${NC}"
echo

# Try running it directly to see what it does
echo "Output from running binary:"
echo "─────────────────────────────────────────"
timeout 5 "$BINARY" 2>&1 || echo "(Binary timed out or exited)"
echo "─────────────────────────────────────────"
echo

# Profile with Nsight Systems
echo -e "${YELLOW}[1/2] Profiling with Nsight Systems...${NC}"

nsys profile \
    --stats=true \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --trace=cuda,nvtx,osrt \
    --duration=30 \
    -o "$REPORTS_DIR/prism_timeline" \
    timeout 30 "$BINARY" 2>&1 | tee "$REPORTS_DIR/nsys.log"

if [ -f "$REPORTS_DIR/prism_timeline.nsys-rep" ]; then
    echo -e "${GREEN}✓ Timeline captured${NC}"

    # Export stats
    nsys stats "$REPORTS_DIR/prism_timeline.nsys-rep" \
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
        --format csv \
        --output "$REPORTS_DIR/prism" 2>&1 | tee -a "$REPORTS_DIR/nsys.log"

    echo -e "${GREEN}✓ Stats exported to CSV${NC}"
else
    echo -e "${RED}✗ No timeline file generated${NC}"
    echo "This could mean:"
    echo "  1. Binary doesn't use CUDA"
    echo "  2. Binary exited before profiling started"
    echo "  3. Binary needs specific arguments"
fi
echo

# Profile with Nsight Compute (lightweight metrics only)
echo -e "${YELLOW}[2/2] Profiling with Nsight Compute (quick metrics)...${NC}"

ncu \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name-base demangled \
    --target-processes all \
    --csv \
    --log-file "$REPORTS_DIR/prism_ncu.csv" \
    timeout 30 "$BINARY" 2>&1 | tee "$REPORTS_DIR/ncu.log"

if [ -f "$REPORTS_DIR/prism_ncu.csv" ] && [ -s "$REPORTS_DIR/prism_ncu.csv" ]; then
    echo -e "${GREEN}✓ Kernel metrics captured${NC}"

    # Show summary
    echo
    echo -e "${BLUE}Top kernels by time:${NC}"
    if [ -f "$REPORTS_DIR/prism_cuda_gpu_kern_sum.csv" ]; then
        head -10 "$REPORTS_DIR/prism_cuda_gpu_kern_sum.csv"
    fi
else
    echo -e "${YELLOW}⚠ No kernel metrics captured${NC}"
fi

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Profiling Complete                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${BLUE}Results:${NC}"
echo "  Timeline:  $REPORTS_DIR/prism_timeline.nsys-rep"
echo "  Metrics:   $REPORTS_DIR/prism_ncu.csv"
echo "  Logs:      $REPORTS_DIR/*.log"
echo
echo -e "${BLUE}View:${NC}"
echo "  nsys-ui $REPORTS_DIR/prism_timeline.nsys-rep"
echo "  cat $REPORTS_DIR/prism_cuda_gpu_kern_sum.csv"
echo
