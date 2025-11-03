#!/bin/bash
# PRISM-AI Setup Verification
# Quick check that everything is ready to run

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "PRISM-AI Setup Verification"
echo "============================"
echo ""

# Check 1: GPU
echo -n "GPU Available: "
if nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} $GPU"
else
    echo -e "${RED}✗ No GPU found${NC}"
    exit 1
fi

# Check 2: Benchmark binary
echo -n "DIMACS Benchmark Binary: "
if [ -f "./target/release/examples/simple_dimacs_benchmark" ]; then
    echo -e "${GREEN}✓${NC} Built"
else
    echo -e "${YELLOW}⊗${NC} Not built (will build on first run)"
fi

# Check 3: MEC binaries
echo -n "MEC Binaries: "
MEC_COUNT=0
[ -f "./target/release/meta-flagsctl" ] && ((MEC_COUNT++))
[ -f "./target/release/meta-ontologyctl" ] && ((MEC_COUNT++))
[ -f "./target/release/meta-reflexive-snapshot" ] && ((MEC_COUNT++))
[ -f "./target/release/federated-sim" ] && ((MEC_COUNT++))
echo -e "${GREEN}✓${NC} $MEC_COUNT/4 found"

# Check 4: Benchmark files
echo -n "DIMACS Benchmark Files: "
BENCH_DIR="/home/diddy/Downloads/PRISM-master/benchmarks/dimacs"
if [ -d "$BENCH_DIR" ]; then
    COUNT=$(ls -1 "$BENCH_DIR"/*.col 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} $COUNT graphs"
else
    echo -e "${RED}✗ Directory not found${NC}"
    exit 1
fi

# Check 5: Test runner
echo -n "Test Runner Script: "
if [ -x "./run_full_dimacs_test.sh" ]; then
    echo -e "${GREEN}✓${NC} Executable"
else
    echo -e "${RED}✗ Not executable${NC}"
    chmod +x ./run_full_dimacs_test.sh
    echo "  Fixed: Made executable"
fi

# Check 6: Output directory
echo -n "Test Results Directory: "
if [ -d "./test_results" ]; then
    echo -e "${GREEN}✓${NC} Ready"
else
    mkdir -p ./test_results
    echo -e "${GREEN}✓${NC} Created"
fi

# Check 7: PTX kernels
echo -n "GPU Kernels (PTX): "
if [ -d "./foundation/kernels/ptx" ]; then
    PTX_COUNT=$(ls -1 ./foundation/kernels/ptx/*.ptx 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} $PTX_COUNT kernels"
else
    echo -e "${YELLOW}⊗${NC} Will compile on build"
fi

echo ""
echo "============================="
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "To run full DIMACS test:"
echo "  ./run_full_dimacs_test.sh"
echo ""
echo "Or run benchmark directly:"
echo "  ./target/release/examples/simple_dimacs_benchmark"
