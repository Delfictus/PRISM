#!/bin/bash
# Complete Profiling Workflow for PRISM-AI
# Profiles all pipelines with both Nsight Systems and Nsight Compute

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BINARY="${BINARY:-./target/release/prism-ai}"
REPORTS_DIR="${REPORTS_DIR:-./reports}"
DOCKER_IMAGE="delfictus/prism-ai-world-record:latest"

# Pipeline configurations
declare -A PIPELINES=(
    ["active-inference"]="Active Inference (evolution + EFE + belief)"
    ["transfer-entropy"]="Transfer Entropy (KSG method)"
    ["linear-algebra"]="Linear Algebra (GEMM, GEMV)"
    ["quantum"]="Quantum Computing (gates + circuits)"
    ["neuromorphic"]="Neuromorphic (reservoir + STDP)"
    ["graph-coloring"]="Graph Coloring (DSJC1000-5)"
)

# Hot kernels to profile with Nsight Compute
declare -A HOT_KERNELS=(
    ["active-inference"]="fused_active_inference_step,belief_update_kernel,prediction_error_kernel,kl_divergence_kernel"
    ["transfer-entropy"]="compute_distances_kernel,build_histogram_3d_kernel,compute_transfer_entropy_kernel"
    ["linear-algebra"]="gemv_kernel,matmul_kernel,saxpy_kernel"
    ["quantum"]="hadamard_gate_kernel,cnot_gate_kernel,time_evolution_kernel"
    ["neuromorphic"]="leaky_integration_kernel,reservoir_update,stdp_update"
    ["graph-coloring"]="parallel_greedy_coloring_kernel,validate_coloring"
)

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PRISM-AI Complete Profiling Workflow                       ║${NC}"
echo -e "${BLUE}║  Nsight Systems + Nsight Compute                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check prerequisites
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check for nsys
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}ERROR: nsys not found. Install CUDA Toolkit 12.0+${NC}"
    exit 1
fi

# Check for ncu
if ! command -v ncu &> /dev/null; then
    echo -e "${RED}ERROR: ncu not found. Install CUDA Toolkit 12.0+${NC}"
    exit 1
fi

# Check for binary
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}Binary not found at $BINARY${NC}"
    echo -e "${YELLOW}Attempting to extract from Docker image...${NC}"

    docker create --name tmp_extract "$DOCKER_IMAGE" > /dev/null 2>&1
    docker cp tmp_extract:/usr/local/bin/world_record "$BINARY" || {
        echo -e "${RED}Failed to extract binary. Run: cargo build --release${NC}"
        docker rm tmp_extract > /dev/null 2>&1
        exit 1
    }
    docker rm tmp_extract > /dev/null 2>&1
    chmod +x "$BINARY"
    echo -e "${GREEN}✓ Binary extracted from Docker${NC}"
fi

# Verify GPU access
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi failed. Check GPU drivers${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites satisfied${NC}"
echo

# Create report directories
echo -e "${YELLOW}[2/7] Creating report directories...${NC}"
for pipeline in "${!PIPELINES[@]}"; do
    mkdir -p "$REPORTS_DIR/nsys/$pipeline"
    mkdir -p "$REPORTS_DIR/ncu/$pipeline"
done
mkdir -p "$REPORTS_DIR/analysis"
mkdir -p "$REPORTS_DIR/csv"
echo -e "${GREEN}✓ Directories created${NC}"
echo

# Function to run Nsight Systems profiling
profile_nsys() {
    local pipeline=$1
    local description=$2

    echo -e "${BLUE}[Nsight Systems] Profiling: $description${NC}"

    nsys profile \
        --stats=true \
        --force-overwrite=true \
        --capture-range=cudaProfilerApi \
        --cuda-memory-usage=true \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --cpuctxsw=true \
        --gpu-metrics-device=all \
        -o "$REPORTS_DIR/nsys/$pipeline/${pipeline}_timeline" \
        "$BINARY" --pipeline "$pipeline" --profile-mode \
        2>&1 | tee "$REPORTS_DIR/nsys/$pipeline/nsys.log"

    # Export to CSV for analysis
    nsys stats "$REPORTS_DIR/nsys/$pipeline/${pipeline}_timeline.nsys-rep" \
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
        --format csv \
        --output "$REPORTS_DIR/csv/${pipeline}_nsys" \
        2>&1 | tee -a "$REPORTS_DIR/nsys/$pipeline/nsys.log"

    echo -e "${GREEN}✓ Nsight Systems: $pipeline complete${NC}"
    echo
}

# Function to run Nsight Compute profiling
profile_ncu() {
    local pipeline=$1
    local description=$2
    local kernels=${HOT_KERNELS[$pipeline]}

    echo -e "${BLUE}[Nsight Compute] Profiling: $description${NC}"
    echo -e "${BLUE}Hot kernels: $kernels${NC}"

    # Full metrics set
    ncu \
        --set full \
        --kernel-name-base demangled \
        --kernel-name "$kernels" \
        --target-processes all \
        --details-all \
        --csv \
        --log-file "$REPORTS_DIR/ncu/$pipeline/ncu_full.csv" \
        "$BINARY" --pipeline "$pipeline" --profile-mode \
        2>&1 | tee "$REPORTS_DIR/ncu/$pipeline/ncu.log"

    # Memory-focused metrics (faster)
    ncu \
        --set memory \
        --kernel-name-base demangled \
        --kernel-name "$kernels" \
        --csv \
        --log-file "$REPORTS_DIR/ncu/$pipeline/ncu_memory.csv" \
        "$BINARY" --pipeline "$pipeline" --profile-mode \
        2>&1 | tee -a "$REPORTS_DIR/ncu/$pipeline/ncu.log"

    # Occupancy metrics
    ncu \
        --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct \
        --kernel-name-base demangled \
        --kernel-name "$kernels" \
        --csv \
        --log-file "$REPORTS_DIR/ncu/$pipeline/ncu_occupancy.csv" \
        "$BINARY" --pipeline "$pipeline" --profile-mode \
        2>&1 | tee -a "$REPORTS_DIR/ncu/$pipeline/ncu.log"

    echo -e "${GREEN}✓ Nsight Compute: $pipeline complete${NC}"
    echo
}

# Phase 1: Nsight Systems (Timeline Analysis)
echo -e "${YELLOW}[3/7] Phase 1: Nsight Systems Timeline Profiling${NC}"
echo -e "${YELLOW}This phase captures kernel launch patterns, CPU-GPU gaps${NC}"
echo

for pipeline in "${!PIPELINES[@]}"; do
    profile_nsys "$pipeline" "${PIPELINES[$pipeline]}"
    sleep 2  # Cool down between runs
done

echo -e "${GREEN}✓ Phase 1 Complete${NC}"
echo

# Phase 2: Nsight Compute (Kernel-Level Analysis)
echo -e "${YELLOW}[4/7] Phase 2: Nsight Compute Kernel-Level Profiling${NC}"
echo -e "${YELLOW}This phase captures memory bandwidth, occupancy, tensor cores${NC}"
echo -e "${RED}WARNING: This phase is SLOW (~5-10 min per pipeline)${NC}"
echo

for pipeline in "${!PIPELINES[@]}"; do
    profile_ncu "$pipeline" "${PIPELINES[$pipeline]}"
    sleep 2  # Cool down between runs
done

echo -e "${GREEN}✓ Phase 2 Complete${NC}"
echo

# Generate summary
echo -e "${YELLOW}[5/7] Generating summary report...${NC}"

cat > "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md" << 'EOF'
# PRISM-AI Profiling Summary

**Generated:** $(date)
**Hardware:** $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

## Pipelines Profiled

EOF

for pipeline in "${!PIPELINES[@]}"; do
    echo "### $pipeline: ${PIPELINES[$pipeline]}" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "**Nsight Systems:**" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo '```' >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    if [ -f "$REPORTS_DIR/nsys/$pipeline/nsys.log" ]; then
        grep -A 5 "CUDA Kernel Statistics" "$REPORTS_DIR/nsys/$pipeline/nsys.log" | head -10 >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md" || true
    fi
    echo '```' >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"

    echo "**Nsight Compute:**" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "- Full metrics: reports/ncu/$pipeline/ncu_full.csv" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "- Memory metrics: reports/ncu/$pipeline/ncu_memory.csv" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "- Occupancy metrics: reports/ncu/$pipeline/ncu_occupancy.csv" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
    echo "" >> "$REPORTS_DIR/analysis/PROFILING_SUMMARY.md"
done

echo -e "${GREEN}✓ Summary generated: $REPORTS_DIR/analysis/PROFILING_SUMMARY.md${NC}"
echo

# List all generated files
echo -e "${YELLOW}[6/7] Generated files:${NC}"
echo
find "$REPORTS_DIR" -type f -name "*.nsys-rep" -o -name "*.csv" -o -name "*.log" | sort
echo

# Final instructions
echo -e "${YELLOW}[7/7] Next Steps${NC}"
echo
echo -e "${BLUE}View timeline in Nsight Systems GUI:${NC}"
for pipeline in "${!PIPELINES[@]}"; do
    echo "  nsys-ui $REPORTS_DIR/nsys/$pipeline/${pipeline}_timeline.nsys-rep"
done
echo
echo -e "${BLUE}View kernel details in Nsight Compute GUI:${NC}"
echo "  ncu-ui (then open .ncu-rep files from reports/ncu/)"
echo
echo -e "${BLUE}Analyze CSV data:${NC}"
echo "  ./scripts/analyze_profiles.sh"
echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Profiling Complete!                                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${YELLOW}Upload CSV files for detailed analysis:${NC}"
echo "  reports/csv/*.csv"
echo "  reports/ncu/*/*.csv"
echo
