#!/bin/bash
# Compare baseline (unfused) vs fused kernel performance
# Usage: ./compare_profiles.sh <baseline_dir> <fused_dir>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline_reports_dir> <fused_reports_dir>"
    echo
    echo "Example:"
    echo "  $0 ./reports_baseline ./reports_fused"
    exit 1
fi

BASELINE_DIR=$1
FUSED_DIR=$2
OUTPUT_DIR="./reports/comparison"

mkdir -p "$OUTPUT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PRISM-AI Performance Comparison                            ║${NC}"
echo -e "${BLUE}║  Baseline (Unfused) vs Fused Kernels                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

echo -e "${YELLOW}Baseline: $BASELINE_DIR${NC}"
echo -e "${YELLOW}Fused:    $FUSED_DIR${NC}"
echo

# Function to extract total kernel time from Nsight Systems
extract_kernel_time() {
    local csv_file=$1
    if [ -f "$csv_file" ]; then
        # Sum all kernel durations
        tail -n +2 "$csv_file" | awk -F',' '{sum += $3} END {print sum}'
    else
        echo "0"
    fi
}

# Create comparison report
cat > "$OUTPUT_DIR/SPEEDUP_REPORT.md" << 'EOF'
# PRISM-AI Speedup Report

**Comparison:** Baseline (Unfused) vs Fused Kernels
**Generated:** $(date)

## Summary

EOF

echo -e "${YELLOW}Analyzing pipelines...${NC}"

total_baseline=0
total_fused=0

for pipeline in active-inference transfer-entropy linear-algebra quantum neuromorphic graph-coloring; do
    baseline_csv="$BASELINE_DIR/csv/${pipeline}_nsys_cuda_gpu_kern_sum.csv"
    fused_csv="$FUSED_DIR/csv/${pipeline}_nsys_cuda_gpu_kern_sum.csv"

    if [ -f "$baseline_csv" ] && [ -f "$fused_csv" ]; then
        baseline_time=$(extract_kernel_time "$baseline_csv")
        fused_time=$(extract_kernel_time "$fused_csv")

        if [ "$baseline_time" != "0" ] && [ "$fused_time" != "0" ]; then
            # Calculate speedup using bc for floating point
            speedup=$(echo "scale=2; $baseline_time / $fused_time" | bc)

            # Add to totals
            total_baseline=$(echo "$total_baseline + $baseline_time" | bc)
            total_fused=$(echo "$total_fused + $fused_time" | bc)

            # Write to report
            cat >> "$OUTPUT_DIR/SPEEDUP_REPORT.md" << PIPELINE_REPORT

### $pipeline

- **Baseline:** ${baseline_time}μs
- **Fused:** ${fused_time}μs
- **Speedup:** ${speedup}x

PIPELINE_REPORT

            # Console output
            if (( $(echo "$speedup >= 3.0" | bc -l) )); then
                echo -e "${GREEN}✓ $pipeline: ${speedup}x speedup (${baseline_time}μs → ${fused_time}μs)${NC}"
            elif (( $(echo "$speedup >= 2.0" | bc -l) )); then
                echo -e "${YELLOW}◐ $pipeline: ${speedup}x speedup (${baseline_time}μs → ${fused_time}μs)${NC}"
            else
                echo -e "${RED}✗ $pipeline: ${speedup}x speedup (${baseline_time}μs → ${fused_time}μs)${NC}"
            fi
        fi
    fi
done

# Overall speedup
if [ "$total_baseline" != "0" ] && [ "$total_fused" != "0" ]; then
    overall_speedup=$(echo "scale=2; $total_baseline / $total_fused" | bc)

    cat >> "$OUTPUT_DIR/SPEEDUP_REPORT.md" << OVERALL_REPORT

---

## Overall Platform Performance

- **Total Baseline Time:** ${total_baseline}μs
- **Total Fused Time:** ${total_fused}μs
- **Overall Speedup:** ${overall_speedup}x

### Speedup Analysis

OVERALL_REPORT

    if (( $(echo "$overall_speedup >= 3.0" | bc -l) )); then
        echo "**Status:** ✅ EXCEEDS TARGET (>3x)" >> "$OUTPUT_DIR/SPEEDUP_REPORT.md"
        echo -e "${GREEN}═══════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ OVERALL SPEEDUP: ${overall_speedup}x${NC}"
        echo -e "${GREEN}✓ TARGET ACHIEVED (>3x)${NC}"
        echo -e "${GREEN}═══════════════════════════════════════${NC}"
    elif (( $(echo "$overall_speedup >= 2.5" | bc -l) )); then
        echo "**Status:** ⚠️ CLOSE TO TARGET (2.5-3x)" >> "$OUTPUT_DIR/SPEEDUP_REPORT.md"
        echo -e "${YELLOW}═══════════════════════════════════════${NC}"
        echo -e "${YELLOW}◐ OVERALL SPEEDUP: ${overall_speedup}x${NC}"
        echo -e "${YELLOW}◐ CLOSE TO TARGET (need 3x)${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════${NC}"
    else
        echo "**Status:** ❌ BELOW TARGET (<2.5x)" >> "$OUTPUT_DIR/SPEEDUP_REPORT.md"
        echo -e "${RED}═══════════════════════════════════════${NC}"
        echo -e "${RED}✗ OVERALL SPEEDUP: ${overall_speedup}x${NC}"
        echo -e "${RED}✗ BELOW TARGET (need 3x)${NC}"
        echo -e "${RED}═══════════════════════════════════════${NC}"
    fi
fi

echo
cat >> "$OUTPUT_DIR/SPEEDUP_REPORT.md" << 'EOF'

## Detailed Kernel Comparison

See individual pipeline reports for kernel-level breakdowns.

EOF

# Generate detailed comparison tables
echo -e "${YELLOW}Generating detailed comparison tables...${NC}"

if command -v python3 &> /dev/null; then
    python3 << 'PYTHON_COMPARE'
import csv
import os

baseline_dir = os.environ.get('BASELINE_DIR', './reports_baseline')
fused_dir = os.environ.get('FUSED_DIR', './reports_fused')
output_dir = './reports/comparison'

pipelines = ['active-inference', 'transfer-entropy', 'linear-algebra', 'quantum', 'neuromorphic', 'graph-coloring']

for pipeline in pipelines:
    baseline_csv = f"{baseline_dir}/csv/{pipeline}_nsys_cuda_gpu_kern_sum.csv"
    fused_csv = f"{fused_dir}/csv/{pipeline}_nsys_cuda_gpu_kern_sum.csv"

    if not os.path.exists(baseline_csv) or not os.path.exists(fused_csv):
        continue

    # Read baseline kernels
    baseline_kernels = {}
    try:
        with open(baseline_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel = row.get('Name', 'unknown')
                duration = float(row.get('Duration', 0))
                baseline_kernels[kernel] = duration
    except:
        continue

    # Read fused kernels
    fused_kernels = {}
    try:
        with open(fused_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel = row.get('Name', 'unknown')
                duration = float(row.get('Duration', 0))
                fused_kernels[kernel] = duration
    except:
        continue

    # Generate comparison table
    with open(f"{output_dir}/{pipeline}_comparison.csv", 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Kernel', 'Baseline_us', 'Fused_us', 'Speedup', 'Status'])

        for kernel in sorted(baseline_kernels.keys()):
            baseline_time = baseline_kernels[kernel]
            fused_time = fused_kernels.get(kernel, baseline_time)

            if fused_time > 0:
                speedup = baseline_time / fused_time
                status = 'FASTER' if speedup > 1.1 else 'SAME' if speedup > 0.9 else 'SLOWER'
            else:
                speedup = 0
                status = 'MISSING'

            writer.writerow([kernel, f"{baseline_time:.2f}", f"{fused_time:.2f}", f"{speedup:.2f}", status])

    print(f"  ✓ {pipeline} comparison table generated")

print(f"\nComparison tables saved to: {output_dir}/")
PYTHON_COMPARE
fi

echo
echo -e "${BLUE}Report saved to: $OUTPUT_DIR/SPEEDUP_REPORT.md${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review: cat $OUTPUT_DIR/SPEEDUP_REPORT.md"
echo "  2. Analyze details: ls $OUTPUT_DIR/*_comparison.csv"
echo "  3. Visualize: Open .nsys-rep files in nsys-ui for side-by-side comparison"
echo
