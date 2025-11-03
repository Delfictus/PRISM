#!/bin/bash
# Automated Analysis of PRISM-AI Profiling Data
# Extracts key metrics from Nsight Systems and Nsight Compute reports

set -e

REPORTS_DIR="${REPORTS_DIR:-./reports}"
ANALYSIS_DIR="$REPORTS_DIR/analysis"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PRISM-AI Profile Analysis                                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Create analysis spreadsheet
cat > "$ANALYSIS_DIR/kernel_metrics.csv" << 'EOF'
Pipeline,Kernel,Duration(us),Occupancy(%),DRAM_Throughput(%),Tensor_Core_Active(%),Fusion_Score
EOF

echo -e "${YELLOW}[1/5] Extracting kernel durations from Nsight Systems...${NC}"

for pipeline_dir in "$REPORTS_DIR"/nsys/*/; do
    pipeline=$(basename "$pipeline_dir")

    if [ -f "$REPORTS_DIR/csv/${pipeline}_nsys_cuda_gpu_kern_sum.csv" ]; then
        echo "Processing: $pipeline"

        # Extract top 10 kernels by duration
        tail -n +2 "$REPORTS_DIR/csv/${pipeline}_nsys_cuda_gpu_kern_sum.csv" | \
        awk -F',' '{print $2","$3","$4}' | \
        sort -t',' -k2 -rn | \
        head -10 > "$ANALYSIS_DIR/${pipeline}_top_kernels.csv"

        echo "  Top kernels saved to: $ANALYSIS_DIR/${pipeline}_top_kernels.csv"
    fi
done

echo -e "${GREEN}✓ Nsight Systems analysis complete${NC}"
echo

echo -e "${YELLOW}[2/5] Extracting kernel metrics from Nsight Compute...${NC}"

for pipeline_dir in "$REPORTS_DIR"/ncu/*/; do
    pipeline=$(basename "$pipeline_dir")

    if [ -f "$pipeline_dir/ncu_full.csv" ]; then
        echo "Processing: $pipeline"

        # Extract key metrics using Python (if available) or awk
        if command -v python3 &> /dev/null; then
            python3 << PYTHON_SCRIPT
import csv
import sys

pipeline = "$pipeline"
csv_file = "$pipeline_dir/ncu_full.csv"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract key columns
    metrics = []
    for row in rows:
        kernel = row.get('Kernel Name', 'unknown')
        duration = row.get('Duration', '0')
        occupancy = row.get('SM %', '0')
        dram = row.get('DRAM %', '0')

        # Clean values
        kernel = kernel.split('(')[0] if '(' in kernel else kernel
        duration = duration.replace(' us', '').replace(',', '')
        occupancy = occupancy.replace('%', '')
        dram = dram.replace('%', '')

        metrics.append([pipeline, kernel, duration, occupancy, dram])

    # Write to analysis CSV
    with open('$ANALYSIS_DIR/${pipeline}_ncu_metrics.csv', 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Pipeline', 'Kernel', 'Duration_us', 'Occupancy_%', 'DRAM_%'])
        writer.writerows(metrics)

    print(f"  Extracted {len(metrics)} kernel metrics")

except Exception as e:
    print(f"  Warning: Could not parse CSV: {e}", file=sys.stderr)
PYTHON_SCRIPT
        else
            echo "  Warning: Python3 not available, skipping detailed analysis"
        fi
    fi
done

echo -e "${GREEN}✓ Nsight Compute analysis complete${NC}"
echo

echo -e "${YELLOW}[3/5] Calculating fusion opportunities...${NC}"

# Fusion score calculation
if command -v python3 &> /dev/null; then
    python3 << 'PYTHON_FUSION'
import os
import csv
import glob

reports_dir = os.environ.get('REPORTS_DIR', './reports')
analysis_dir = f"{reports_dir}/analysis"

# Read all kernel metrics
all_metrics = []
for csv_file in glob.glob(f"{analysis_dir}/*_ncu_metrics.csv"):
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            all_metrics.extend(list(reader))
    except:
        pass

# Calculate fusion scores
fusion_opportunities = []

for metric in all_metrics:
    try:
        occupancy = float(metric.get('Occupancy_%', 0))
        dram = float(metric.get('DRAM_%', 0))
        duration = float(metric.get('Duration_us', 0))

        # Fusion score heuristic:
        # - Low occupancy (<50%) = opportunity
        # - High DRAM (>70%) = memory-bound
        # - Short duration (<100us) = launch overhead dominant
        fusion_score = 0

        if dram > 70:
            fusion_score += 2  # Memory-bound
        if occupancy < 50:
            fusion_score += 2  # Low occupancy
        if duration < 100:
            fusion_score += 1  # Launch overhead

        fusion_opportunities.append({
            'pipeline': metric['Pipeline'],
            'kernel': metric['Kernel'],
            'duration': duration,
            'occupancy': occupancy,
            'dram': dram,
            'fusion_score': min(fusion_score, 5),
            'recommendation': 'HIGH' if fusion_score >= 4 else 'MEDIUM' if fusion_score >= 2 else 'LOW'
        })
    except:
        continue

# Sort by fusion score
fusion_opportunities.sort(key=lambda x: x['fusion_score'], reverse=True)

# Write fusion report
with open(f"{analysis_dir}/fusion_opportunities.csv", 'w', newline='') as out:
    if fusion_opportunities:
        writer = csv.DictWriter(out, fieldnames=fusion_opportunities[0].keys())
        writer.writeheader()
        writer.writerows(fusion_opportunities)

print(f"Identified {len([x for x in fusion_opportunities if x['recommendation'] == 'HIGH'])} high-priority fusion opportunities")
print(f"Results: {analysis_dir}/fusion_opportunities.csv")
PYTHON_FUSION
fi

echo -e "${GREEN}✓ Fusion analysis complete${NC}"
echo

echo -e "${YELLOW}[4/5] Generating comparison report...${NC}"

cat > "$ANALYSIS_DIR/BASELINE_METRICS.md" << 'EOF'
# PRISM-AI Baseline Performance Metrics

**Generated:** $(date)
**Purpose:** Unfused kernel baseline for comparison with fused implementations

## Summary Statistics

### By Pipeline

EOF

for pipeline in active-inference transfer-entropy linear-algebra quantum neuromorphic graph-coloring; do
    if [ -f "$ANALYSIS_DIR/${pipeline}_top_kernels.csv" ]; then
        echo "#### $pipeline" >> "$ANALYSIS_DIR/BASELINE_METRICS.md"
        echo '```' >> "$ANALYSIS_DIR/BASELINE_METRICS.md"
        head -5 "$ANALYSIS_DIR/${pipeline}_top_kernels.csv" >> "$ANALYSIS_DIR/BASELINE_METRICS.md"
        echo '```' >> "$ANALYSIS_DIR/BASELINE_METRICS.md"
        echo "" >> "$ANALYSIS_DIR/BASELINE_METRICS.md"
    fi
done

cat >> "$ANALYSIS_DIR/BASELINE_METRICS.md" << 'EOF'

## Fusion Opportunities

High-priority kernels for fusion (score >= 4):

```
EOF

if [ -f "$ANALYSIS_DIR/fusion_opportunities.csv" ]; then
    grep "HIGH" "$ANALYSIS_DIR/fusion_opportunities.csv" | head -10 >> "$ANALYSIS_DIR/BASELINE_METRICS.md" || true
fi

cat >> "$ANALYSIS_DIR/BASELINE_METRICS.md" << 'EOF'
```

## Next Steps

1. Review fusion opportunities in: `reports/analysis/fusion_opportunities.csv`
2. View detailed timelines in Nsight Systems GUI
3. Implement fused kernels per `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md`
4. Re-profile after fusion to measure speedup

## Files Generated

- Kernel metrics: `reports/analysis/*_ncu_metrics.csv`
- Top kernels: `reports/analysis/*_top_kernels.csv`
- Fusion opportunities: `reports/analysis/fusion_opportunities.csv`
- Raw data: `reports/csv/`

EOF

echo -e "${GREEN}✓ Baseline metrics report generated${NC}"
echo

echo -e "${YELLOW}[5/5] Summary${NC}"
echo
echo -e "${BLUE}Key Files:${NC}"
echo "  📊 Fusion opportunities:  $ANALYSIS_DIR/fusion_opportunities.csv"
echo "  📈 Baseline metrics:      $ANALYSIS_DIR/BASELINE_METRICS.md"
echo "  📁 All analysis files:    $ANALYSIS_DIR/"
echo
echo -e "${BLUE}Next Actions:${NC}"
echo "  1. Review: cat $ANALYSIS_DIR/BASELINE_METRICS.md"
echo "  2. Open timelines: nsys-ui $REPORTS_DIR/nsys/*/*.nsys-rep"
echo "  3. Implement fusion per guide: CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md"
echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Analysis Complete!                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
