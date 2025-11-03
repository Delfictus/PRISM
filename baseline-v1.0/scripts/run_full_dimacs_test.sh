#!/bin/bash
# PRISM-AI Full DIMACS Benchmark Test Suite
# Automated test runner with results collection and analysis

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PRISM_DIR="/home/diddy/Desktop/PRISM-FINNAL-PUSH"
BENCHMARK_DIR="/home/diddy/Downloads/PRISM-master/benchmarks/dimacs"
OUTPUT_DIR="${PRISM_DIR}/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${OUTPUT_DIR}/dimacs_results_${TIMESTAMP}.txt"
JSON_RESULTS="${OUTPUT_DIR}/dimacs_results_${TIMESTAMP}.json"
LOG_FILE="${OUTPUT_DIR}/test_log_${TIMESTAMP}.log"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           PRISM-AI Full DIMACS Benchmark Suite                ║${NC}"
echo -e "${BLUE}║                GPU-Accelerated Testing                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Pre-flight checks
echo -e "${YELLOW}[1/6] Running pre-flight checks...${NC}"

# Check if we're in the right directory
if [ ! -d "$PRISM_DIR" ]; then
    echo -e "${RED}Error: PRISM directory not found at $PRISM_DIR${NC}"
    exit 1
fi

cd "$PRISM_DIR"

# Check if benchmark directory exists
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo -e "${RED}Error: Benchmark directory not found at $BENCHMARK_DIR${NC}"
    exit 1
fi

# Check GPU availability
echo -n "  Checking GPU... "
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} $GPU_NAME"
else
    echo -e "${RED}✗ GPU not found${NC}"
    exit 1
fi

# Check if binary exists
BENCHMARK_BIN="./target/release/examples/simple_dimacs_benchmark"
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo -e "${YELLOW}  Binary not found. Building...${NC}"
    cargo build --example simple_dimacs_benchmark --release --features cuda 2>&1 | tee -a "$LOG_FILE"
fi

echo -n "  Checking benchmark binary... "
if [ -f "$BENCHMARK_BIN" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Count benchmark files
BENCHMARK_COUNT=$(ls -1 "$BENCHMARK_DIR"/*.col 2>/dev/null | wc -l)
echo -e "  Found ${GREEN}${BENCHMARK_COUNT}${NC} DIMACS benchmark files"

# Step 2: Create output directory
echo -e "\n${YELLOW}[2/6] Setting up output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
echo -e "  Output dir: ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "  Results file: ${GREEN}${RESULTS_FILE}${NC}"

# Step 3: System information
echo -e "\n${YELLOW}[3/6] Collecting system information...${NC}"

cat > "${OUTPUT_DIR}/system_info_${TIMESTAMP}.txt" <<EOF
PRISM-AI DIMACS Benchmark System Information
============================================
Date: $(date)
Hostname: $(hostname)
User: $(whoami)

GPU Information:
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv)

CPU Information:
$(lscpu | grep -E "Model name|CPU\(s\):|Thread|Core")

Memory:
$(free -h | grep -E "Mem:|Swap:")

PRISM Version:
Directory: $PRISM_DIR
Git: $(git -C "$PRISM_DIR" rev-parse --short HEAD 2>/dev/null || echo "Not a git repo")

Benchmark Location: $BENCHMARK_DIR
Benchmark Count: $BENCHMARK_COUNT files
EOF

cat "${OUTPUT_DIR}/system_info_${TIMESTAMP}.txt"

# Step 4: Enable required feature flags
echo -e "\n${YELLOW}[4/6] Configuring feature flags...${NC}"

if [ -f "./target/release/meta-flagsctl" ]; then
    echo -n "  Enabling meta_generation... "
    ./target/release/meta-flagsctl enable meta_generation \
        --actor "automated_test" \
        --justification "Full DIMACS benchmark suite run at ${TIMESTAMP}" \
        &>>"$LOG_FILE" && echo -e "${GREEN}✓${NC}" || echo -e "${YELLOW}⊘${NC} (may already be enabled)"

    echo -n "  Checking flag status... "
    ./target/release/meta-flagsctl status &>>"$LOG_FILE" && echo -e "${GREEN}✓${NC}"
else
    echo -e "  ${YELLOW}⊘${NC} meta-flagsctl not available, skipping"
fi

# Step 5: Run the full benchmark suite
echo -e "\n${YELLOW}[5/6] Running DIMACS benchmark suite...${NC}"
echo -e "  This may take several minutes for large graphs..."
echo ""

# Start timer
START_TIME=$(date +%s)

# Run benchmark and capture output
echo "Starting benchmark at $(date)" | tee -a "$LOG_FILE"
"$BENCHMARK_BIN" "$BENCHMARK_DIR" 2>&1 | tee "$RESULTS_FILE" "$LOG_FILE"

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "$RESULTS_FILE"
echo "Benchmark completed in ${DURATION} seconds ($(date -u -d @${DURATION} +%T))" | tee -a "$RESULTS_FILE"

# Step 6: Generate analysis
echo -e "\n${YELLOW}[6/6] Generating analysis...${NC}"

# Extract summary statistics
TOTAL_GRAPHS=$(grep -c "^[A-Za-z]" "$RESULTS_FILE" | head -1 || echo "0")
PERFECT_SCORES=$(grep "0.0%" "$RESULTS_FILE" | wc -l)
AVG_GAP=$(grep -E "DSJC|queen|myciel|le450" "$RESULTS_FILE" | awk '{sum+=$9; count++} END {if(count>0) printf "%.1f", sum/count}')

# Create summary report
cat > "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" <<EOF
╔════════════════════════════════════════════════════════════════╗
║              DIMACS Benchmark Summary Report                   ║
╚════════════════════════════════════════════════════════════════╝

Test Run: $TIMESTAMP
Duration: ${DURATION}s ($(date -u -d @${DURATION} +%T))

Results:
--------
Total Graphs Tested: $TOTAL_GRAPHS
Perfect Scores (0% gap): $PERFECT_SCORES
Average Gap to Best Known: ${AVG_GAP}%

GPU Performance:
----------------
GPU: $GPU_NAME
Total GPU Time: $(grep "Time (ms)" "$RESULTS_FILE" | awk '{sum+=$4} END {printf "%.2f seconds", sum/1000}')
Fastest Graph: $(grep -E "DSJC|queen|myciel|le450" "$RESULTS_FILE" | sort -k4 -n | head -1 | awk '{print $1 " - " $4 "ms"}')
Slowest Graph: $(grep -E "DSJC|queen|myciel|le450" "$RESULTS_FILE" | sort -k4 -n | tail -1 | awk '{print $1 " - " $4 "ms"}')

Best Results:
-------------
$(grep "0.0%" "$RESULTS_FILE" | awk '{print "  " $1 ": " $8 " colors (perfect match!)"}')

World Record Target (DSJC1000.5):
----------------------------------
$(grep "DSJC1000.5" "$RESULTS_FILE" | awk '{print "  Colors: " $8 " (best known: " $9 ")\n  Gap: " $10 "\n  Time: " $4 "ms"}')

Files Generated:
----------------
Results: ${RESULTS_FILE}
Log: ${LOG_FILE}
System Info: ${OUTPUT_DIR}/system_info_${TIMESTAMP}.txt
Summary: ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt

EOF

cat "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"

# Create JSON output for programmatic access
echo "{" > "$JSON_RESULTS"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$JSON_RESULTS"
echo "  \"duration_seconds\": $DURATION," >> "$JSON_RESULTS"
echo "  \"gpu_name\": \"$GPU_NAME\"," >> "$JSON_RESULTS"
echo "  \"total_graphs\": $TOTAL_GRAPHS," >> "$JSON_RESULTS"
echo "  \"perfect_scores\": $PERFECT_SCORES," >> "$JSON_RESULTS"
echo "  \"average_gap_percent\": ${AVG_GAP:-0}" >> "$JSON_RESULTS"
echo "}" >> "$JSON_RESULTS"

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Test Suite Completed!                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  📊 Results saved to: ${BLUE}${OUTPUT_DIR}${NC}"
echo -e "  📝 View summary: ${BLUE}cat ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt${NC}"
echo -e "  📋 Full results: ${BLUE}cat ${RESULTS_FILE}${NC}"
echo -e "  🔍 JSON data: ${BLUE}cat ${JSON_RESULTS}${NC}"
echo ""

# Offer to view results
read -p "Would you like to view the summary now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    less "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
fi

echo -e "\n${GREEN}✓${NC} All done! Test suite completed successfully."
echo -e "  Run again: ${BLUE}./run_full_dimacs_test.sh${NC}"
