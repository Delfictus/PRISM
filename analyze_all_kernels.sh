#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   COMPLETE GPU KERNEL ANALYSIS - ALL SOURCES                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

OUTPUT_DIR="complete_kernel_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo

# 1. Analyze all .cu files
echo "[1/5] Analyzing all .cu source files..."
find . -name "*.cu" -type f | grep -v target | sort > "$OUTPUT_DIR/cu_files_list.txt"

total_cu_kernels=0
while read cu_file; do
    if [ -f "$cu_file" ]; then
        count=$(grep -c "__global__" "$cu_file" 2>/dev/null || echo 0)
        total_cu_kernels=$((total_cu_kernels + count))
        echo "$count kernels: $cu_file" >> "$OUTPUT_DIR/cu_kernel_counts.txt"

        # Extract kernel names
        grep "__global__" "$cu_file" 2>/dev/null | grep -o "void [a-zA-Z_][a-zA-Z0-9_]*" | cut -d' ' -f2 >> "$OUTPUT_DIR/cu_kernel_names.txt"
    fi
done < "$OUTPUT_DIR/cu_files_list.txt"

echo "  ✓ Found $total_cu_kernels kernels in .cu files"

# 2. Analyze all PTX files
echo "[2/5] Analyzing all PTX files..."
find . -name "*.ptx" -type f | grep -v target | sort > "$OUTPUT_DIR/ptx_files_list.txt"

total_ptx_kernels=0
while read ptx_file; do
    if [ -f "$ptx_file" ]; then
        count=$(grep -c "\.entry" "$ptx_file" 2>/dev/null || echo 0)
        total_ptx_kernels=$((total_ptx_kernels + count))
        echo "$count kernels: $ptx_file" >> "$OUTPUT_DIR/ptx_kernel_counts.txt"

        # Extract kernel names
        grep "\.entry" "$ptx_file" 2>/dev/null | sed 's/.*\.entry //' | sed 's/(.*//' >> "$OUTPUT_DIR/ptx_kernel_names.txt"
    fi
done < "$OUTPUT_DIR/ptx_files_list.txt"

echo "  ✓ Found $total_ptx_kernels kernels in PTX files"

# 3. Analyze embedded kernels in Rust
echo "[3/5] Analyzing embedded CUDA in .rs files..."
find . -name "*.rs" -type f | grep -v target | xargs grep -l "__global__" 2>/dev/null | sort > "$OUTPUT_DIR/rs_files_with_cuda.txt"

total_rs_kernels=0
while read rs_file; do
    if [ -f "$rs_file" ]; then
        count=$(grep -c "__global__" "$rs_file" 2>/dev/null || echo 0)
        total_rs_kernels=$((total_rs_kernels + count))
        echo "$count kernels: $rs_file" >> "$OUTPUT_DIR/rs_kernel_counts.txt"

        # Extract kernel names
        grep "__global__" "$rs_file" 2>/dev/null | grep -o "void [a-zA-Z_][a-zA-Z0-9_]*" | cut -d' ' -f2 >> "$OUTPUT_DIR/rs_kernel_names.txt"
    fi
done < "$OUTPUT_DIR/rs_files_with_cuda.txt"

echo "  ✓ Found $total_rs_kernels kernels in .rs files"

# 4. Check what's actually loaded
echo "[4/5] Finding kernels actually loaded in code..."
grep -rh "load_ptx" . --include="*.rs" 2>/dev/null | grep -v target | grep -v "//" > "$OUTPUT_DIR/load_ptx_calls.txt"
grep -rh "get_func\|load_function" . --include="*.rs" 2>/dev/null | grep -v target | grep -v "//" | grep -o '"[a-zA-Z_][a-zA-Z0-9_]*"' | tr -d '"' | sort -u > "$OUTPUT_DIR/loaded_kernel_names.txt"

loaded_count=$(wc -l < "$OUTPUT_DIR/loaded_kernel_names.txt")
echo "  ✓ Found $loaded_count unique kernels being loaded"

# 5. Generate summary
echo "[5/5] Generating summary report..."

cat > "$OUTPUT_DIR/ANALYSIS_SUMMARY.txt" <<EOF
═══════════════════════════════════════════════════════════════
COMPLETE KERNEL ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════

KERNEL COUNTS BY SOURCE:
  .cu files (CUDA C):        $total_cu_kernels kernels
  .rs files (embedded):      $total_rs_kernels kernels
  PTX files (compiled):      $total_ptx_kernels kernels
  Actually loaded in code:   $loaded_count kernels

TOTAL KERNEL DEFINITIONS: $((total_cu_kernels + total_rs_kernels))

FILES ANALYZED:
  .cu files:  $(wc -l < "$OUTPUT_DIR/cu_files_list.txt")
  .ptx files: $(wc -l < "$OUTPUT_DIR/ptx_files_list.txt")
  .rs files:  $(wc -l < "$OUTPUT_DIR/rs_files_with_cuda.txt")

DETAILED RESULTS:
  See individual *_counts.txt files for breakdowns
  See *_names.txt files for complete kernel lists
  See load_ptx_calls.txt for actual usage

═══════════════════════════════════════════════════════════════
EOF

cat "$OUTPUT_DIR/ANALYSIS_SUMMARY.txt"

echo
echo "✅ Analysis complete! Results saved to: $OUTPUT_DIR/"
echo
echo "Key files:"
echo "  📊 ANALYSIS_SUMMARY.txt - Overview"
echo "  📋 cu_kernel_names.txt - All kernels from .cu files"
echo "  📋 rs_kernel_names.txt - All kernels from .rs files"
echo "  📋 ptx_kernel_names.txt - All kernels in PTX"
echo "  📋 loaded_kernel_names.txt - Kernels actually used"
echo "  📋 load_ptx_calls.txt - All load_ptx invocations"
