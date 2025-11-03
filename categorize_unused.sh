#!/bin/bash

OUTPUT_DIR="complete_kernel_analysis_20251026_105230"

echo "=== UNUSED KERNELS BY PTX FILE ==="
echo

for ptx in foundation/kernels/ptx/*.ptx; do
    if [ -f "$ptx" ]; then
        name=$(basename "$ptx")
        echo "📄 $name:"

        # Extract all kernels from this PTX
        grep "\.entry" "$ptx" 2>/dev/null | sed 's/.*\.entry //' | sed 's/(.*//' | while read kernel; do
            # Check if this kernel is in the unused list
            if grep -Fxq "$kernel" "$OUTPUT_DIR/unused_kernels.txt" 2>/dev/null; then
                echo "  ❌ UNUSED: $kernel"
            fi
        done
        echo
    fi
done

echo
echo "Total unused kernels: $(wc -l < $OUTPUT_DIR/unused_kernels.txt)"
