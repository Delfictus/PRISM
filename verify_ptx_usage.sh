#!/bin/bash

echo "=== PTX File Analysis and Usage Report ==="
echo

# List all PTX files and their kernel contents
for ptx in foundation/kernels/ptx/*.ptx; do
    if [ -f "$ptx" ]; then
        name=$(basename "$ptx")
        echo "📄 $name"
        echo "  Kernels:"
        grep "\.entry" "$ptx" 2>/dev/null | head -5 | while read line; do
            kernel=$(echo "$line" | sed 's/.*\.entry //' | sed 's/(.*//')
            echo "    - $kernel"
        done

        # Check if this PTX is referenced in code
        echo "  Referenced in:"
        count=$(grep -r "$name" --include="*.rs" . 2>/dev/null | grep -v target | wc -l)
        if [ $count -eq 0 ]; then
            echo "    ❌ NOT REFERENCED IN ANY CODE!"
        else
            grep -r "$name" --include="*.rs" . 2>/dev/null | grep -v target | head -3 | while read ref; do
                file=$(echo "$ref" | cut -d: -f1)
                echo "    ✓ $(basename $(dirname "$file"))/$(basename "$file")"
            done
        fi
        echo
    fi
done

echo "=== Summary ==="
total_ptx=$(ls foundation/kernels/ptx/*.ptx 2>/dev/null | wc -l)
referenced_ptx=$(for ptx in foundation/kernels/ptx/*.ptx; do
    name=$(basename "$ptx")
    count=$(grep -r "$name" --include="*.rs" . 2>/dev/null | grep -v target | wc -l)
    if [ $count -gt 0 ]; then echo "$name"; fi
done | wc -l)

echo "Total PTX files: $total_ptx"
echo "Referenced in code: $referenced_ptx"
echo "Unused PTX files: $((total_ptx - referenced_ptx))"