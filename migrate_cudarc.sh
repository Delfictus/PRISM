#!/bin/bash
# Comprehensive CUDA API migration script from old cudarc to cudarc 0.9

set -e

echo "=========================================="
echo "CUDA API Migration to cudarc 0.9"
echo "=========================================="

# Find all Rust files that might use CUDA
GPU_FILES=$(find . -type f -name "*.rs" \
    -path "*/gpu*.rs" -o \
    -path "*/cuda*.rs" -o \
    -path "*/*gpu*.rs" | \
    grep -v "target/" | \
    grep -v ".backup" | \
    sort -u)

echo "Found $(echo "$GPU_FILES" | wc -l) GPU-related files to check"

migrated=0
skipped=0

for file in $GPU_FILES; do
    # Check if file contains old API patterns
    if grep -q "default_stream\|\.load_module\|\.load_function\|memcpy_stod\|memcpy_dtov\|launch_builder" "$file" 2>/dev/null; then
        echo ""
        echo "Migrating: $file"

        # Create backup
        cp "$file" "$file.pre-migration-backup"

        # Pattern 1: Change context to device in struct definitions
        sed -i 's/context: Arc<CudaDevice>/device: Arc<CudaDevice>/g' "$file"
        sed -i 's/let context = CudaDevice/let device = CudaDevice/g' "$file"

        # Pattern 2: Replace default_stream() calls
        sed -i 's/self\.context\.default_stream()/\/\/ stream removed in cudarc 0.9/g' "$file"
        sed -i 's/let stream = self\.context\.default_stream();/\/\/ stream removed in cudarc 0.9/g' "$file"
        sed -i 's/let stream = context\.default_stream();/\/\/ stream removed in cudarc 0.9/g' "$file"
        sed -i 's/let stream = self\.device\.default_stream();/\/\/ stream removed in cudarc 0.9/g' "$file"

        # Pattern 3: Replace memory operations
        sed -i 's/stream\.memcpy_stod/self.device.htod_sync_copy/g' "$file"
        sed -i 's/stream\.memcpy_dtov/self.device.dtoh_sync_copy/g' "$file"
        sed -i 's/stream\.alloc_zeros/self.device.alloc_zeros/g' "$file"

        # Pattern 4: Replace synchronize
        sed -i 's/stream\.synchronize()/self.device.synchronize()/g' "$file"

        # Pattern 5: Update struct field references
        sed -i 's/self\.context\./self.device./g' "$file"
        sed -i 's/context\./device./g' "$file"

        # Pattern 6: Fix Ok(Self { context, ... }) to Ok(Self { device, ... })
        sed -i 's/Ok(Self {$/Ok(Self {/g' "$file"
        sed -i 's/context,$/device,/g' "$file"

        echo "✓ Migrated: $file"
        ((migrated++))
    else
        ((skipped++))
    fi
done

echo ""
echo "=========================================="
echo "Migration Summary:"
echo "  Migrated: $migrated files"
echo "  Skipped:  $skipped files (no old API found)"
echo "=========================================="
echo ""
echo "NOTE: Manual fixes still needed for:"
echo "  1. load_module() -> load_ptx()"
echo "  2. load_function() -> get_func()"
echo "  3. launch_builder() -> direct launch()"
echo ""
echo "Run 'cargo build' to check for remaining errors."
