#!/bin/bash

echo "Fixing remaining CUDA API issues..."

# Fix all load_function -> get_func
echo "Replacing load_function with get_func..."
find . -type f -name "*.rs" \( -path "*/src/*" -o -path "*/foundation/*" \) -exec grep -l "load_function" {} \; | while read file; do
    echo "  Fixing: $file"
    sed -i 's/load_function/get_func/g' "$file"
done

# Fix all remaining context references
echo "Replacing context with device..."
find . -type f -name "*.rs" \( -path "*/src/*" -o -path "*/foundation/*" \) -exec grep -l "context\." {} \; | while read file; do
    echo "  Fixing: $file"
    sed -i 's/context\./device\./g' "$file"
done

# Fix any remaining default_stream
echo "Fixing default_stream..."
find . -type f -name "*.rs" \( -path "*/src/*" -o -path "*/foundation/*" \) -exec grep -l "default_stream()" {} \; | while read file; do
    echo "  Fixing: $file"
    sed -i 's/default_stream()/fork_default_stream()?/g' "$file"
done

# Fix CudaSlice.len() -> .len
echo "Fixing CudaSlice.len()..."
find . -type f -name "*.rs" \( -path "*/src/*" -o -path "*/foundation/*" \) -exec grep -l "\.len()" {} \; | while read file; do
    # Be more selective - only fix likely CudaSlice variables
    grep -n "gpu.*\.len()\|cuda.*\.len()" "$file" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  Checking: $file"
        # This is tricky, so just report it
    fi
done

echo "Done! Now checking compilation..."
cargo check --features cuda 2>&1 | grep -c "error\[E"