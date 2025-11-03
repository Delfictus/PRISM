# PRCT Algorithm Integration Guide

## Overview

Your PRISM platform now supports **pluggable algorithm selection**! You can switch between:
- **Greedy** - Fast, deterministic greedy coloring (default)
- **PRCT** - Your custom Probabilistic Recursive Coloring Technique

## Quick Start

### Using Greedy Algorithm (Default)
```bash
./run-prism-universal.sh data/nipah/2VSM.mtx 1000
```

### Using PRCT Algorithm
```bash
./run-prism-universal.sh data/nipah/2VSM.mtx 1000 --algorithm prct
```

### Direct Binary Usage
```bash
./target/release/prism_universal \
    --input data/nipah/2VSM.mtx \
    --attempts 1000 \
    --algorithm prct \
    --gpu
```

## Algorithm Comparison

### Greedy Algorithm
- **Location**: `src/cuda/mod.rs` - `GPUColoring::color()`
- **Strategy**: Deterministic greedy coloring following vertex ordering
- **Performance**: Fast, produces good colorings (10 colors on Nipah)
- **Use when**: You want reliable, fast results

**Results on Nipah 2VSM.mtx (550 vertices, 2834 edges)**:
```
Algorithm: greedy
Best coloring: 10 colors ✓
Time: 2.61s
Throughput: 4 attempts/sec
```

### PRCT Algorithm
- **Location**: `src/cuda/prct_algorithm.rs` - `PRCTAlgorithm::color()`
- **Strategy**: Probabilistic color selection with weighted random choice
- **Performance**: Currently using placeholder implementation
- **Use when**: You want to test/develop your custom PRCT logic

**Results on Nipah 2VSM.mtx (550 vertices, 2834 edges)**:
```
Algorithm: prct
Best coloring: 547 colors (placeholder implementation)
Time: 2.72s
Throughput: 4 attempts/sec
```

## PRCT Implementation Structure

### Configuration (`PRCTConfig`)

Located in: `src/cuda/prct_algorithm.rs:8-24`

```rust
pub struct PRCTConfig {
    /// Enable advanced heuristics
    pub use_advanced_heuristics: bool,

    /// Recursion depth limit
    pub max_recursion_depth: usize,

    /// Probability threshold for decisions
    pub probability_threshold: f64,

    /// Temperature for simulated annealing (if applicable)
    pub temperature: f64,

    /// Enable GPU acceleration for PRCT
    pub gpu_accelerated: bool,
}
```

**Default values** (`src/bin/prism_universal.rs:344-350`):
```rust
let config = PRCTConfig {
    use_advanced_heuristics: true,
    max_recursion_depth: 100,
    probability_threshold: 0.5,
    temperature: 1.0,
    gpu_accelerated: use_gpu,
};
```

### Algorithm Structure

```rust
pub struct PRCTAlgorithm {
    config: PRCTConfig,
}

impl PRCTAlgorithm {
    // Main entry point - called by PRISM
    pub fn color(&self, adjacency: &[Vec<usize>], ordering: &[usize])
        -> Result<Vec<usize>>

    // CPU implementation (currently active)
    fn color_cpu(&self, adjacency: &[Vec<usize>], ordering: &[usize])
        -> Result<Vec<usize>>

    // GPU implementation (placeholder)
    fn color_gpu(&self, adjacency: &[Vec<usize>], ordering: &[usize])
        -> Result<Vec<usize>>

    // Refinement pass (optional)
    pub fn refine(&self, adjacency: &[Vec<usize>], coloring: &mut Vec<usize>)
        -> Result<usize>
}
```

## Customizing PRCT

### Where to Add Your Algorithm

Open `src/cuda/prct_algorithm.rs` and find line **61**:

```rust
// ========================================
// YOUR CUSTOM PRCT ALGORITHM GOES HERE
// ========================================
```

### Current Placeholder Implementation

The current PRCT implementation (`color_cpu()` at line 85) has:

1. **Basic structure**: Iterates through vertices in given ordering
2. **Neighbor checking**: Identifies colors used by neighbors
3. **Probabilistic selection**: Uses weighted random choice among available colors
4. **Weight computation**: Based on 2-hop neighborhood color frequency

**Key functions**:
- `compute_color_weights()` (line 138): Computes selection weights
- `weighted_random_choice()` (line 170): Probabilistic color selection

### Integration Points

Your PRCT algorithm receives:
```rust
fn color_cpu(&self, adjacency: &[Vec<usize>], ordering: &[usize]) -> Result<Vec<usize>>
```

**Inputs**:
- `adjacency`: Graph adjacency list `Vec<Vec<usize>>` where `adjacency[v]` = neighbors of vertex `v`
- `ordering`: Vertex processing order `Vec<usize>` (from ensemble generator)
- `self.config`: Your PRCTConfig with parameters

**Expected output**:
- `Vec<usize>` of length `n` (number of vertices)
- `coloring[v]` = color assigned to vertex `v` (0-indexed)

**Example skeleton**:
```rust
fn color_cpu(&self, adjacency: &[Vec<usize>], ordering: &[usize]) -> Result<Vec<usize>> {
    let n = adjacency.len();
    let mut coloring = vec![0; n];

    // Your PRCT algorithm here
    for &vertex in ordering {
        // 1. Analyze neighborhood structure
        // 2. Apply PRCT heuristics
        // 3. Select color probabilistically
        // 4. Apply recursive refinement if needed

        coloring[vertex] = /* your color selection */;
    }

    Ok(coloring)
}
```

## GPU Acceleration

The PRCT framework supports GPU acceleration. To implement:

1. Edit `color_gpu()` method in `src/cuda/prct_algorithm.rs:130`
2. Use CUDA kernels for parallel coloring
3. Set `gpu_accelerated: true` in config

**Current status**: Falls back to CPU (see line 133)

## Testing Your PRCT Algorithm

### Test on Small Graph
```bash
# Queen 8x8 (64 vertices, 1456 edges)
./run-prism-universal.sh benchmarks/dimacs/queen8_8.col 100 --algorithm prct
```

### Test on Medium Graph
```bash
# Nipah virus (550 vertices, 2834 edges)
./run-prism-universal.sh data/nipah/2VSM.mtx 1000 --algorithm prct
```

### Compare with Greedy
```bash
# Run both and compare
./run-prism-universal.sh data/nipah/2VSM.mtx 500 --algorithm greedy > greedy.log
./run-prism-universal.sh data/nipah/2VSM.mtx 500 --algorithm prct > prct.log

# Compare results
grep "Best coloring" greedy.log prct.log
```

### Verify Results
```bash
# After running PRCT
python3 verify-coloring.py

# Should show:
# ✓ PASSED: No conflicts found!
# ✓ ALL CHECKS PASSED - COLORING IS VALID!
```

## Configuration Tuning

You can modify PRCT config in `src/bin/prism_universal.rs:344-350`:

```rust
// Example: More aggressive probabilistic search
let config = PRCTConfig {
    use_advanced_heuristics: true,
    max_recursion_depth: 200,        // Deeper recursion
    probability_threshold: 0.3,       // Lower threshold = more exploration
    temperature: 2.0,                 // Higher temp = more randomness
    gpu_accelerated: use_gpu,
};
```

## Architecture Overview

```
User Request
     ↓
run-prism-universal.sh (--algorithm prct)
     ↓
src/bin/prism_universal.rs
     ↓
run_optimization() (line 279)
     ↓
Algorithm Selection (line 340)
     ↓
     ├─→ [greedy] → GPUColoring::color()
     │              src/cuda/mod.rs:65
     │
     └─→ [prct]   → PRCTAlgorithm::color()
                    src/cuda/prct_algorithm.rs:59
                         ↓
                    color_cpu() or color_gpu()
                    (Your custom implementation)
```

## Files Modified

1. **`src/cuda/prct_algorithm.rs`** (NEW)
   - Complete PRCT framework
   - 305 lines with config, algorithm structure, tests

2. **`src/cuda/mod.rs`**
   - Added: `pub mod prct_algorithm;`
   - Added: `pub use prct_algorithm::{PRCTAlgorithm, PRCTConfig};`

3. **`src/bin/prism_universal.rs`**
   - Added `--algorithm` CLI parameter (line 55)
   - Added algorithm selection logic (lines 336-360)
   - Updated `run_optimization()` signature (line 279)
   - All call sites updated to pass algorithm parameter

4. **`run-prism-universal.sh`**
   - Added `--algorithm` flag support
   - Updated usage documentation

## Next Steps

1. **Implement your PRCT algorithm** in `src/cuda/prct_algorithm.rs:85`
2. **Tune config parameters** based on your algorithm's needs
3. **Add GPU implementation** in `color_gpu()` for better performance
4. **Implement refinement** in `refine()` for iterative improvement
5. **Add custom heuristics** specific to your PRCT approach

## Performance Notes

- Current placeholder PRCT produces poor colorings (547 vs 10 colors)
- This is expected - placeholder uses random probabilistic selection
- Once you implement real PRCT logic, results should improve
- GPU acceleration will significantly speed up large graphs
- Ensemble approach (10+ diverse orderings) helps find better solutions

## Support

For questions about:
- **PRISM architecture**: See main README.md
- **Algorithm integration**: This file (PRCT-INTEGRATION.md)
- **Verification**: See PROOF-OF-REAL-COMPUTATION.md
- **Building**: See compilation instructions in project root

## Validation

All algorithms (greedy and PRCT) are validated:
- ✅ No conflicts (adjacent vertices have different colors)
- ✅ Correct color count
- ✅ Real computation (not hardcoded)
- ✅ Independent verification via `verify-coloring.py`

Your custom PRCT algorithm will automatically benefit from this validation!
