# 🚀 PRISM CLI Complete Access Guide

## Overview

The PRISM platform provides multiple CLI tools for different purposes. This guide shows you how to access and use **ALL** of them, with the new **configuration management CLI** being the most powerful for hyperparameter tuning.

---

## 📋 Available CLI Tools

### 1. **prism-config** - Complete Configuration Management (NEW!)
**Purpose**: Full control over ALL parameters with validation, tuning, and verification

```bash
# Access the configuration CLI
./target/release/prism-config --help
```

### 2. **prism_universal** - Universal Data Processing
**Purpose**: Process any data type (graphs, proteins, matrices) with PRISM

```bash
# Access the universal CLI
./target/release/prism_universal --help
```

### 3. **meta-flagsctl** - Feature Flag Control
**Purpose**: Enable/disable meta-level features

```bash
# Access feature control
./target/release/meta-flagsctl --help
```

### 4. **World Record Examples** - Direct Algorithm Access
**Purpose**: Run world record attempts directly

```bash
# Run world record attempt
./target/release/examples/world_record_dsjc1000 <config.toml>
```

---

## 🎯 **prism-config**: The Ultimate Configuration CLI

### **Complete Feature List**

#### 1. List All Parameters
```bash
# List all parameters with their current values
./target/release/prism-config list

# Filter by category
./target/release/prism-config list --category gpu
./target/release/prism-config list --category thermo
./target/release/prism-config list --category quantum
./target/release/prism-config list --category memetic
./target/release/prism-config list --category phases

# Show type information
./target/release/prism-config list --types

# Show only modified parameters
./target/release/prism-config list --modified
```

**Output Example**:
```
═══════════════════════════════════════════════════════════
                 PRISM CONFIGURATION PARAMETERS
═══════════════════════════════════════════════════════════

► GPU
────────────────────────────────────────────────────────────────────────
  GPU   gpu.batch_size                      =       1024
      Range: [32 .. 8192]
      GPU batch size for parallel operations
  GPU ® gpu.device_id                       =          0
      Range: [0 .. 8]
      CUDA device ID

► THERMO
────────────────────────────────────────────────────────────
  GPU   thermo.replicas                     =         56
      Range: [1 .. 56]
      Number of temperature replicas (VRAM limited to 56 for 8GB)

Legend: GPU = Affects GPU | ® = Requires Restart
```

#### 2. Get Specific Parameter Value
```bash
# Get simple value
./target/release/prism-config get thermo.replicas
# Output: 56

# Get with full metadata
./target/release/prism-config get thermo.replicas --verbose

# Get from specific config file
./target/release/prism-config get thermo.replicas --config foundation/prct-core/configs/world_record.toml
```

#### 3. Set Parameter Values
```bash
# Set a parameter value
./target/release/prism-config set thermo.replicas 48 --config my_config.toml

# Validate without applying (dry run)
./target/release/prism-config set thermo.replicas 100 --config my_config.toml --dry-run
# Error: thermo.replicas above maximum: 100 > 56

# Set multiple parameters
./target/release/prism-config set gpu.batch_size 2048 --config my_config.toml
./target/release/prism-config set quantum.iterations 50 --config my_config.toml
./target/release/prism-config set memetic.population_size 512 --config my_config.toml
```

#### 4. Validate Configurations
```bash
# Basic validation
./target/release/prism-config validate my_config.toml

# GPU memory validation
./target/release/prism-config validate my_config.toml --gpu

# Deep validation (checks dependencies)
./target/release/prism-config validate my_config.toml --gpu --deep
```

#### 5. Generate Configuration Files
```bash
# Generate minimal config
./target/release/prism-config generate minimal_config.toml --template minimal

# Generate world record config
./target/release/prism-config generate wr_config.toml --template world-record

# Generate full config with ALL parameters
./target/release/prism-config generate full_config.toml --template full
```

#### 6. Compare and Merge Configurations
```bash
# Show differences
./target/release/prism-config diff baseline.toml optimized.toml

# Merge multiple configs
./target/release/prism-config merge final.toml base.toml layer1.toml layer2.toml
```

#### 7. Interactive Tuning
```bash
# Tune all parameters interactively
./target/release/prism-config tune my_config.toml

# Tune specific category
./target/release/prism-config tune my_config.toml --category thermo
```

#### 8. Reset Parameters
```bash
# Reset all to defaults
./target/release/prism-config reset my_config.toml

# Reset specific category
./target/release/prism-config reset my_config.toml --category gpu
```

---

## 🔧 Real-World Usage Examples

### Example 1: Optimize for World Record
```bash
# 1. Generate world record config
./target/release/prism-config generate wr.toml --template world-record

# 2. Tune GPU parameters
./target/release/prism-config set gpu.batch_size 4096 --config wr.toml
./target/release/prism-config set gpu.streams 16 --config wr.toml

# 3. Maximize thermodynamic replicas (within VRAM)
./target/release/prism-config set thermo.replicas 56 --config wr.toml
./target/release/prism-config set thermo.num_temps 56 --config wr.toml

# 4. Validate VRAM usage
./target/release/prism-config validate wr.toml --gpu

# 5. Run world record attempt
./target/release/examples/world_record_dsjc1000 wr.toml
```

### Example 2: Quick Testing Configuration
```bash
# 1. Generate minimal config for fast testing
./target/release/prism-config generate test.toml --template minimal

# 2. Set for speed over quality
./target/release/prism-config set max_runtime_hours 0.1 --config test.toml
./target/release/prism-config set thermo.replicas 8 --config test.toml
./target/release/prism-config set quantum.iterations 5 --config test.toml

# 3. Run quick test
./target/release/prism_universal --input benchmarks/dimacs/DSJC125.5.col --config test.toml
```

### Example 3: A/B Testing Configurations
```bash
# 1. Create baseline
./target/release/prism-config generate baseline.toml --template full

# 2. Create variant A (focus on thermodynamic)
cp baseline.toml variant_a.toml
./target/release/prism-config set use_thermodynamic_equilibration true --config variant_a.toml
./target/release/prism-config set thermo.t_max 50.0 --config variant_a.toml
./target/release/prism-config set use_quantum_classical_hybrid false --config variant_a.toml

# 3. Create variant B (focus on quantum)
cp baseline.toml variant_b.toml
./target/release/prism-config set use_thermodynamic_equilibration false --config variant_b.toml
./target/release/prism-config set use_quantum_classical_hybrid true --config variant_b.toml
./target/release/prism-config set quantum.iterations 100 --config variant_b.toml

# 4. Compare configurations
./target/release/prism-config diff variant_a.toml variant_b.toml

# 5. Run both and compare results
./target/release/examples/world_record_dsjc1000 variant_a.toml > result_a.log
./target/release/examples/world_record_dsjc1000 variant_b.toml > result_b.log
```

### Example 4: Hyperparameter Grid Search
```bash
#!/bin/bash
# grid_search.sh - Automated hyperparameter search

for replicas in 16 32 48 56; do
  for temps in 8 16 32 48; do
    for batch in 512 1024 2048 4096; do
      # Generate config name
      config="grid_r${replicas}_t${temps}_b${batch}.toml"
      
      # Create config
      ./target/release/prism-config generate $config --template minimal
      
      # Set parameters
      ./target/release/prism-config set thermo.replicas $replicas --config $config
      ./target/release/prism-config set thermo.num_temps $temps --config $config
      ./target/release/prism-config set gpu.batch_size $batch --config $config
      
      # Validate
      if ./target/release/prism-config validate $config --gpu; then
        # Run experiment
        echo "Testing $config..."
        timeout 300 ./target/release/examples/world_record_dsjc1000 $config > results/$config.log
      fi
    done
  done
done
```

---

## 📊 Parameter Categories Reference

### GPU Parameters
- `gpu.device_id` - CUDA device to use
- `gpu.batch_size` - Parallel batch size
- `gpu.streams` - Number of CUDA streams
- `gpu.enable_*` - GPU acceleration flags

### Thermodynamic Parameters
- `thermo.replicas` - Temperature replicas (VRAM limited)
- `thermo.num_temps` - Temperature levels
- `thermo.t_min/t_max` - Temperature range
- `thermo.steps_per_temp` - Equilibration steps

### Quantum Parameters
- `quantum.iterations` - Solver iterations
- `quantum.target_chromatic` - Target colors
- `quantum.failure_retries` - Retry attempts

### Memetic Algorithm Parameters
- `memetic.population_size` - Population size
- `memetic.generations` - Number of generations
- `memetic.mutation_rate` - Mutation probability
- `memetic.local_search_depth` - Local search iterations

### Phase Toggles
- `use_reservoir_prediction` - Neuromorphic reservoir
- `use_active_inference` - Active inference optimization
- `use_transfer_entropy` - Transfer entropy analysis
- `use_thermodynamic_equilibration` - Thermodynamic phase
- `use_quantum_classical_hybrid` - Quantum solver
- `use_multiscale_analysis` - Multi-scale analysis
- `use_ensemble_consensus` - Consensus voting
- `use_geodesic_features` - Geodesic features

---

## 🎮 Quick Start Commands

### Essential Commands for Immediate Use

```bash
# 1. See all available parameters
./target/release/prism-config list

# 2. Generate a config to work with
./target/release/prism-config generate my_config.toml --template world-record

# 3. Modify key parameters
./target/release/prism-config set thermo.replicas 48 --config my_config.toml
./target/release/prism-config set gpu.batch_size 2048 --config my_config.toml

# 4. Validate before running
./target/release/prism-config validate my_config.toml --gpu --deep

# 5. Run with your config
./target/release/examples/world_record_dsjc1000 my_config.toml
```

---

## 🔥 Power User Tips

### 1. Layered Configuration Strategy
```bash
# Create base layer
./target/release/prism-config generate base.toml --template minimal

# Create optimization layers
echo "gpu.batch_size = 4096" > gpu_layer.toml
echo "thermo.replicas = 56" >> gpu_layer.toml

echo "quantum.iterations = 100" > algorithm_layer.toml
echo "memetic.generations = 5000" >> algorithm_layer.toml

# Merge all layers
./target/release/prism-config merge final.toml base.toml gpu_layer.toml algorithm_layer.toml
```

### 

### 2. Batch Parameter Updates
```bash
# Create a parameter update script
cat > update_params.sh << 'SCRIPT'
#!/bin/bash
CONFIG=$1
./target/release/prism-config set gpu.batch_size 4096 --config $CONFIG
./target/release/prism-config set gpu.streams 16 --config $CONFIG
./target/release/prism-config set thermo.replicas 56 --config $CONFIG
./target/release/prism-config set quantum.iterations 50 --config $CONFIG
./target/release/prism-config validate $CONFIG --gpu --deep
SCRIPT

chmod +x update_params.sh
./update_params.sh my_config.toml
```

### 3. Parameter Sweeps with Validation
```bash
# Sweep replicas while checking VRAM
for r in 8 16 24 32 40 48 56; do
  ./target/release/prism-config set thermo.replicas $r --config test.toml
  if ./target/release/prism-config validate test.toml --gpu 2>/dev/null; then
    echo "Replicas=$r: VALID"
    ./target/release/examples/world_record_dsjc1000 test.toml
  else
    echo "Replicas=$r: VRAM exceeded"
  fi
done
```

---

## 📝 Summary

The **prism-config** CLI gives you:
- **Complete control** over 200+ parameters
- **Validation** to prevent errors
- **VRAM checking** to prevent GPU OOM
- **Diff/merge** for configuration management
- **Interactive tuning** for exploration
- **Templates** for quick starts

### Key Benefits:
✅ No source code changes needed for tuning
✅ All parameters validated before use
✅ VRAM limits enforced automatically
✅ Easy A/B testing and comparison
✅ Scriptable for automation
✅ Beautiful colored output

### Best Practices:
1. Always validate configs before running
2. Use templates as starting points
3. Check VRAM when increasing replicas/batch sizes
4. Use diff to understand changes
5. Keep baseline configs for comparison

---

## 🚀 Get Started Now!

```bash
# Your first command - see what you can tune:
./target/release/prism-config list

# Generate a config and start experimenting:
./target/release/prism-config generate experiment.toml --template world-record

# You now have COMPLETE CONTROL over PRISM!
```

