#!/usr/bin/env python3
"""
Fix remaining stream references after initial migration.
This handles cases missed by the first pass.
"""

import re
from pathlib import Path

files_with_stream = [
    "foundation/neuromorphic/src/gpu_reservoir.rs",
    "foundation/neuromorphic/src/cuda_kernels.rs",
    "foundation/neuromorphic/src/gpu_memory.rs",
    "foundation/quantum/src/gpu_tsp.rs",
    "foundation/quantum/src/gpu_k_opt.rs",
    "foundation/quantum/src/gpu_coloring.rs",
    "foundation/cma/transfer_entropy_gpu.rs",
    "foundation/cma/quantum/pimc_gpu.rs",
    "foundation/information_theory/gpu.rs",
    "foundation/quantum_mlir/cuda_kernels.rs",
    "foundation/statistical_mechanics/gpu_bindings.rs",
    "foundation/statistical_mechanics/gpu.rs",
    "foundation/active_inference/gpu_inference.rs",
    "foundation/active_inference/gpu_policy_eval.rs",
    "foundation/active_inference/gpu.rs",
    "foundation/gpu_coloring.rs",
    "foundation/gpu/gpu_tensor_optimized.rs",
    "foundation/gpu/optimized_gpu_tensor.rs",
    "foundation/orchestration/local_llm/gpu_transformer.rs",
    "foundation/orchestration/thermodynamic/gpu_thermodynamic_consensus.rs",
    "foundation/orchestration/thermodynamic/optimized_thermodynamic_consensus.rs",
    "foundation/phase6/gpu_tda.rs",
    "foundation/gpu/kernel_executor.rs",
    "src/cma/quantum/pimc_gpu.rs",
    "src/cma/transfer_entropy_gpu.rs",
]

root = Path("/home/diddy/Desktop/PRISM-FINNAL-PUSH")

fixed_count = 0

for rel_path in files_with_stream:
    filepath = root / rel_path

    if not filepath.exists():
        continue

    content = filepath.read_text()
    original = content

    # Replace remaining stream operations
    content = re.sub(r'\bstream\.alloc_zeros', 'self.device.alloc_zeros', content)
    content = re.sub(r'\bstream\.htod_sync_copy', 'self.device.htod_sync_copy', content)
    content = re.sub(r'\bstream\.dtoh_sync_copy', 'self.device.dtoh_sync_copy', content)
    content = re.sub(r'\bstream\.synchronize', 'self.device.synchronize', content)
    content = re.sub(r'\bstream\.launch_builder', 'self.device.launch_builder', content)

    if content != original:
        filepath.write_text(content)
        print(f"✓ Fixed: {rel_path}")
        fixed_count += 1

print(f"\nFixed {fixed_count} files")
