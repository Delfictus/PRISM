#!/usr/bin/env python3
"""
Comprehensive CUDA API migration script from old cudarc to cudarc 0.9
Migrates all 10 required API changes automatically where possible.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

def migrate_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Migrate a single file to cudarc 0.9 API. Returns (changed, warnings)."""

    try:
        content = filepath.read_text()
        original = content
        warnings = []

        # Pattern 1: Change context to device in struct definitions
        content = re.sub(
            r'\bcontext: Arc<CudaDevice>',
            'device: Arc<CudaDevice>',
            content
        )

        # Pattern 2: Change local variable names
        content = re.sub(
            r'\blet context = CudaDevice',
            'let device = CudaDevice',
            content
        )

        # Pattern 3: Remove default_stream() calls and update subsequent operations
        # Replace stream.operation() with device.operation()
        content = re.sub(
            r'let stream = (?:self\.)?(?:context|device)\.default_stream\(\);?\s*\n',
            '',
            content
        )
        content = re.sub(
            r'(?:self\.)?context\.default_stream\(\)',
            '',
            content
        )

        # Pattern 4: Replace memory operations - from stream to device
        content = re.sub(
            r'\bstream\.memcpy_stod\(',
            'self.device.htod_sync_copy(',
            content
        )
        content = re.sub(
            r'\bstream\.memcpy_dtov\(',
            'self.device.dtoh_sync_copy(',
            content
        )
        content = re.sub(
            r'\bstream\.alloc_zeros\(',
            'self.device.alloc_zeros(',
            content
        )

        # Pattern 5: Replace synchronize
        content = re.sub(
            r'\bstream\.synchronize\(\)',
            'self.device.synchronize()',
            content
        )

        # Pattern 6: Update all self.context to self.device
        content = re.sub(
            r'\bself\.context\.',
            'self.device.',
            content
        )

        # Pattern 7: Fix Ok(Self { context, ... }) patterns
        content = re.sub(
            r'(\s+)context,(\s*\n)',
            r'\1device,\2',
            content
        )

        # Pattern 8: Detect load_module patterns that need manual fixing
        if 'load_module' in content:
            warnings.append(f"⚠️  Contains load_module() - needs manual conversion to load_ptx()")

        # Pattern 9: Detect load_function patterns that need manual fixing
        if 'load_function' in content:
            warnings.append(f"⚠️  Contains load_function() - needs manual conversion to get_func()")

        # Pattern 10: Detect launch_builder patterns that need manual fixing
        if 'launch_builder' in content:
            warnings.append(f"⚠️  Contains launch_builder() - needs manual conversion to direct launch()")

        # Pattern 11: Fix CudaSlice.len() to CudaSlice.len (field not method)
        content = re.sub(
            r'\.len\(\)(\s+(?://|/\*).*?(?:field|not method))',
            r'.len\1',
            content
        )

        # Check if anything changed
        if content != original:
            # Write back
            filepath.write_text(content)
            return (True, warnings)
        else:
            return (False, warnings)

    except Exception as e:
        return (False, [f"❌ Error: {str(e)}"])

def main():
    """Main migration function."""

    # Files that need migration (from grep results)
    files_to_migrate = [
        "src/cma/transfer_entropy_gpu.rs",
        "src/cma/quantum/pimc_gpu.rs",
        "src/integration/multi_modal_reasoner.rs",
        "foundation/gpu/kernel_executor.rs",
        "foundation/phase6/gpu_tda.rs",
        "foundation/orchestration/thermodynamic/optimized_thermodynamic_consensus.rs",
        "foundation/orchestration/thermodynamic/gpu_thermodynamic_consensus.rs",
        "foundation/orchestration/local_llm/gpu_transformer.rs",
        "foundation/gpu/optimized_gpu_tensor.rs",
        "foundation/integration/multi_modal_reasoner.rs",
        "foundation/gpu/gpu_tensor_optimized.rs",
        "foundation/gpu_coloring.rs",
        "foundation/active_inference/gpu.rs",
        "foundation/active_inference/gpu_policy_eval.rs",
        "foundation/active_inference/gpu_inference.rs",
        "foundation/statistical_mechanics/gpu.rs",
        "foundation/statistical_mechanics/gpu_bindings.rs",
        "foundation/quantum_mlir/gpu_memory.rs",
        "foundation/quantum_mlir/cuda_kernels.rs",
        "foundation/quantum_mlir/runtime.rs",
        "foundation/information_theory/gpu.rs",
        "foundation/cma/quantum/pimc_gpu.rs",
        "foundation/cma/transfer_entropy_gpu.rs",
        "foundation/quantum/src/gpu_coloring.rs",
        "foundation/quantum/src/gpu_k_opt.rs",
        "foundation/quantum/src/gpu_tsp.rs",
        "foundation/neuromorphic/src/gpu_memory.rs",
        "foundation/neuromorphic/src/cuda_kernels.rs",
        "foundation/neuromorphic/src/gpu_optimization.rs",
        "foundation/neuromorphic/src/gpu_reservoir.rs",
    ]

    root = Path("/home/diddy/Desktop/PRISM-FINNAL-PUSH")

    print("=" * 60)
    print("CUDA API Migration to cudarc 0.9")
    print("=" * 60)
    print()

    migrated_count = 0
    skipped_count = 0
    all_warnings = []

    for rel_path in files_to_migrate:
        filepath = root / rel_path

        if not filepath.exists():
            print(f"⊘  Skipped (not found): {rel_path}")
            skipped_count += 1
            continue

        changed, warnings = migrate_file(filepath)

        if changed:
            print(f"✓  Migrated: {rel_path}")
            migrated_count += 1
            if warnings:
                for warn in warnings:
                    print(f"   {warn}")
                all_warnings.extend([(rel_path, w) for w in warnings])
        else:
            if warnings:
                print(f"⊘  Skipped (errors): {rel_path}")
                for warn in warnings:
                    print(f"   {warn}")
                skipped_count += 1
            else:
                print(f"⊘  Skipped (no changes needed): {rel_path}")
                skipped_count += 1

    print()
    print("=" * 60)
    print(f"Migration Summary:")
    print(f"  Migrated: {migrated_count} files")
    print(f"  Skipped:  {skipped_count} files")
    print(f"  Warnings: {len(all_warnings)} manual fixes needed")
    print("=" * 60)
    print()

    if all_warnings:
        print("Manual fixes still needed:")
        for filepath, warning in all_warnings:
            print(f"  {filepath}: {warning}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
