#!/usr/bin/env python3
"""Fix ALL remaining launch_builder patterns in CUDA code."""

import os
import re
from pathlib import Path

def fix_launch_builder_comprehensive(filepath):
    """Fix all launch_builder patterns in a file."""
    print(f"\n📄 Processing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    if 'launch_builder' not in content:
        print("  ✓ No launch_builder patterns")
        return False

    original = content

    # Pattern 1: Simple launch_builder without intermediate variables
    # let mut launch = self.device.launch_builder(&kernel);
    # launch.arg(&var);
    # unsafe { launch.launch(config)?; }
    pattern1 = re.compile(
        r'let mut (\w+) = self\.(\w+)\.launch_builder\(&(\w+)\);(.*?)unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
        re.DOTALL
    )

    def replace_simple(match):
        var_name = match.group(1)
        device_field = match.group(2)
        kernel_var = match.group(3)
        args_block = match.group(4)
        config_var = match.group(5)

        # Extract arguments
        arg_pattern = re.compile(r'\w+\.arg\(([^)]+)\);')
        args = []
        lets = []

        for line in args_block.split('\n'):
            if 'let ' in line:
                lets.append(line.strip())
            elif '.arg(' in line:
                m = arg_pattern.search(line)
                if m:
                    args.append(m.group(1))

        # Build replacement
        result = ""
        if lets:
            result = '\n'.join(['        ' + l for l in lets]) + '\n\n'

        result += f"""        unsafe {{
            {kernel_var}.launch(
                {config_var},
                ({', '.join(args)})
            )?;
        }}"""

        return result

    content = pattern1.sub(replace_simple, content)

    # Pattern 2: launch_builder with explicit builder variable
    # let mut builder = device.launch_builder(&kernel);
    pattern2 = re.compile(
        r'let mut (\w+) = (\w+)\.launch_builder\(&(\w+)\);(.*?)unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
        re.DOTALL
    )

    def replace_device(match):
        var_name = match.group(1)
        device_var = match.group(2)
        kernel_var = match.group(3)
        args_block = match.group(4)
        config_var = match.group(5)

        # Extract arguments
        arg_pattern = re.compile(r'\w+\.arg\(([^)]+)\);')
        args = []
        lets = []

        for line in args_block.split('\n'):
            if 'let ' in line:
                lets.append(line.strip())
            elif '.arg(' in line:
                m = arg_pattern.search(line)
                if m:
                    args.append(m.group(1))

        # Build replacement
        result = ""
        if lets:
            result = '\n'.join(['        ' + l for l in lets]) + '\n\n'

        result += f"""        unsafe {{
            {kernel_var}.launch(
                {config_var},
                ({', '.join(args)})
            )?;
        }}"""

        return result

    content = pattern2.sub(replace_device, content)

    # Pattern 3: launch_builder with function call
    # let mut launch = device.launch_builder(kernel_func);
    pattern3 = re.compile(
        r'let mut (\w+) = (\w+)\.launch_builder\((\w+)\);(.*?)unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
        re.DOTALL
    )

    content = pattern3.sub(replace_device, content)

    # Save if changed
    if content != original:
        # Backup
        backup_path = f"{filepath}.launcher-backup"
        with open(backup_path, 'w') as f:
            f.write(original)

        # Write fixed
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✅ Fixed launch_builder patterns")
        print(f"  💾 Backup: {backup_path}")
        return True
    else:
        print("  ℹ️ No patterns matched")
        return False

def main():
    """Fix all launch_builder patterns."""
    print("=" * 60)
    print("🔧 FIXING ALL launch_builder PATTERNS")
    print("=" * 60)

    # List of files with launch_builder errors
    files_to_fix = [
        "src/cma/quantum/pimc_gpu.rs",
        "src/cma/transfer_entropy_gpu.rs",
        "foundation/active_inference/gpu_inference.rs",
        "foundation/active_inference/gpu_policy_eval.rs",
        "foundation/active_inference/gpu.rs",
        "foundation/cma/quantum/pimc_gpu.rs",
        "foundation/cma/transfer_entropy_gpu.rs",
        "foundation/gpu/gpu_tensor_optimized.rs",
        "foundation/gpu/kernel_executor.rs",
        "foundation/orchestration/local_llm/gpu_transformer.rs",
        "foundation/phase6/gpu_tda.rs",
        "foundation/quantum_mlir/cuda_kernels.rs",
        "foundation/statistical_mechanics/gpu_bindings.rs",
        "foundation/statistical_mechanics/gpu.rs",
    ]

    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_launch_builder_comprehensive(filepath):
                fixed_count += 1
        else:
            print(f"❌ File not found: {filepath}")

    print("\n" + "=" * 60)
    print(f"✅ Fixed {fixed_count} files")
    print("=" * 60)

    return fixed_count

if __name__ == "__main__":
    main()