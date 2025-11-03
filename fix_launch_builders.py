#!/usr/bin/env python3
"""Fix remaining launch_builder patterns in CUDA code."""

import re
import os
from pathlib import Path

def fix_launch_builder_pattern(content):
    """Fix launch_builder patterns to new cudarc 0.9 API."""

    # Pattern 1: Find launch_builder blocks
    # Match: let mut launch_args = self.device.launch_builder(&func);
    #        launch_args.arg(&var1);
    #        launch_args.arg(&var2);
    #        ...
    #        unsafe { launch_args.launch(config)?; }

    pattern = re.compile(
        r'let mut (\w+) = self\.device\.launch_builder\(&(\w+)\);(.*?)unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
        re.DOTALL
    )

    def replace_launch(match):
        var_name = match.group(1)
        func_name = match.group(2)
        args_block = match.group(3)
        config_name = match.group(4)

        # Extract all .arg() calls
        arg_pattern = re.compile(r'(\w+)\.arg\(([^)]+)\);')
        args = []

        # Also capture any let statements for variables
        let_statements = []
        lines = args_block.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('let '):
                let_statements.append(line)
            elif '.arg(' in line:
                arg_match = arg_pattern.search(line)
                if arg_match:
                    arg_value = arg_match.group(2)
                    # Clean up the argument
                    if arg_value.startswith('&'):
                        args.append(arg_value)
                    else:
                        args.append(arg_value)

        # Build replacement
        replacement = ""
        if let_statements:
            replacement += '\n'.join(let_statements) + '\n\n'

        replacement += f"""unsafe {{
            {func_name}.launch(
                {config_name},
                ({', '.join(args)})
            )?;
        }}"""

        return replacement

    # Apply the fix
    fixed = pattern.sub(replace_launch, content)

    # Also fix simpler patterns without intermediate variables
    simple_pattern = re.compile(
        r'let mut (\w+) = self\.device\.launch_builder\(&(\w+)\);\s*'
        r'((?:\s*\w+\.arg\([^)]+\);\s*)*)'
        r'unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
        re.MULTILINE
    )

    def replace_simple(match):
        func_name = match.group(2)
        args_block = match.group(3)
        config_name = match.group(4)

        # Extract arguments
        arg_pattern = re.compile(r'\.arg\(([^)]+)\)')
        args = arg_pattern.findall(args_block)

        return f"""unsafe {{
            {func_name}.launch(
                {config_name},
                ({', '.join(args)})
            )?;
        }}"""

    fixed = simple_pattern.sub(replace_simple, fixed)

    return fixed

def process_file(filepath):
    """Process a single file."""
    print(f"Processing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    if 'launch_builder' not in content:
        print(f"  No launch_builder found, skipping")
        return False

    # Create backup
    backup_path = f"{filepath}.launch-backup"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"  Created backup: {backup_path}")

    # Fix the content
    fixed = fix_launch_builder_pattern(content)

    if fixed != content:
        with open(filepath, 'w') as f:
            f.write(fixed)
        print(f"  ✓ Fixed launch_builder patterns")
        return True
    else:
        print(f"  No changes needed")
        return False

# Files to process
files_to_fix = [
    "foundation/cma/transfer_entropy_gpu.rs",
    "foundation/cma/quantum/pimc_gpu.rs",
    "foundation/active_inference/gpu.rs",
    "foundation/active_inference/gpu_inference.rs",
    "foundation/active_inference/gpu_policy_eval.rs",
    "foundation/orchestration/routing/gpu_transfer_entropy_router.rs",
    "foundation/orchestration/thermodynamic/gpu_thermodynamic_consensus.rs",
    "foundation/orchestration/local_llm/gpu_transformer.rs",
    "foundation/statistical_mechanics/gpu.rs",
    "foundation/statistical_mechanics/gpu_integration.rs",
    "foundation/gpu/gpu_tensor_optimized.rs",
    "foundation/gpu/optimized_gpu_tensor.rs",
    "foundation/pwsa/gpu_classifier.rs",
]

if __name__ == "__main__":
    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if process_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")

    print(f"\n✓ Fixed {fixed_count} files with launch_builder patterns")