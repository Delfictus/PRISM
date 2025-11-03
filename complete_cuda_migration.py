#!/usr/bin/env python3
"""Complete the CUDA migration to cudarc 0.9 API."""

import os
import re
from pathlib import Path
import subprocess

def fix_file(filepath):
    """Apply all cudarc 0.9 API fixes to a single file."""
    print(f"\n📁 Processing: {filepath}")

    if not os.path.exists(filepath):
        print(f"  ❌ File not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    fixes_applied = []

    # 1. Fix load_module -> load_ptx
    if 'load_module' in content:
        # Extract PTX loading context
        module_pattern = re.compile(
            r'let (\w+) = (\w+)\.load_module\(([^)]+)\)'
        )

        # Find all functions used with this module
        func_pattern = re.compile(
            r'(\w+)\.(?:load_function|get_func)\("([^"]+)"\)'
        )
        functions = list(set(func_pattern.findall(content)))
        func_names = [f[1] for f in functions]

        if func_names:
            # Build function list
            func_list = ',\n            '.join([f'"{name}"' for name in func_names])

            # Replace load_module
            def replace_load_module(match):
                var_name = match.group(1)
                device_name = match.group(2)
                ptx_expr = match.group(3)

                # Determine module name from context or filepath
                module_name = Path(filepath).stem.replace('_gpu', '').replace('gpu_', '')

                return f"""use cudarc::nvrtc::Ptx;
        let ptx_data = {ptx_expr};
        let ptx = Ptx::from_src(std::str::from_utf8(&ptx_data).unwrap());
        let {var_name} = {device_name}.load_ptx(ptx, "{module_name}", &[
            {func_list}
        ])"""

            content = module_pattern.sub(replace_load_module, content, count=1)
            fixes_applied.append("load_module → load_ptx")

    # 2. Fix load_function -> get_func
    if 'load_function' in content:
        content = content.replace('load_function', 'get_func')
        fixes_applied.append("load_function → get_func")

    # 3. Fix launch_builder patterns
    if 'launch_builder' in content:
        # Complex pattern to match entire launch_builder blocks
        launch_pattern = re.compile(
            r'let mut (\w+) = self\.(\w+)\.launch_builder\(&(\w+)\);(.*?)'
            r'unsafe \{\s*\1\.launch\((\w+)\)\?\;\s*\}',
            re.DOTALL
        )

        def replace_launch(match):
            var_name = match.group(1)
            device_field = match.group(2)
            func_name = match.group(3)
            args_block = match.group(4)
            config_name = match.group(5)

            # Extract arguments and let statements
            lines = args_block.strip().split('\n')
            args = []
            let_statements = []

            for line in lines:
                line = line.strip()
                if line.startswith('let '):
                    let_statements.append('        ' + line)
                elif '.arg(' in line:
                    # Extract the argument
                    arg_match = re.search(r'\.arg\(([^)]+)\)', line)
                    if arg_match:
                        args.append(arg_match.group(1))

            # Build replacement
            result = ""
            if let_statements:
                result = '\n'.join(let_statements) + '\n\n'

            result += f"""        unsafe {{
            {func_name}.launch(
                {config_name},
                ({', '.join(args)})
            )?;
        }}"""

            return result

        content = launch_pattern.sub(replace_launch, content)
        fixes_applied.append("launch_builder → launch()")

    # 4. Fix CudaSlice.len() -> .len
    if 'CudaSlice' in content and '.len()' in content:
        # This is tricky - need to identify CudaSlice variables
        # For now, do a conservative fix
        slice_pattern = re.compile(r'(\w+_gpu|gpu_\w+)\.len\(\)')
        content = slice_pattern.sub(r'\1.len', content)
        fixes_applied.append("CudaSlice.len() → .len")

    # 5. Fix remaining API changes
    replacements = [
        ('stream.memcpy_stod', 'device.htod_sync_copy'),
        ('stream.memcpy_dtos', 'device.dtoh_sync_copy'),
        ('stream.memcpy_dtod', 'device.dtod_copy'),
        ('stream.synchronize()', 'device.synchronize()?'),
        ('stream.alloc_zeros', 'device.alloc_zeros'),
        ('default_stream()', 'fork_default_stream()?'),
        ('context.', 'device.'),  # Fix any remaining context references
    ]

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            fixes_applied.append(f"{old} → {new}")

    # Save if changed
    if content != original:
        # Create backup
        backup_path = f"{filepath}.cuda-migration-backup"
        with open(backup_path, 'w') as f:
            f.write(original)

        # Write fixed content
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✅ Fixed: {', '.join(fixes_applied)}")
        print(f"  💾 Backup: {backup_path}")
        return True
    else:
        print(f"  ℹ️  No changes needed")
        return False

# Find all GPU files that might need fixing
def find_gpu_files():
    """Find all GPU-related Rust files."""
    gpu_files = []

    # Use find command
    result = subprocess.run(
        ['find', '.', '-type', 'f', '-name', '*.rs', '-path', '*gpu*'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if line and ('foundation' in line or 'src' in line):
                gpu_files.append(line)

    # Also find files with cuda in name
    result = subprocess.run(
        ['find', '.', '-type', 'f', '-name', '*.rs', '-path', '*cuda*'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if line and ('foundation' in line or 'src' in line):
                if line not in gpu_files:
                    gpu_files.append(line)

    return sorted(gpu_files)

def main():
    print("=" * 60)
    print("🔧 COMPLETE CUDA MIGRATION TO cudarc 0.9")
    print("=" * 60)

    # Find all GPU files
    gpu_files = find_gpu_files()
    print(f"\n📊 Found {len(gpu_files)} GPU-related files")

    # Process each file
    fixed_count = 0
    for filepath in gpu_files:
        if fix_file(filepath):
            fixed_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Migration Complete!")
    print(f"📊 Fixed {fixed_count} out of {len(gpu_files)} files")
    print("=" * 60)

    # Test compilation
    print("\n🧪 Testing compilation...")
    result = subprocess.run(
        ['cargo', 'check', '--features', 'cuda'],
        capture_output=True,
        text=True
    )

    if 'error' in result.stderr:
        error_count = result.stderr.count('error[')
        print(f"⚠️  Still {error_count} compilation errors")
        print("Run: cargo check --features cuda 2>&1 | grep 'error\\[' | head -20")
    else:
        print("✅ All CUDA code compiles successfully!")

    return fixed_count > 0

if __name__ == "__main__":
    main()