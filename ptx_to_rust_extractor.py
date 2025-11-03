#!/usr/bin/env python3
"""
PTX Parameter Extractor - Reverse Engineer Rust Kernel Signatures
Analyzes PTX .param declarations and generates exact Rust function signatures
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class KernelParameter:
    """Represents a single kernel parameter"""
    name: str
    ptx_type: str
    rust_type: str
    is_pointer: bool
    is_const: bool
    address_space: Optional[str] = None

class PTXParameterExtractor:
    """Extract and convert PTX parameters to Rust signatures"""
    
    # PTX to Rust type mapping
    TYPE_MAP = {
        'u64': 'u64',
        'u32': 'u32', 
        'u16': 'u16',
        'u8': 'u8',
        'b64': 'u64',
        'b32': 'u32',
        'b16': 'u16',
        'b8': 'u8',
        's64': 'i64',
        's32': 'i32',
        's16': 'i16',
        's8': 'i8',
        'f64': 'f64',
        'f32': 'f32',
        'f16': 'f16',
    }
    
    def __init__(self, ptx_path: Path):
        self.ptx_path = ptx_path
        self.content = ptx_path.read_text()
    
    def extract_kernels(self) -> List[tuple]:
        """Extract all kernel entry points with their full signatures"""
        kernel_pattern = r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)'
        
        kernels = []
        for match in re.finditer(kernel_pattern, self.content, re.DOTALL):
            kernel_name = match.group(1)
            params_block = match.group(2)
            
            # Extract full parameter block (may span multiple lines after the signature)
            start_pos = match.end()
            param_section = self._extract_param_section(start_pos)
            
            params = self._parse_parameters(param_section)
            kernels.append((kernel_name, params))
        
        return kernels
    
    def _extract_param_section(self, start_pos: int) -> str:
        """Extract the parameter declaration section after kernel signature"""
        # Look for .param declarations in the next ~200 lines
        lines = self.content[start_pos:start_pos+5000].split('\n')
        param_lines = []
        
        for line in lines:
            if '.param' in line:
                param_lines.append(line)
            elif '{' in line:  # Start of kernel body
                break
        
        return '\n'.join(param_lines)
    
    def _parse_parameters(self, param_section: str) -> List[KernelParameter]:
        """Parse PTX parameter declarations"""
        params = []
        
        # Pattern: .param .u64 .ptr .align 8 param_name
        param_pattern = r'\.param\s+\.(u64|u32|u16|u8|b64|b32|b16|b8|s64|s32|s16|s8|f64|f32|f16)\s+(\.ptr\s+)?(?:\.align\s+\d+\s+)?(\w+)'
        
        for match in re.finditer(param_pattern, param_section):
            ptx_type = match.group(1)
            is_ptr = match.group(2) is not None
            param_name = match.group(3)
            
            # Infer Rust type
            base_type = self.TYPE_MAP.get(ptx_type, 'unknown')
            
            if is_ptr:
                # Pointers - check if const by heuristics
                is_const = 'input' in param_name.lower() or 'src' in param_name.lower() or param_name.startswith('in_')
                
                # Infer element type from parameter name
                if 'matrix' in param_name.lower() or 'weight' in param_name.lower():
                    rust_type = f"*{'const' if is_const else 'mut'} f32"
                elif 'index' in param_name.lower() or 'offset' in param_name.lower():
                    rust_type = f"*{'const' if is_const else 'mut'} i32"
                elif 'spike' in param_name.lower() or 'mask' in param_name.lower():
                    rust_type = f"*{'const' if is_const else 'mut'} u8"
                else:
                    rust_type = f"*{'const' if is_const else 'mut'} {base_type}"
            else:
                rust_type = base_type
                is_const = False
            
            params.append(KernelParameter(
                name=self._sanitize_param_name(param_name),
                ptx_type=ptx_type,
                rust_type=rust_type,
                is_pointer=is_ptr,
                is_const=is_const
            ))
        
        return params
    
    def _sanitize_param_name(self, name: str) -> str:
        """Convert PTX parameter names to Rust-friendly names"""
        # Remove common PTX prefixes
        name = re.sub(r'^param_', '', name)
        name = re.sub(r'^\w+_param_', '', name)
        
        # Convert to snake_case if needed
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
        name = name.lower()
        
        return name
    
    def generate_rust_signature(self, kernel_name: str, params: List[KernelParameter]) -> str:
        """Generate Rust kernel function signature"""
        lines = []
        lines.append(f"#[kernel]")
        lines.append(f"pub unsafe fn {kernel_name}(")
        
        if not params:
            lines.append(") {")
        else:
            for i, param in enumerate(params):
                comma = "," if i < len(params) - 1 else ""
                lines.append(f"    {param.name}: {param.rust_type}{comma}")
            lines.append(") {")
        
        lines.append("    // Kernel implementation")
        lines.append("    let tid = thread::index_1d();")
        lines.append("    // TODO: Add kernel logic")
        lines.append("}")
        lines.append("")
        
        return '\n'.join(lines)

def analyze_ptx_directory(ptx_dir: Path, output_dir: Path):
    """Analyze all PTX files in directory and generate Rust signatures"""
    output_dir.mkdir(exist_ok=True)
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║   PTX → Rust Kernel Signature Generator                       ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    for ptx_file in sorted(ptx_dir.glob("*.ptx")):
        print(f"📄 Analyzing: {ptx_file.name}")
        
        extractor = PTXParameterExtractor(ptx_file)
        kernels = extractor.extract_kernels()
        
        if not kernels:
            print(f"   ⚠️  No kernels found\n")
            continue
        
        # Generate Rust source file
        rust_filename = f"{ptx_file.stem}.rs"
        rust_file = output_dir / rust_filename
        
        with rust_file.open('w') as f:
            f.write(f"//! GPU Kernels: {ptx_file.stem}\n")
            f.write(f"//! Auto-generated from: {ptx_file.name}\n")
            f.write(f"//!\n")
            f.write(f"//! IMPORTANT: This is a template. Verify types and add implementation.\n\n")
            f.write(f"#![no_std]\n")
            f.write(f"#![feature(abi_ptx)]\n\n")
            f.write(f"use cuda_std::*;\n\n")
            
            for kernel_name, params in kernels:
                signature = extractor.generate_rust_signature(kernel_name, params)
                f.write(signature)
                f.write("\n")
                print(f"   ✓ {kernel_name} ({len(params)} params)")
        
        print(f"   💾 Generated: {rust_filename}\n")
    
    # Generate integration guide
    guide_file = output_dir / "INTEGRATION_GUIDE.md"
    with guide_file.open('w') as f:
        f.write("# Kernel Integration Guide\n\n")
        f.write("## Generated Rust Kernel Templates\n\n")
        f.write("This directory contains auto-generated Rust kernel templates based on PTX analysis.\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. **Review Parameter Types**: Verify pointer types (const vs mut, element types)\n")
        f.write("2. **Add Kernel Logic**: Implement the actual computation\n")
        f.write("3. **Add Cargo.toml**: Configure nvptx64-nvidia-cuda target\n")
        f.write("4. **Build PTX**: `cargo build --release --target nvptx64-nvidia-cuda`\n")
        f.write("5. **Test Kernels**: Create unit tests with cudarc\n\n")
        f.write("## Example Cargo.toml\n\n")
        f.write("```toml\n")
        f.write("[package]\n")
        f.write("name = \"foundation-cuda-kernels\"\n")
        f.write("version = \"0.1.0\"\n")
        f.write("edition = \"2021\"\n\n")
        f.write("[lib]\n")
        f.write("crate-type = [\"cdylib\"]\n\n")
        f.write("[dependencies]\n")
        f.write("cuda-std = \"0.3\"\n\n")
        f.write("[profile.release]\n")
        f.write("lto = true\n")
        f.write("opt-level = 3\n")
        f.write("codegen-units = 1\n")
        f.write("```\n\n")
        f.write("## Build Command\n\n")
        f.write("```bash\n")
        f.write("rustup target add nvptx64-nvidia-cuda\n")
        f.write("cargo build --release --target nvptx64-nvidia-cuda\n")
        f.write("```\n")
    
    print(f"\n✅ Analysis complete. Check {output_dir}/ for results.\n")

if __name__ == "__main__":
    ptx_dir = Path("foundation/kernels/ptx")
    output_dir = Path("kernel_analysis_rust_templates")
    
    if not ptx_dir.exists():
        print(f"❌ PTX directory not found: {ptx_dir}")
        sys.exit(1)
    
    analyze_ptx_directory(ptx_dir, output_dir)
