#!/bin/bash
#
# PTX Kernel Deep Analysis Tool
# Extracts: kernel signatures, ABI, memory layout, calling conventions, fusion indicators
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   PTX Kernel Deep Analysis - Custom Fused Rust Kernel Audit   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PTX_DIR="foundation/kernels/ptx"

if [ ! -d "$PTX_DIR" ]; then
    echo -e "${RED}ERROR: PTX directory not found: $PTX_DIR${NC}"
    exit 1
fi

# Create analysis output directory
ANALYSIS_DIR="kernel_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ANALYSIS_DIR"

echo -e "${BLUE}Analysis output directory: $ANALYSIS_DIR${NC}\n"

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ARCHITECTURAL METADATA
#═══════════════════════════════════════════════════════════════════════════════

echo -e "${GREEN}[1/10] Extracting Architectural Metadata...${NC}"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    outfile="$ANALYSIS_DIR/${basename%.ptx}_arch.txt"
    
    echo "=== ARCHITECTURAL METADATA: $basename ===" > "$outfile"
    echo "" >> "$outfile"
    
    # Target architecture
    echo "TARGET ARCHITECTURE:" >> "$outfile"
    grep -E "\.target|\.address_size|\.version" "$ptx" | head -10 >> "$outfile" 2>/dev/null || echo "  (none found)" >> "$outfile"
    echo "" >> "$outfile"
    
    # Minimum compute capability
    echo "COMPUTE CAPABILITY REQUIREMENTS:" >> "$outfile"
    grep -E "sm_[0-9]+|compute_[0-9]+" "$ptx" | head -5 >> "$outfile" 2>/dev/null || echo "  (none specified)" >> "$outfile"
    echo "" >> "$outfile"
    
    echo -e "  ${CYAN}✓${NC} $basename architecture metadata extracted"
done

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: KERNEL ENTRY POINTS AND SIGNATURES
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[2/10] Extracting Kernel Entry Points and Full Signatures...${NC}"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    outfile="$ANALYSIS_DIR/${basename%.ptx}_kernels.txt"
    
    echo "=== KERNEL SIGNATURES: $basename ===" > "$outfile"
    echo "" >> "$outfile"
    
    # Extract kernel entry points with full signatures
    grep -A 30 "\.visible \.entry" "$ptx" | grep -B 1 -A 30 "\.entry" >> "$outfile" 2>/dev/null || echo "(no kernels found)" >> "$outfile"
    
    # Count kernels
    kernel_count=$(grep -c "\.visible \.entry" "$ptx" 2>/dev/null || echo "0")
    echo "" >> "$outfile"
    echo "TOTAL KERNELS: $kernel_count" >> "$outfile"
    
    echo -e "  ${CYAN}✓${NC} $basename - Found $kernel_count kernel(s)"
done

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PARAMETER EXTRACTION (ABI)
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[3/10] Extracting Kernel Parameters (ABI/Calling Convention)...${NC}"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    outfile="$ANALYSIS_DIR/${basename%.ptx}_parameters.txt"
    
    echo "=== KERNEL PARAMETERS: $basename ===" > "$outfile"
    echo "" >> "$outfile"
    
    # Extract parameter declarations
    echo "PARAMETER LAYOUT:" >> "$outfile"
    grep -E "\.param.*\.u64|\.param.*\.u32|\.param.*\.f32|\.param.*\.f64|\.param.*\.b64|\.param.*\.b32" "$ptx" >> "$outfile" 2>/dev/null || echo "(no parameters found)" >> "$outfile"
    echo "" >> "$outfile"
    
    echo -e "  ${CYAN}✓${NC} $basename parameters extracted"
done

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MEMORY ACCESS PATTERNS (FUSION INDICATOR)
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[4/10] Analyzing Memory Access Patterns (Fusion Detection)...${NC}"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    outfile="$ANALYSIS_DIR/${basename%.ptx}_memory.txt"
    
    echo "=== MEMORY PATTERN ANALYSIS: $basename ===" > "$outfile"
    echo "" >> "$outfile"
    
    # Count global memory operations (high count = unfused)
    global_loads=$(grep -c "ld\.global" "$ptx" 2>/dev/null || echo "0")
    global_stores=$(grep -c "st\.global" "$ptx" 2>/dev/null || echo "0")
    
    # Count shared memory (high count = fused)
    shared_decls=$(grep -c "\.shared\." "$ptx" 2>/dev/null || echo "0")
    shared_loads=$(grep -c "ld\.shared" "$ptx" 2>/dev/null || echo "0")
    shared_stores=$(grep -c "st\.shared" "$ptx" 2>/dev/null || echo "0")
    
    # Count registers (high count = fused, keeping intermediates in registers)
    reg_count=$(grep -E "\.reg\.(f32|f64|u32|u64|b32|b64)" "$ptx" | wc -l)
    
    echo "GLOBAL MEMORY OPERATIONS:" >> "$outfile"
    echo "  ld.global: $global_loads" >> "$outfile"
    echo "  st.global: $global_stores" >> "$outfile"
    echo "" >> "$outfile"
    
    echo "SHARED MEMORY USAGE:" >> "$outfile"
    echo "  .shared declarations: $shared_decls" >> "$outfile"
    echo "  ld.shared: $shared_loads" >> "$outfile"
    echo "  st.shared: $shared_stores" >> "$outfile"
    echo "" >> "$outfile"
    
    echo "REGISTER FILE USAGE:" >> "$outfile"
    echo "  Register declarations: $reg_count" >> "$outfile"
    echo "" >> "$outfile"
    
    # Fusion heuristic
    echo "FUSION LIKELIHOOD SCORE:" >> "$outfile"
    fusion_score=0
    
    if [ "$shared_decls" -gt 5 ]; then
        echo "  ✓ High shared memory usage (+2)" >> "$outfile"
        fusion_score=$((fusion_score + 2))
    fi
    
    if [ "$reg_count" -gt 50 ]; then
        echo "  ✓ High register pressure (+2)" >> "$outfile"
        fusion_score=$((fusion_score + 2))
    fi
    
    if [ "$global_loads" -lt 20 ] && [ "$global_stores" -lt 10 ]; then
        echo "  ✓ Low global memory traffic (+1)" >> "$outfile"
        fusion_score=$((fusion_score + 1))
    fi
    
    if [ "$fusion_score" -ge 3 ]; then
        echo -e "  ${GREEN}VERDICT: Likely FUSED kernel (score: $fusion_score/5)${NC}" >> "$outfile"
        echo -e "  ${GREEN}✓${NC} $basename - ${GREEN}FUSED${NC} (score: $fusion_score/5)"
    else
        echo "  VERDICT: Likely UNFUSED kernel (score: $fusion_score/5)" >> "$outfile"
        echo -e "  ${YELLOW}⚠${NC} $basename - ${YELLOW}UNFUSED${NC} (score: $fusion_score/5)"
    fi
done

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXTRACT VISIBLE FUNCTION NAMES (KERNEL NAMES FOR load_ptx)
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[5/10] Extracting Callable Kernel Names...${NC}"

KERNEL_NAMES_FILE="$ANALYSIS_DIR/kernel_names_for_load_ptx.txt"
echo "=== KERNEL NAMES FOR device.load_ptx() ===" > "$KERNEL_NAMES_FILE"
echo "" >> "$KERNEL_NAMES_FILE"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    
    echo "// $basename" >> "$KERNEL_NAMES_FILE"
    echo "let module = device.load_ptx(ptx, \"${basename%.ptx}\", &[" >> "$KERNEL_NAMES_FILE"
    
    grep "\.visible \.entry" "$ptx" | sed 's/.*\.entry /    "/' | sed 's/(.*$/",/' >> "$KERNEL_NAMES_FILE"
    
    echo "])?;" >> "$KERNEL_NAMES_FILE"
    echo "" >> "$KERNEL_NAMES_FILE"
    
    echo -e "  ${CYAN}✓${NC} $basename kernel names extracted"
done

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RUST KERNEL SOURCE DETECTION
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[6/10] Searching for Rust Kernel Source Files...${NC}"

RUST_SOURCE_FILE="$ANALYSIS_DIR/rust_kernel_sources.txt"
echo "=== RUST KERNEL SOURCE FILES ===" > "$RUST_SOURCE_FILE"
echo "" >> "$RUST_SOURCE_FILE"

# Search for kernel attribute macros
echo "FILES WITH #[kernel] OR #[cuda] ATTRIBUTES:" >> "$RUST_SOURCE_FILE"
find foundation -name "*.rs" -type f -exec grep -l "#\[kernel\]" {} \; >> "$RUST_SOURCE_FILE" 2>/dev/null || echo "(none found)" >> "$RUST_SOURCE_FILE"
find foundation -name "*.rs" -type f -exec grep -l "#\[cuda(" {} \; >> "$RUST_SOURCE_FILE" 2>/dev/null
echo "" >> "$RUST_SOURCE_FILE"

# Search for cuda_std usage
echo "FILES USING cuda-std OR cuda_std:" >> "$RUST_SOURCE_FILE"
find foundation -name "*.rs" -type f -exec grep -l "use cuda_std" {} \; >> "$RUST_SOURCE_FILE" 2>/dev/null || echo "(none found)" >> "$RUST_SOURCE_FILE"
echo "" >> "$RUST_SOURCE_FILE"

# Search for nvptx target references
echo "CARGO.TOML FILES WITH NVPTX TARGET:" >> "$RUST_SOURCE_FILE"
find foundation -name "Cargo.toml" -type f -exec grep -l "nvptx" {} \; >> "$RUST_SOURCE_FILE" 2>/dev/null || echo "(none found)" >> "$RUST_SOURCE_FILE"
echo "" >> "$RUST_SOURCE_FILE"

source_count=$(find foundation -name "*.rs" -type f -exec grep -l "#\[kernel\]" {} \; 2>/dev/null | wc -l)
if [ "$source_count" -eq 0 ]; then
    echo -e "  ${RED}✗${NC} No Rust kernel source files found"
else
    echo -e "  ${GREEN}✓${NC} Found $source_count Rust kernel source file(s)"
fi

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CUDARC API USAGE ANALYSIS
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[7/10] Analyzing cudarc API Usage in Codebase...${NC}"

CUDARC_API_FILE="$ANALYSIS_DIR/cudarc_api_usage.txt"
echo "=== CUDARC API USAGE ===" > "$CUDARC_API_FILE"
echo "" >> "$CUDARC_API_FILE"

echo "CURRENT load_ptx() CALLS:" >> "$CUDARC_API_FILE"
find foundation -name "*.rs" -type f -exec grep -H "load_ptx" {} \; >> "$CUDARC_API_FILE" 2>/dev/null || echo "(none found)" >> "$CUDARC_API_FILE"
echo "" >> "$CUDARC_API_FILE"

echo "PTX FILE PATH REFERENCES:" >> "$CUDARC_API_FILE"
find foundation -name "*.rs" -type f -exec grep -H "\.ptx" {} \; >> "$CUDARC_API_FILE" 2>/dev/null || echo "(none found)" >> "$CUDARC_API_FILE"
echo "" >> "$CUDARC_API_FILE"

echo "KERNEL LAUNCH PATTERNS:" >> "$CUDARC_API_FILE"
find foundation -name "*.rs" -type f -exec grep -H "launch\|LaunchConfig" {} \; >> "$CUDARC_API_FILE" 2>/dev/null || echo "(none found)" >> "$CUDARC_API_FILE"
echo "" >> "$CUDARC_API_FILE"

echo -e "  ${CYAN}✓${NC} cudarc API usage analyzed"

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXTRACT COMPLETE RUST CALLING SIGNATURES
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[8/10] Generating Rust Calling Signatures from PTX...${NC}"

RUST_SIGS_FILE="$ANALYSIS_DIR/rust_kernel_signatures.rs"
echo "// Auto-generated Rust kernel signatures from PTX analysis" > "$RUST_SIGS_FILE"
echo "// These should match your actual kernel implementations" >> "$RUST_SIGS_FILE"
echo "" >> "$RUST_SIGS_FILE"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    
    echo "// ═══════════════════════════════════════════════════════════" >> "$RUST_SIGS_FILE"
    echo "// Source: $basename" >> "$RUST_SIGS_FILE"
    echo "// ═══════════════════════════════════════════════════════════" >> "$RUST_SIGS_FILE"
    echo "" >> "$RUST_SIGS_FILE"
    
    # Extract kernel names and attempt parameter reconstruction
    grep "\.visible \.entry" "$ptx" | while read -r line; do
        kernel_name=$(echo "$line" | sed 's/.*\.entry //' | sed 's/(.*//')
        
        echo "#[kernel]" >> "$RUST_SIGS_FILE"
        echo "pub unsafe fn $kernel_name(" >> "$RUST_SIGS_FILE"
        echo "    // TODO: Add parameters based on .param declarations in PTX" >> "$RUST_SIGS_FILE"
        echo "    // See: ${basename%.ptx}_parameters.txt" >> "$RUST_SIGS_FILE"
        echo ") {" >> "$RUST_SIGS_FILE"
        echo "    unimplemented!()" >> "$RUST_SIGS_FILE"
        echo "}" >> "$RUST_SIGS_FILE"
        echo "" >> "$RUST_SIGS_FILE"
    done
done

echo -e "  ${CYAN}✓${NC} Rust signatures generated"

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PERFORMANCE CHARACTERISTICS
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[9/10] Analyzing Performance Characteristics...${NC}"

PERF_FILE="$ANALYSIS_DIR/performance_analysis.txt"
echo "=== PERFORMANCE CHARACTERISTICS ===" > "$PERF_FILE"
echo "" >> "$PERF_FILE"

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    basename=$(basename "$ptx")
    
    echo "FILE: $basename" >> "$PERF_FILE"
    echo "─────────────────────────────────────────" >> "$PERF_FILE"
    
    # Instruction mix analysis
    adds=$(grep -c "add\." "$ptx" 2>/dev/null || echo "0")
    muls=$(grep -c "mul\." "$ptx" 2>/dev/null || echo "0")
    fmas=$(grep -c "fma\." "$ptx" 2>/dev/null || echo "0")
    branches=$(grep -c "bra\|@" "$ptx" 2>/dev/null || echo "0")
    
    echo "  Arithmetic Ops: add=$adds, mul=$muls, fma=$fmas" >> "$PERF_FILE"
    echo "  Control Flow: branches=$branches" >> "$PERF_FILE"
    
    # Memory coalescing hints
    echo "  Memory Access Patterns:" >> "$PERF_FILE"
    if grep -q "\.u64.*tid\.x" "$ptx" 2>/dev/null; then
        echo "    ✓ Thread-based indexing detected (likely coalesced)" >> "$PERF_FILE"
    fi
    
    # Double-double precision detection
    if echo "$basename" | grep -q "double_double"; then
        echo "  ⚠ High-precision arithmetic (quad-precision emulation)" >> "$PERF_FILE"
    fi
    
    echo "" >> "$PERF_FILE"
done

echo -e "  ${CYAN}✓${NC} Performance characteristics analyzed"

#═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: GENERATE INTEGRATION TEMPLATE
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${GREEN}[10/10] Generating Rust Integration Template...${NC}"

TEMPLATE_FILE="$ANALYSIS_DIR/kernel_integration_template.rs"

cat > "$TEMPLATE_FILE" << 'RUST_TEMPLATE'
//! Auto-generated kernel integration template
//! Based on PTX analysis - customize for your needs

use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use anyhow::{Result, Context, anyhow};

pub struct KernelModule {
    device: Arc<CudaDevice>,
    module: CudaModule,
}

impl KernelModule {
    pub fn load(device: Arc<CudaDevice>, ptx_path: &str) -> Result<Self> {
        // Validate compute capability
        let (major, minor) = device.compute_capability()
            .context("Failed to get compute capability")?;
        let compute_cap = major * 10 + minor;
        
        // Require at least sm_75 (Turing) for modern features
        if compute_cap < 75 {
            return Err(anyhow!(
                "Insufficient compute capability: {}.{} (need 7.5+)",
                major, minor
            ));
        }
        
        // Read PTX file
        let ptx_data = std::fs::read(ptx_path)
            .with_context(|| format!("Failed to read PTX: {}", ptx_path))?;
        
        // Convert to string (PTX is ASCII text)
        let ptx_string = String::from_utf8(ptx_data)
            .context("Invalid PTX encoding (not UTF-8)")?;
        
        // Create Ptx object
        let ptx = Ptx::from_src(&ptx_string);
        
        // Extract module name from path
        let module_name = std::path::Path::new(ptx_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid PTX path"))?;
        
        // Load with kernel names (CUSTOMIZE THIS LIST)
        let kernel_names = vec![
            // TODO: Replace with actual kernel names from *_kernels.txt
            "example_kernel_name",
        ];
        
        let module = device.load_ptx(
            ptx,
            module_name,
            &kernel_names
        ).context("Failed to load PTX module")?;
        
        // Verify kernels loaded successfully
        for kernel_name in &kernel_names {
            module.get_func(kernel_name)
                .ok_or_else(|| anyhow!("Kernel '{}' not found in module", kernel_name))?;
        }
        
        println!("[KERNEL] ✓ Loaded {} kernels from {}", kernel_names.len(), ptx_path);
        
        Ok(Self { device, module })
    }
    
    pub fn launch_kernel(
        &self,
        kernel_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[&dyn cudarc::driver::DeviceRepr],
    ) -> Result<()> {
        let func = self.module.get_func(kernel_name)
            .ok_or_else(|| anyhow!("Kernel '{}' not found", kernel_name))?;
        
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        
        unsafe {
            func.launch(cfg, params)
                .with_context(|| format!("Failed to launch kernel '{}'", kernel_name))?;
        }
        
        self.device.synchronize()
            .context("Kernel synchronization failed")?;
        
        Ok(())
    }
}

// Example usage:
// let module = KernelModule::load(device.clone(), "foundation/kernels/ptx/neuromorphic_gemv.ptx")?;
// module.launch_kernel("matvec_input_kernel", (grid_size, 1, 1), (block_size, 1, 1), &[params])?;
RUST_TEMPLATE

echo -e "  ${CYAN}✓${NC} Integration template generated"

#═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
#═══════════════════════════════════════════════════════════════════════════════

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    ANALYSIS COMPLETE                          ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Results saved to: $ANALYSIS_DIR/${NC}\n"

echo "Key files:"
echo "  📋 kernel_names_for_load_ptx.txt    - Exact names for device.load_ptx()"
echo "  🔧 rust_kernel_signatures.rs        - Rust kernel function templates"
echo "  📊 *_memory.txt                     - Fusion analysis per PTX file"
echo "  🎯 kernel_integration_template.rs   - Complete Rust integration code"
echo "  🔍 cudarc_api_usage.txt            - Current API usage in codebase"
echo "  📁 rust_kernel_sources.txt         - Location of Rust kernel sources"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review *_memory.txt files for fusion scores"
echo "  2. Check rust_kernel_sources.txt to locate original .rs files"
echo "  3. Use kernel_names_for_load_ptx.txt for correct load_ptx() calls"
echo "  4. Implement kernel_integration_template.rs in your codebase"
echo "  5. Verify parameters in *_parameters.txt match your Rust signatures"
echo ""

echo -e "${CYAN}To view results:${NC}"
echo "  cd $ANALYSIS_DIR && ls -lh"
echo ""
