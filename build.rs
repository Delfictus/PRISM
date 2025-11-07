use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=foundation/cuda/adaptive_coloring.cu");
    println!("cargo:rerun-if-changed=foundation/cuda/prct_kernels.cu");
    println!("cargo:rerun-if-changed=foundation/kernels/neuromorphic_gemv.cu");
    println!("cargo:rerun-if-changed=foundation/kernels/transfer_entropy.cu");
    println!("cargo:rerun-if-changed=foundation/kernels/thermodynamic.cu");
    println!("cargo:rerun-if-changed=foundation/kernels/quantum_evolution.cu");
    println!("cargo:rerun-if-changed=foundation/kernels/active_inference.cu");

    // Only compile CUDA if cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    }
}

fn compile_cuda_kernels() {
    println!("cargo:warning=[BUILD] Compiling CUDA kernels for sm_90...");

    // Find nvcc compiler
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    // Check if nvcc exists
    let nvcc_check = Command::new(&nvcc).arg("--version").output();

    if nvcc_check.is_err() {
        println!("cargo:warning=[BUILD] nvcc not found - skipping CUDA compilation");
        println!("cargo:warning=[BUILD] Install CUDA Toolkit or set NVCC environment variable");
        return;
    }

    // Create output directory for PTX in OUT_DIR
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).unwrap();

    // Also create target/ptx for runtime loading (gpu_reservoir.rs looks here)
    let target_ptx_dir = PathBuf::from("target/ptx");
    std::fs::create_dir_all(&target_ptx_dir).unwrap();

    // Compile adaptive_coloring.cu
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/cuda/adaptive_coloring.cu",
        "adaptive_coloring.ptx",
    );

    // Compile prct_kernels.cu
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/cuda/prct_kernels.cu",
        "prct_kernels.ptx",
    );

    // Compile neuromorphic_gemv.cu (critical for reservoir processing)
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/kernels/neuromorphic_gemv.cu",
        "neuromorphic_gemv.ptx",
    );

    // Compile transfer_entropy.cu (Phase 1: TE ordering)
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/kernels/transfer_entropy.cu",
        "transfer_entropy.ptx",
    );

    // Compile thermodynamic.cu (Phase 2: Thermodynamic equilibration)
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/kernels/thermodynamic.cu",
        "thermodynamic.ptx",
    );

    // Compile quantum_evolution.cu (Phase 3: Quantum-classical hybrid)
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/kernels/quantum_evolution.cu",
        "quantum_evolution.ptx",
    );

    // Compile active_inference.cu (Phase 1: Active inference policy)
    compile_cu_file(
        &nvcc,
        &ptx_dir,
        "foundation/kernels/active_inference.cu",
        "active_inference.ptx",
    );

    // Link CUDA runtime libraries
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cublas");
}

fn compile_cu_file(nvcc: &str, ptx_dir: &PathBuf, src_path: &str, output_name: &str) {
    let cuda_src = PathBuf::from(src_path);
    let ptx_output = ptx_dir.join(output_name);

    println!("cargo:warning=[BUILD]   Input:  {}", cuda_src.display());
    println!("cargo:warning=[BUILD]   Output: {}", ptx_output.display());

    // Compile to PTX for sm_90 (Hopper: H200)
    // NOTE: RTX 5070 Laptop has CC 12.0 (Blackwell), but CUDA 12.0 doesn't support sm_120 yet
    // Solution: Compile to sm_90 PTX, which is forward-compatible
    //   - H200 runs sm_90 PTX natively
    //   - RTX 5070 (CC 12.0) JIT-compiles sm_90 PTX to native code at runtime
    let status = Command::new(&nvcc)
        .args(&[
            "--ptx",                    // Compile to PTX (portable, forward-compatible)
            "-O3",                      // Optimize
            "--gpu-architecture=sm_90", // Hopper architecture (H200 + forward-compat for newer GPUs)
            "--use_fast_math",          // Fast math operations
            "--extended-lambda",        // Enable device lambdas
            "-Xptxas",
            "-v", // Verbose PTX assembly
            "--default-stream",
            "per-thread", // Thread-safe streams
            "-I",
            "/usr/local/cuda/include", // CUDA headers
            cuda_src.to_str().unwrap(),
            "-o",
            ptx_output.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:warning=[BUILD] ✅ Compiled: {}", output_name);

            // Copy to target/ptx for runtime loading
            let target_ptx = PathBuf::from("target/ptx").join(output_name);
            if let Err(e) = std::fs::copy(&ptx_output, &target_ptx) {
                println!("cargo:warning=[BUILD] ⚠️  Failed to copy {} to target/ptx: {}", output_name, e);
            } else {
                println!("cargo:warning=[BUILD] ✅ Copied to: {}", target_ptx.display());
            }
        }
        Ok(status) => {
            println!(
                "cargo:warning=[BUILD] ❌ nvcc compilation failed for {}: status {}",
                src_path, status
            );
            panic!("CUDA compilation failed");
        }
        Err(e) => {
            println!("cargo:warning=[BUILD] ❌ Failed to run nvcc for {}: {}", src_path, e);
            panic!("Failed to run nvcc");
        }
    }
}
