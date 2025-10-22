use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=foundation/cuda/adaptive_coloring.cu");

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

    // Create output directory for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).unwrap();

    let cuda_src = PathBuf::from("foundation/cuda/adaptive_coloring.cu");
    let ptx_output = ptx_dir.join("adaptive_coloring.ptx");

    println!("cargo:warning=[BUILD]   Input:  {}", cuda_src.display());
    println!("cargo:warning=[BUILD]   Output: {}", ptx_output.display());

    // Compile to PTX for sm_90 (Hopper: H200)
    // NOTE: RTX 5070 Laptop has CC 12.0 (Blackwell), but CUDA 12.0 doesn't support sm_120 yet
    // Solution: Compile to sm_90 PTX, which is forward-compatible
    //   - H200 runs sm_90 PTX natively
    //   - RTX 5070 (CC 12.0) JIT-compiles sm_90 PTX to native code at runtime
    let status = Command::new(&nvcc)
        .args([
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
            println!("cargo:warning=[BUILD] ✅ CUDA kernels compiled successfully!");
            println!("cargo:warning=[BUILD]   PTX: {}", ptx_output.display());

            // Tell cargo where to find the PTX file
            println!("cargo:rustc-env=CUDA_PTX_PATH={}", ptx_output.display());
        }
        Ok(status) => {
            println!(
                "cargo:warning=[BUILD] ❌ nvcc compilation failed with status: {}",
                status
            );
            panic!("CUDA compilation failed");
        }
        Err(e) => {
            println!("cargo:warning=[BUILD] ❌ Failed to run nvcc: {}", e);
            panic!("Failed to run nvcc");
        }
    }

    // Link CUDA runtime libraries
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=curand");
}
