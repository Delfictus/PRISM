///! GPU-Accelerated Graph Coloring using CUDA
///!
///! Loads compiled PTX and executes adaptive coloring kernels on GPU.
///! GPU-ONLY: No CPU fallbacks - fails if CUDA unavailable.

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use ndarray::Array2;
use std::sync::Arc;
use anyhow::{Result, anyhow};

/// GPU coloring result
#[derive(Debug, Clone)]
pub struct GpuColoringResult {
    /// Best coloring found [N]
    pub coloring: Vec<usize>,

    /// Chromatic number (colors used)
    pub chromatic_number: usize,

    /// All attempts' chromatic numbers
    pub all_attempts: Vec<usize>,

    /// Runtime in milliseconds
    pub runtime_ms: f64,
}

/// GPU Graph Coloring Engine
pub struct GpuColoringEngine {
    context: Arc<CudaDevice>,
    sparse_kernel: Arc<CudaFunction>,
    dense_kernel: Arc<CudaFunction>,
}

impl GpuColoringEngine {
    /// Create new GPU coloring engine
    ///
    /// GPU-ONLY: Fails if CUDA unavailable
    pub fn new() -> Result<Self> {
        // Initialize CUDA context (GPU 0)
        let context = CudaDevice::new(0)
            .map_err(|e| anyhow!("Failed to initialize CUDA device 0: {:?}", e))?;

        println!("[GPU] Initialized CUDA device");

        // Load PTX module from build output
        let ptx_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/adaptive_coloring.ptx"));
        let ptx = Ptx::from_src(std::str::from_utf8(ptx_bytes)
            .map_err(|e| anyhow!("Invalid PTX UTF-8: {}", e))?);

        let kernel_names = vec![
            "_Z28sparse_parallel_coloring_csrPKiS0_PKfPiS3_Pfiiify",
            "_Z30dense_parallel_coloring_tensorPKfS0_PiS1_Pfiiify",
        ];
        context.load_ptx(ptx, "gpu_coloring", &kernel_names)
            .map_err(|e| anyhow!("Failed to load PTX module: {:?}", e))?;

        // Load kernel functions (using C++ mangled names from PTX)
        // Updated signatures with workspace parameter
        let sparse_kernel = Arc::new(context.get_func("gpu_coloring", "_Z28sparse_parallel_coloring_csrPKiS0_PKfPiS3_Pfiiify")
            .map_err(|e| anyhow!("Failed to load sparse kernel: {:?}", e))?);

        let dense_kernel = Arc::new(context.get_func("gpu_coloring", "_Z30dense_parallel_coloring_tensorPKfS0_PiS1_Pfiiify")
            .map_err(|e| anyhow!("Failed to load dense kernel: {:?}", e))?);

        println!("[GPU] ✅ Loaded adaptive_coloring.ptx");
        println!("[GPU]   - sparse_parallel_coloring_csr with dynamic memory");
        println!("[GPU]   - dense_parallel_coloring_tensor with dynamic memory");

        Ok(Self {
            context,
            sparse_kernel,
            dense_kernel,
        })
    }

    /// Color graph using GPU acceleration (without PRISM-AI)
    ///
    /// # Arguments
    /// - `adjacency`: Boolean adjacency matrix [N, N]
    /// - `num_attempts`: Parallel exploration attempts (higher = better quality, slower)
    /// - `temperature`: Exploration temperature (1.0 = balanced, higher = more random)
    /// - `max_colors`: Maximum colors to try (typically N)
    ///
    /// Uses uniform coherence (no PRISM-AI features)
    pub fn color_graph(
        &self,
        adjacency: &Array2<bool>,
        num_attempts: usize,
        temperature: f32,
        max_colors: usize,
    ) -> Result<GpuColoringResult> {
        self.color_graph_with_coherence(adjacency, None, num_attempts, temperature, max_colors)
    }

    /// Color graph using GPU acceleration with PRISM-AI coherence
    ///
    /// # Arguments
    /// - `adjacency`: Boolean adjacency matrix [N, N]
    /// - `coherence`: PRISM-AI enhanced coherence matrix [N*N] (optional)
    /// - `num_attempts`: Parallel exploration attempts (higher = better quality, slower)
    /// - `temperature`: Exploration temperature (1.0 = balanced, higher = more random)
    /// - `max_colors`: Maximum colors to try (typically N)
    ///
    /// # GPU Strategy
    /// - Sparse graphs (<40% density): CSR format kernel
    /// - Dense graphs (≥40% density): Tensor Core FP16 kernel
    pub fn color_graph_with_coherence(
        &self,
        adjacency: &Array2<bool>,
        coherence: Option<&[f32]>,
        num_attempts: usize,
        temperature: f32,
        max_colors: usize,
    ) -> Result<GpuColoringResult> {
        let n = adjacency.nrows();

        if n != adjacency.ncols() {
            return Err(anyhow!("Adjacency must be square"));
        }

        // Calculate density
        let num_edges: usize = adjacency.iter().filter(|&&x| x).count();
        let max_edges = n * (n - 1);
        let density = if max_edges > 0 {
            num_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        println!("[GPU] Coloring graph: {} vertices, {} edges ({:.1}% density)",
                 n, num_edges, density * 100.0);
        println!("[GPU]   Attempts: {}, Temperature: {:.2}, Max colors: {}",
                 num_attempts, temperature, max_colors);

        let start = std::time::Instant::now();

        // Use provided coherence or default uniform
        let coherence_vec = if let Some(coh) = coherence {
            if coh.len() != n * n {
                return Err(anyhow!("Coherence size mismatch: expected {}, got {}", n * n, coh.len()));
            }
            println!("[GPU] Using PRISM-AI enhanced coherence");
            coh.to_vec()
        } else {
            println!("[GPU] Using uniform coherence (PRISM-AI disabled)");
            vec![1.0f32; n * n]
        };

        // Configurable selection logic (hardcoded for now, will be wired to config)
        let threshold = 0.40;
        let prefer_sparse = false;
        let mask_width = 128;  // Using dual u64 masks

        let use_sparse = if prefer_sparse {
            true
        } else {
            density < threshold
        };

        // Enhanced logging as per WR requirements
        let mask_strategy = if mask_width == 64 { "single-u64" } else { "dual-u64" };
        println!("[GPU][COLORING] Selection: dense={} density={:.3} threshold={:.2} prefer_sparse={} mask={} width={}",
                 !use_sparse, density, threshold, prefer_sparse, mask_strategy, mask_width);

        // One-time startup confirmation
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!("[GPU][COLORING] Dynamic workspace enabled; no shared-mem vertex cap");
        });

        // Select strategy
        let result = if use_sparse {
            println!("[GPU] Using SPARSE kernel (CSR format)");
            self.color_sparse(adjacency, &coherence_vec, num_attempts, temperature, max_colors)?
        } else {
            println!("[GPU] Using DENSE kernel (FP16 Tensor Core)");
            self.color_dense(adjacency, &coherence_vec, num_attempts, temperature, max_colors)?
        };

        let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("[GPU] ✅ Best chromatic: {} colors ({:.2}ms)",
                 result.chromatic_number, runtime_ms);

        Ok(GpuColoringResult {
            runtime_ms,
            ..result
        })
    }

    /// Sparse graph coloring using CSR format
    fn color_sparse(
        &self,
        adjacency: &Array2<bool>,
        coherence: &[f32],
        num_attempts: usize,
        temperature: f32,
        max_colors: usize,
    ) -> Result<GpuColoringResult> {
        let n = adjacency.nrows();
        let device = &*self.context;

        // Convert to CSR format
        let (row_ptr, col_idx) = adjacency_to_csr(adjacency);

        // Upload to GPU
        let row_ptr_gpu: CudaSlice<i32> = device.htod_sync_copy(&row_ptr)?;
        let col_idx_gpu: CudaSlice<i32> = device.htod_sync_copy(&col_idx)?;
        let coherence_vec: Vec<f32> = coherence.to_vec();
        let coherence_gpu: CudaSlice<f32> = device.htod_sync_copy(&coherence_vec)?;

        // Allocate output buffers
        let mut colorings_gpu: CudaSlice<i32> = device.alloc_zeros(n * num_attempts)?;
        let mut chromatic_gpu: CudaSlice<i32> = device.alloc_zeros(num_attempts)?;

        // Allocate workspace for dynamic arrays (3 arrays per attempt: priorities, order, position)
        // Each array needs n elements, so 3*n per attempt
        let workspace_size = n * 3 * num_attempts;
        let mut workspace_gpu: CudaSlice<f32> = device.alloc_zeros(workspace_size)?;

        // Verify workspace allocation
        if workspace_size == 0 {
            return Err(anyhow!("Invalid workspace size: n={} attempts={}", n, num_attempts));
        }

        println!("[GPU][COLORING] Workspace allocated: {} floats ({:.2} MB)",
                 workspace_size, (workspace_size * 4) as f64 / (1024.0 * 1024.0));

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (num_attempts + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        println!("[GPU] Launching sparse kernel: {} blocks x {} threads = {} threads",
                 num_blocks, threads_per_block, num_blocks * threads_per_block);

        // Prepare kernel arguments
        let n_i32 = n as i32;
        let num_attempts_i32 = num_attempts as i32;
        let max_colors_i32 = max_colors as i32;
        let seed: u64 = 42; // Fixed seed for reproducibility

        unsafe {
            self.sparse_kernel.clone().launch(config, (
                &row_ptr_gpu,
                &col_idx_gpu,
                &coherence_gpu,
                &mut colorings_gpu,
                &mut chromatic_gpu,
                &mut workspace_gpu,  // Pass workspace for dynamic arrays
                &n_i32,
                &num_attempts_i32,
                &max_colors_i32,
                &temperature,
                &seed,
            ))?;
        }

        // Synchronize
        device.synchronize()?;

        // Copy results back
        let colorings_host: Vec<i32> = device.dtoh_sync_copy(&colorings_gpu)?;
        let chromatic_host: Vec<i32> = device.dtoh_sync_copy(&chromatic_gpu)?;

        // Find best attempt
        let (best_idx, &best_chromatic) = chromatic_host.iter()
            .enumerate()
            .min_by_key(|(_, &c)| c)
            .ok_or_else(|| anyhow!("No valid colorings found"))?;

        // Extract best coloring
        let best_coloring: Vec<usize> = colorings_host[best_idx * n..(best_idx + 1) * n]
            .iter()
            .map(|&c| c as usize)
            .collect();

        let all_attempts: Vec<usize> = chromatic_host.iter().map(|&c| c as usize).collect();

        Ok(GpuColoringResult {
            coloring: best_coloring,
            chromatic_number: best_chromatic as usize,
            all_attempts,
            runtime_ms: 0.0, // Set by caller
        })
    }

    /// Dense graph coloring using FP16 Tensor Cores
    fn color_dense(
        &self,
        adjacency: &Array2<bool>,
        coherence: &[f32],
        num_attempts: usize,
        temperature: f32,
        max_colors: usize,
    ) -> Result<GpuColoringResult> {
        let n = adjacency.nrows();
        let device = &*self.context;

        // Convert to FP16 adjacency matrix (cudarc has built-in half support via f16 feature)
        let adjacency_f32: Vec<f32> = adjacency.iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        // Upload to GPU
        let adjacency_gpu: CudaSlice<f32> = device.htod_sync_copy(&adjacency_f32)?;
        let coherence_vec: Vec<f32> = coherence.to_vec();
        let coherence_gpu: CudaSlice<f32> = device.htod_sync_copy(&coherence_vec)?;

        // Allocate output buffers
        let mut colorings_gpu: CudaSlice<i32> = device.alloc_zeros(n * num_attempts)?;
        let mut chromatic_gpu: CudaSlice<i32> = device.alloc_zeros(num_attempts)?;

        // Allocate workspace for dynamic arrays (3 arrays per attempt: priorities, order, position)
        // Each array needs n elements, so 3*n per attempt
        let workspace_size = n * 3 * num_attempts;
        let mut workspace_gpu: CudaSlice<f32> = device.alloc_zeros(workspace_size)?;

        // Verify workspace allocation
        if workspace_size == 0 {
            return Err(anyhow!("Invalid workspace size: n={} attempts={}", n, num_attempts));
        }

        println!("[GPU][COLORING] Workspace allocated: {} floats ({:.2} MB)",
                 workspace_size, (workspace_size * 4) as f64 / (1024.0 * 1024.0));

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (num_attempts + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        println!("[GPU] Launching dense kernel: {} blocks x {} threads",
                 num_blocks, threads_per_block);

        // Prepare kernel arguments
        let n_i32 = n as i32;
        let num_attempts_i32 = num_attempts as i32;
        let max_colors_i32 = max_colors as i32;
        let seed: u64 = 42;

        unsafe {
            self.dense_kernel.clone().launch(config, (
                &adjacency_gpu,
                &coherence_gpu,
                &mut colorings_gpu,
                &mut chromatic_gpu,
                &mut workspace_gpu,  // Pass workspace for dynamic arrays
                &n_i32,
                &num_attempts_i32,
                &max_colors_i32,
                &temperature,
                &seed,
            ))?;
        }

        // Synchronize
        device.synchronize()?;

        // Copy results back
        let colorings_host: Vec<i32> = device.dtoh_sync_copy(&colorings_gpu)?;
        let chromatic_host: Vec<i32> = device.dtoh_sync_copy(&chromatic_gpu)?;

        // Find best
        let (best_idx, &best_chromatic) = chromatic_host.iter()
            .enumerate()
            .min_by_key(|(_, &c)| c)
            .ok_or_else(|| anyhow!("No valid colorings"))?;

        let best_coloring: Vec<usize> = colorings_host[best_idx * n..(best_idx + 1) * n]
            .iter()
            .map(|&c| c as usize)
            .collect();

        let all_attempts: Vec<usize> = chromatic_host.iter().map(|&c| c as usize).collect();

        Ok(GpuColoringResult {
            coloring: best_coloring,
            chromatic_number: best_chromatic as usize,
            all_attempts,
            runtime_ms: 0.0,
        })
    }
}

/// Convert boolean adjacency matrix to CSR format
///
/// Returns (row_ptr, col_idx) where:
/// - row_ptr[i] = index of first edge for vertex i
/// - col_idx[j] = target vertex for edge j
fn adjacency_to_csr(adjacency: &Array2<bool>) -> (Vec<i32>, Vec<i32>) {
    let n = adjacency.nrows();
    let mut row_ptr = vec![0i32];
    let mut col_idx = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if adjacency[[i, j]] {
                col_idx.push(j as i32);
            }
        }
        row_ptr.push(col_idx.len() as i32);
    }

    (row_ptr, col_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run when GPU available
    fn test_gpu_coloring() {
        // Simple triangle graph
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;
        adj[[1, 2]] = true;
        adj[[2, 1]] = true;
        adj[[0, 2]] = true;
        adj[[2, 0]] = true;

        let engine = GpuColoringEngine::new().expect("GPU unavailable");
        let result = engine.color_graph(&adj, 10, 1.0, 10).expect("Coloring failed");

        // Triangle requires 3 colors
        assert_eq!(result.chromatic_number, 3);
    }

    #[test]
    fn test_csr_conversion() {
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true;
        adj[[1, 2]] = true;

        let (row_ptr, col_idx) = adjacency_to_csr(&adj);

        assert_eq!(row_ptr, vec![0, 1, 2, 2]); // 3 vertices + 1
        assert_eq!(col_idx, vec![1, 2]); // 2 edges
    }
}
