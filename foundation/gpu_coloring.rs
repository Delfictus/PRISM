//! GPU-Accelerated Parallel Graph Coloring
//!
//! Runs thousands of coloring attempts in parallel on GPU
//! to massively explore the solution space.

use shared_types::*;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};

/// GPU-accelerated parallel coloring search
pub struct GpuColoringSearch {
    context: Arc<CudaDevice>,
    greedy_kernel: Arc<CudaFunction>,
    sa_kernel: Arc<CudaFunction>,
}

impl GpuColoringSearch {
    /// Create new GPU coloring search engine with shared context
    pub fn new(context: Arc<CudaDevice>) -> Result<Self> {
        // Load PTX module
        let ptx_path = "target/ptx/parallel_coloring.ptx";
        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Parallel coloring PTX not found at: {}", ptx_path));
        }

        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
        let kernel_names = vec!["parallel_greedy_coloring_kernel", "parallel_sa_kernel"];
        context.load_ptx(ptx, "parallel_coloring", &kernel_names)?;

        let greedy_kernel = Arc::new(context.get_func("parallel_coloring", "parallel_greedy_coloring_kernel")?);
        let sa_kernel = Arc::new(context.get_func("parallel_coloring", "parallel_sa_kernel")?);

        Ok(Self {
            context,
            greedy_kernel,
            sa_kernel,
        })
    }

    /// Run massive parallel search for best coloring
    pub fn massive_parallel_search(
        &self,
        graph: &Graph,
        phase_field: &PhaseField,
        kuramoto: &KuramotoState,
        target_colors: usize,
        n_attempts: usize,
    ) -> Result<ColoringSolution> {
        println!("  🚀 GPU parallel search: {} attempts on GPU...", n_attempts);
        let start = std::time::Instant::now();

        let device = &*self.context;
        let n = graph.num_vertices;

        // Upload graph data
        let adjacency: Vec<bool> = graph.adjacency.clone();
        let adjacency_gpu: CudaSlice<bool> = device.htod_sync_copy(&adjacency)?;

        // Upload phase data
        let phases_gpu: CudaSlice<f64> = device.htod_sync_copy(&phase_field.phases)?;
        let coherence_gpu: CudaSlice<f64> = device.htod_sync_copy(&phase_field.coherence_matrix)?;

        // Create vertex ordering from Kuramoto phases
        let mut vertex_order: Vec<(usize, f64)> = kuramoto.phases.iter()
            .enumerate()
            .take(n)
            .map(|(i, &p)| (i, p))
            .collect();
        vertex_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let order: Vec<i32> = vertex_order.iter().map(|(i, _)| *i as i32).collect();
        let order_gpu: CudaSlice<i32> = device.htod_sync_copy(&order)?;

        // Allocate output buffers
        let mut colorings_gpu: CudaSlice<i32> = device.alloc_zeros(n_attempts * n)?;
        let mut chromatic_gpu: CudaSlice<i32> = device.alloc_zeros(n_attempts)?;
        let mut conflicts_gpu: CudaSlice<i32> = device.alloc_zeros(n_attempts)?;

        // Launch kernel
        let threads = 256;
        let blocks = (n_attempts + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let n_attempts_i32 = n_attempts as i32;
        let max_colors_i32 = target_colors as i32;
        let seed = 12345u64;

        unsafe {
            self.greedy_kernel.clone().launch(
                cfg,
                (
                    &adjacency_gpu,
                    &phases_gpu,
                    &order_gpu,
                    &coherence_gpu,
                    &mut colorings_gpu,
                    &mut chromatic_gpu,
                    &mut conflicts_gpu,
                    &n_i32,
                    &n_attempts_i32,
                    &max_colors_i32,
                    &seed,
                )
            )?;
        }

        // Download results
        let chromatic_numbers = device.dtoh_sync_copy(&chromatic_gpu)?;
        let conflict_counts = device.dtoh_sync_copy(&conflicts_gpu)?;

        // Find best valid solution
        let mut best_idx = 0;
        let mut best_chromatic = chromatic_numbers[0];

        for i in 0..n_attempts {
            if conflict_counts[i] == 0 && chromatic_numbers[i] < best_chromatic {
                best_idx = i;
                best_chromatic = chromatic_numbers[i];
            }
        }

        // Download best coloring
        let all_colorings = device.dtoh_sync_copy(&colorings_gpu)?;
        let best_coloring: Vec<usize> = all_colorings[best_idx * n..(best_idx + 1) * n]
            .iter()
            .map(|&c| c as usize)
            .collect();

        let elapsed = start.elapsed();
        let valid_count = conflict_counts.iter().filter(|&&c| c == 0).count();

        println!("  ✅ GPU search complete: {} colors (best of {} valid in {:?})",
                 best_chromatic, valid_count, elapsed);

        Ok(ColoringSolution {
            colors: best_coloring,
            chromatic_number: best_chromatic as usize,
            conflicts: conflict_counts[best_idx] as usize,
            quality_score: 1.0,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    /// Run parallel SA chains on GPU
    pub fn parallel_sa_search(
        &self,
        graph: &Graph,
        initial_colorings: &[ColoringSolution],
        iterations_per_chain: usize,
        initial_temperature: f64,
    ) -> Result<ColoringSolution> {
        println!("  🔥 GPU parallel SA: {} chains...", initial_colorings.len());

        let device = &*self.context;
        let n = graph.num_vertices;
        let n_chains = initial_colorings.len();

        // Upload graph
        let adjacency_gpu: CudaSlice<bool> = device.htod_sync_copy(&graph.adjacency)?;

        // Upload initial colorings
        let mut colorings: Vec<i32> = Vec::with_capacity(n_chains * n);
        for sol in initial_colorings {
            colorings.extend(sol.colors.iter().map(|&c| c as i32));
        }
        let mut colorings_gpu: CudaSlice<i32> = device.htod_sync_copy(&colorings)?;

        // Initial chromatic numbers
        let chromatic: Vec<i32> = initial_colorings.iter()
            .map(|s| s.chromatic_number as i32)
            .collect();
        let mut chromatic_gpu: CudaSlice<i32> = device.htod_sync_copy(&chromatic)?;

        // Launch SA kernel
        let threads = 256;
        let blocks = (n_chains + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let target_colors = 100;  // Fixed target for SA
        let n_i32 = n as i32;
        let n_chains_i32 = n_chains as i32;
        let target_colors_i32 = target_colors as i32;
        let iterations_i32 = iterations_per_chain as i32;
        let seed = 42u64;

        unsafe {
            self.sa_kernel.clone().launch(
                cfg,
                (
                    &adjacency_gpu,
                    &mut colorings_gpu,
                    &mut chromatic_gpu,
                    &n_i32,
                    &n_chains_i32,
                    &target_colors_i32,
                    &iterations_i32,
                    &initial_temperature,
                    &seed,
                )
            )?;
        }

        // Download results
        let final_colorings = device.dtoh_sync_copy(&colorings_gpu)?;
        let final_chromatic = device.dtoh_sync_copy(&chromatic_gpu)?;

        // Find best
        let best_idx = final_chromatic.iter()
            .enumerate()
            .min_by_key(|(_, &c)| c)
            .unwrap().0;

        let best_coloring: Vec<usize> = final_colorings[best_idx * n..(best_idx + 1) * n]
            .iter()
            .map(|&c| c as usize)
            .collect();

        println!("  ✅ GPU SA complete: {} colors", final_chromatic[best_idx]);

        Ok(ColoringSolution {
            colors: best_coloring,
            chromatic_number: final_chromatic[best_idx] as usize,
            conflicts: 0,  // TODO: verify
            quality_score: 1.0,
            computation_time_ms: 0.0,
        })
    }
}
