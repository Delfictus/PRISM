//! GPU-Accelerated Parallel Graph Coloring
//!
//! Runs thousands of coloring attempts in parallel on GPU
//! to massively explore the solution space.

use shared_types::*;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

/// GPU-accelerated parallel coloring search
pub struct GpuColoringSearch {
    context: Arc<CudaContext>,
    greedy_kernel: Arc<CudaFunction>,
    sa_kernel: Arc<CudaFunction>,
}

impl GpuColoringSearch {
    /// Create new GPU coloring search engine with shared context
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        // Load PTX module
        let ptx_path = "target/ptx/parallel_coloring.ptx";
        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Parallel coloring PTX not found at: {}", ptx_path));
        }

        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
        let module = context.load_module(ptx)?;

        let greedy_kernel = Arc::new(module.load_function("parallel_greedy_coloring_kernel")?);
        let sa_kernel = Arc::new(module.load_function("parallel_sa_kernel")?);

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

        let stream = self.context.default_stream();
        let n = graph.num_vertices;

        // Upload graph data
        let adjacency: Vec<bool> = graph.adjacency.clone();
        let adjacency_gpu: CudaSlice<bool> = stream.memcpy_stod(&adjacency)?;

        // Upload phase data
        let phases_gpu: CudaSlice<f64> = stream.memcpy_stod(&phase_field.phases)?;
        let coherence_gpu: CudaSlice<f64> = stream.memcpy_stod(&phase_field.coherence_matrix)?;

        // Create vertex ordering from Kuramoto phases
        let mut vertex_order: Vec<(usize, f64)> = kuramoto.phases.iter()
            .enumerate()
            .take(n)
            .map(|(i, &p)| (i, p))
            .collect();
        vertex_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let order: Vec<i32> = vertex_order.iter().map(|(i, _)| *i as i32).collect();
        let order_gpu: CudaSlice<i32> = stream.memcpy_stod(&order)?;

        // Allocate output buffers
        let mut colorings_gpu: CudaSlice<i32> = stream.alloc_zeros(n_attempts * n)?;
        let mut chromatic_gpu: CudaSlice<i32> = stream.alloc_zeros(n_attempts)?;
        let mut conflicts_gpu: CudaSlice<i32> = stream.alloc_zeros(n_attempts)?;

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

        let mut launch_greedy = stream.launch_builder(&self.greedy_kernel);
        launch_greedy.arg(&adjacency_gpu);
        launch_greedy.arg(&phases_gpu);
        launch_greedy.arg(&order_gpu);
        launch_greedy.arg(&coherence_gpu);
        launch_greedy.arg(&mut colorings_gpu);
        launch_greedy.arg(&mut chromatic_gpu);
        launch_greedy.arg(&mut conflicts_gpu);
        launch_greedy.arg(&n_i32);
        launch_greedy.arg(&n_attempts_i32);
        launch_greedy.arg(&max_colors_i32);
        launch_greedy.arg(&seed);
        unsafe { launch_greedy.launch(cfg)?; }

        // Download results
        let chromatic_numbers = stream.memcpy_dtov(&chromatic_gpu)?;
        let conflict_counts = stream.memcpy_dtov(&conflicts_gpu)?;

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
        let all_colorings = stream.memcpy_dtov(&colorings_gpu)?;
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

        let stream = self.context.default_stream();
        let n = graph.num_vertices;
        let n_chains = initial_colorings.len();

        // Upload graph
        let adjacency_gpu: CudaSlice<bool> = stream.memcpy_stod(&graph.adjacency)?;

        // Upload initial colorings
        let mut colorings: Vec<i32> = Vec::with_capacity(n_chains * n);
        for sol in initial_colorings {
            colorings.extend(sol.colors.iter().map(|&c| c as i32));
        }
        let mut colorings_gpu: CudaSlice<i32> = stream.memcpy_stod(&colorings)?;

        // Initial chromatic numbers
        let chromatic: Vec<i32> = initial_colorings.iter()
            .map(|s| s.chromatic_number as i32)
            .collect();
        let mut chromatic_gpu: CudaSlice<i32> = stream.memcpy_stod(&chromatic)?;

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

        let mut launch_sa = stream.launch_builder(&self.sa_kernel);
        launch_sa.arg(&adjacency_gpu);
        launch_sa.arg(&mut colorings_gpu);
        launch_sa.arg(&mut chromatic_gpu);
        launch_sa.arg(&n_i32);
        launch_sa.arg(&n_chains_i32);
        launch_sa.arg(&target_colors_i32);
        launch_sa.arg(&iterations_i32);
        launch_sa.arg(&initial_temperature);
        launch_sa.arg(&seed);
        unsafe { launch_sa.launch(cfg)?; }

        // Download results
        let final_colorings = stream.memcpy_dtov(&colorings_gpu)?;
        let final_chromatic = stream.memcpy_dtov(&chromatic_gpu)?;

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
