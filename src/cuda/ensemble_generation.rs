use anyhow::{anyhow, Result};
///! GPU Thermodynamic Ensemble Generation for Graph Coloring
///!
///! Generates multiple diverse initial states (vertex orderings) using
///! temperature-based sampling for parallel exploration.
///!
///! GPU-ONLY: All generation on CUDA
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Ensemble of thermodynamic replicas
#[derive(Debug, Clone)]
pub struct Ensemble {
    /// Vertex orderings for each replica [ensemble_size, N]
    pub orderings: Vec<Vec<usize>>,

    /// Temperature used for each replica
    pub temperatures: Vec<f32>,

    /// Random seeds for each replica
    pub seeds: Vec<u64>,

    /// Energy (degree variance) for each ordering
    pub energies: Vec<f32>,
}

/// GPU Ensemble Generator
pub struct GpuEnsembleGenerator {
    device: Arc<CudaDevice>,
    generate_kernel: Arc<cudarc::driver::CudaFunction>,
}

impl GpuEnsembleGenerator {
    /// Create new GPU ensemble generator
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)
            .map_err(|e| anyhow!("Failed to initialize CUDA for ensemble generation: {:?}", e))?;

        // Load ensemble generation kernel from PTX
        let ptx_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/adaptive_coloring.ptx"));
        let ptx = Ptx::from_src(
            std::str::from_utf8(ptx_bytes).map_err(|e| anyhow!("Invalid PTX UTF-8: {}", e))?,
        );

        device
            .load_ptx(
                ptx,
                "adaptive_coloring",
                &["generate_thermodynamic_ordering"],
            )
            .map_err(|e| anyhow!("Failed to load PTX module: {:?}", e))?;

        let generate_kernel = Arc::new(
            device
                .get_func("adaptive_coloring", "generate_thermodynamic_ordering")
                .ok_or_else(|| anyhow!("Failed to load generate_thermodynamic_ordering kernel"))?,
        );

        println!("[ENSEMBLE] GPU ensemble generator initialized");

        Ok(Self {
            device,
            generate_kernel,
        })
    }

    /// Generate thermodynamic ensemble on GPU
    ///
    /// # Arguments
    /// - `degrees`: Vertex degrees [N]
    /// - `ensemble_size`: Number of replicas to generate
    /// - `base_temperature`: Base temperature for thermodynamic sampling
    ///
    /// # Returns
    /// Ensemble with diverse vertex orderings
    pub fn generate(
        &self,
        degrees: &[usize],
        ensemble_size: usize,
        base_temperature: f32,
    ) -> Result<Ensemble> {
        let n = degrees.len();

        println!(
            "[ENSEMBLE] Generating {} replicas for {} vertices",
            ensemble_size, n
        );
        println!("[ENSEMBLE]   Base temperature: {:.2}", base_temperature);

        // Upload degrees to GPU
        let degrees_f32: Vec<f32> = degrees.iter().map(|&d| d as f32).collect();
        let degrees_gpu: CudaSlice<f32> = self.device.htod_sync_copy(&degrees_f32)?;

        // Allocate output buffers on GPU
        let mut orderings_gpu: CudaSlice<i32> = self.device.alloc_zeros(ensemble_size * n)?;
        let mut energies_gpu: CudaSlice<f32> = self.device.alloc_zeros(ensemble_size)?;

        // Generate temperatures (vary around base)
        let temperatures: Vec<f32> = (0..ensemble_size)
            .map(|i| {
                let scale = 0.5 + (i as f32 / ensemble_size as f32) * 1.5;
                base_temperature * scale
            })
            .collect();

        let temps_gpu: CudaSlice<f32> = self.device.htod_sync_copy(&temperatures)?;

        // Generate random seeds
        let seeds: Vec<u64> = (0..ensemble_size)
            .map(|i| 42u64 + i as u64 * 12345)
            .collect();

        let seeds_gpu: CudaSlice<u64> = self.device.htod_sync_copy(&seeds)?;

        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = (ensemble_size + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        println!(
            "[ENSEMBLE] Launching GPU kernel: {} blocks x {} threads",
            num_blocks, threads_per_block
        );

        let n_i32 = n as i32;
        let ensemble_size_i32 = ensemble_size as i32;

        unsafe {
            use cudarc::driver::LaunchAsync;
            let func_clone = cudarc::driver::CudaFunction::clone(&self.generate_kernel);
            func_clone.launch(
                config,
                (
                    &degrees_gpu,
                    &temps_gpu,
                    &seeds_gpu,
                    &mut orderings_gpu,
                    &mut energies_gpu,
                    n_i32,
                    ensemble_size_i32,
                ),
            )?;
        }

        // Synchronize
        self.device.synchronize()?;

        // Download results
        let orderings_flat: Vec<i32> = self.device.dtoh_sync_copy(&orderings_gpu)?;
        let energies: Vec<f32> = self.device.dtoh_sync_copy(&energies_gpu)?;

        // Reshape orderings
        let orderings: Vec<Vec<usize>> = (0..ensemble_size)
            .map(|i| {
                orderings_flat[i * n..(i + 1) * n]
                    .iter()
                    .map(|&v| v as usize)
                    .collect()
            })
            .collect();

        println!("[ENSEMBLE] ✅ Generated {} replicas", ensemble_size);
        println!(
            "[ENSEMBLE]   Energy range: [{:.2}, {:.2}]",
            energies
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            energies
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0)
        );

        Ok(Ensemble {
            orderings,
            temperatures,
            seeds,
            energies,
        })
    }

    /// Generate ensemble from graph adjacency
    ///
    /// Computes degrees automatically
    pub fn generate_from_adjacency(
        &self,
        adjacency: &ndarray::Array2<bool>,
        ensemble_size: usize,
        base_temperature: f32,
    ) -> Result<Ensemble> {
        let n = adjacency.nrows();

        // Compute degrees
        let degrees: Vec<usize> = (0..n)
            .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
            .collect();

        self.generate(&degrees, ensemble_size, base_temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    #[ignore] // Requires GPU
    fn test_ensemble_generation() {
        let generator = GpuEnsembleGenerator::new().expect("GPU required");

        // Simple graph: triangle
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;
        adj[[1, 2]] = true;
        adj[[2, 1]] = true;
        adj[[0, 2]] = true;
        adj[[2, 0]] = true;

        let ensemble = generator
            .generate_from_adjacency(&adj, 10, 1.0)
            .expect("Ensemble generation failed");

        assert_eq!(ensemble.orderings.len(), 10);
        assert_eq!(ensemble.temperatures.len(), 10);
        assert_eq!(ensemble.seeds.len(), 10);

        // Each ordering should have all vertices
        for ordering in &ensemble.orderings {
            assert_eq!(ordering.len(), 3);
            let mut sorted = ordering.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2]);
        }

        println!("✅ Ensemble generation test passed");
    }

    #[test]
    fn test_degree_computation() {
        let mut adj = Array2::from_elem((4, 4), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;
        adj[[1, 2]] = true;
        adj[[2, 1]] = true;
        adj[[2, 3]] = true;
        adj[[3, 2]] = true;

        let n = adj.nrows();
        let degrees: Vec<usize> = (0..n)
            .map(|i| adj.row(i).iter().filter(|&&x| x).count())
            .collect();

        assert_eq!(degrees, vec![1, 2, 2, 1]);
    }
}
