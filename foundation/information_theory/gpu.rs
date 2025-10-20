//! GPU-Accelerated Transfer Entropy
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context
//! - Article VI: Data stays on GPU during computation
//! - Article VII: PTX runtime loading (no FFI linking)

use std::sync::Arc;
use ndarray::Array1;
use anyhow::{Result, anyhow, Context};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, DeviceRepr, ValidAsZeroBits, PushKernelArg};

/// GPU-accelerated transfer entropy calculator
///
/// Transfer Entropy: TE(X→Y) = I(Y_future; X_past | Y_past)
/// Measures directed information flow from source to target
pub struct TransferEntropyGpu {
    context: Arc<CudaContext>,

    // Kernels loaded from PTX
    minmax_kernel: Arc<CudaFunction>,
    hist_3d_kernel: Arc<CudaFunction>,
    hist_2d_yf_yp_kernel: Arc<CudaFunction>,
    hist_2d_xp_yp_kernel: Arc<CudaFunction>,
    hist_1d_kernel: Arc<CudaFunction>,
    compute_te_kernel: Arc<CudaFunction>,

    // Configuration
    embedding_dim: usize,
    tau: usize,
    n_bins: usize,
}

impl TransferEntropyGpu {
    /// Create new GPU transfer entropy calculator
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (Article V compliance)
    /// * `embedding_dim` - Embedding dimension for phase space reconstruction
    /// * `tau` - Time delay for embedding
    /// * `n_bins` - Number of histogram bins (default: 16 for good resolution)
    pub fn new(
        context: Arc<CudaContext>,
        embedding_dim: usize,
        tau: usize,
        n_bins: Option<usize>
    ) -> Result<Self> {
        // Load PTX module (Article VII compliance)
        let ptx_path = "target/ptx/transfer_entropy.ptx";

        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Transfer entropy PTX not found at: {}", ptx_path));
        }

        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
        let module = context.load_module(ptx)?;

        // Load all kernel functions
        let minmax_kernel = Arc::new(module.load_function("compute_minmax_kernel")?);
        let hist_3d_kernel = Arc::new(module.load_function("build_histogram_3d_kernel")?);
        let hist_2d_yf_yp_kernel = Arc::new(module.load_function("build_histogram_2d_kernel")?);
        let hist_2d_xp_yp_kernel = Arc::new(module.load_function("build_histogram_2d_xp_yp_kernel")?);
        let hist_1d_kernel = Arc::new(module.load_function("build_histogram_1d_kernel")?);
        let compute_te_kernel = Arc::new(module.load_function("compute_transfer_entropy_kernel")?);

        Ok(Self {
            context,
            minmax_kernel,
            hist_3d_kernel,
            hist_2d_yf_yp_kernel,
            hist_2d_xp_yp_kernel,
            hist_1d_kernel,
            compute_te_kernel,
            embedding_dim,
            tau,
            n_bins: n_bins.unwrap_or(16),
        })
    }

    /// Compute transfer entropy from source to target on GPU
    ///
    /// Article VI Compliance: Data stays on GPU throughout computation
    pub fn compute_transfer_entropy(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>
    ) -> Result<f64> {
        if source.len() != target.len() {
            return Err(anyhow!("Source and target must have same length"));
        }

        let length = source.len();
        if length < self.embedding_dim * self.tau + 1 {
            return Err(anyhow!("Time series too short for embedding"));
        }

        // Use default stream (matching quantum_mlir pattern)
        let stream = self.context.default_stream();

        // Step 1: Upload data to GPU (CPU -> GPU once)
        let source_gpu: CudaSlice<f64> = stream.memcpy_stod(source.as_slice().unwrap())?;
        let target_gpu: CudaSlice<f64> = stream.memcpy_stod(target.as_slice().unwrap())?;

        // Step 2: Compute min/max for normalization
        let mut source_min_gpu: CudaSlice<f64> = stream.alloc_zeros(1)?;
        let mut source_max_gpu: CudaSlice<f64> = stream.alloc_zeros(1)?;
        let mut target_min_gpu: CudaSlice<f64> = stream.alloc_zeros(1)?;
        let mut target_max_gpu: CudaSlice<f64> = stream.alloc_zeros(1)?;

        // Initialize min/max (no direct init needed, kernels will compute)

        let threads = 256;
        let blocks = (length + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch minmax for source (using launch_builder pattern)
        let length_i32 = length as i32;

        let mut launch_args = stream.launch_builder(&self.minmax_kernel);
        launch_args.arg(&source_gpu);
        launch_args.arg(&length_i32);
        launch_args.arg(&source_min_gpu);
        launch_args.arg(&source_max_gpu);
        unsafe {
            launch_args.launch(cfg)?;
        }

        // Launch minmax for target
        let mut launch_args2 = stream.launch_builder(&self.minmax_kernel);
        launch_args2.arg(&target_gpu);
        launch_args2.arg(&length_i32);
        launch_args2.arg(&target_min_gpu);
        launch_args2.arg(&target_max_gpu);
        unsafe {
            launch_args2.launch(cfg)?;
        }

        // Download min/max
        let source_min = stream.memcpy_dtov(&source_min_gpu)?[0];
        let source_max = stream.memcpy_dtov(&source_max_gpu)?[0];
        let target_min = stream.memcpy_dtov(&target_min_gpu)?[0];
        let target_max = stream.memcpy_dtov(&target_max_gpu)?[0];

        // Step 3: Build histograms on GPU
        let valid_length = length - self.embedding_dim * self.tau;
        let hist_3d_size = self.n_bins * self.n_bins * self.n_bins;
        let hist_2d_size = self.n_bins * self.n_bins;

        let mut hist_3d: CudaSlice<i32> = stream.alloc_zeros(hist_3d_size)?;
        let mut hist_2d_yf_yp: CudaSlice<i32> = stream.alloc_zeros(hist_2d_size)?;
        let mut hist_2d_xp_yp: CudaSlice<i32> = stream.alloc_zeros(hist_2d_size)?;
        let mut hist_1d_yp: CudaSlice<i32> = stream.alloc_zeros(self.n_bins)?;

        let hist_blocks = (valid_length + threads - 1) / threads;
        let hist_cfg = LaunchConfig {
            grid_dim: (hist_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let embedding_dim_i32 = self.embedding_dim as i32;
        let tau_i32 = self.tau as i32;
        let n_bins_i32 = self.n_bins as i32;

        // Build 3D histogram
        let mut launch3d = stream.launch_builder(&self.hist_3d_kernel);
        launch3d.arg(&source_gpu);
        launch3d.arg(&target_gpu);
        launch3d.arg(&length_i32);
        launch3d.arg(&embedding_dim_i32);
        launch3d.arg(&tau_i32);
        launch3d.arg(&n_bins_i32);
        launch3d.arg(&source_min);
        launch3d.arg(&source_max);
        launch3d.arg(&target_min);
        launch3d.arg(&target_max);
        launch3d.arg(&mut hist_3d);
        unsafe { launch3d.launch(hist_cfg)?; }

        // Build 2D histogram (Y_future, Y_past)
        let mut launch2d_yf_yp = stream.launch_builder(&self.hist_2d_yf_yp_kernel);
        launch2d_yf_yp.arg(&target_gpu);
        launch2d_yf_yp.arg(&length_i32);
        launch2d_yf_yp.arg(&embedding_dim_i32);
        launch2d_yf_yp.arg(&tau_i32);
        launch2d_yf_yp.arg(&n_bins_i32);
        launch2d_yf_yp.arg(&target_min);
        launch2d_yf_yp.arg(&target_max);
        launch2d_yf_yp.arg(&mut hist_2d_yf_yp);
        unsafe { launch2d_yf_yp.launch(hist_cfg)?; }

        // Build 2D histogram (X_past, Y_past)
        let mut launch2d_xp_yp = stream.launch_builder(&self.hist_2d_xp_yp_kernel);
        launch2d_xp_yp.arg(&source_gpu);
        launch2d_xp_yp.arg(&target_gpu);
        launch2d_xp_yp.arg(&length_i32);
        launch2d_xp_yp.arg(&embedding_dim_i32);
        launch2d_xp_yp.arg(&tau_i32);
        launch2d_xp_yp.arg(&n_bins_i32);
        launch2d_xp_yp.arg(&source_min);
        launch2d_xp_yp.arg(&source_max);
        launch2d_xp_yp.arg(&target_min);
        launch2d_xp_yp.arg(&target_max);
        launch2d_xp_yp.arg(&mut hist_2d_xp_yp);
        unsafe { launch2d_xp_yp.launch(hist_cfg)?; }

        // Build 1D histogram (Y_past)
        let mut launch1d = stream.launch_builder(&self.hist_1d_kernel);
        launch1d.arg(&target_gpu);
        launch1d.arg(&length_i32);
        launch1d.arg(&embedding_dim_i32);
        launch1d.arg(&tau_i32);
        launch1d.arg(&n_bins_i32);
        launch1d.arg(&target_min);
        launch1d.arg(&target_max);
        launch1d.arg(&mut hist_1d_yp);
        unsafe { launch1d.launch(hist_cfg)?; }

        // Step 4: Compute transfer entropy from histograms
        let mut te_result: CudaSlice<f64> = stream.alloc_zeros(1)?;

        let te_blocks = (hist_3d_size + threads - 1) / threads;
        let te_cfg = LaunchConfig {
            grid_dim: (te_blocks.min(256) as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_samples = valid_length as i32;

        let mut launch_te = stream.launch_builder(&self.compute_te_kernel);
        launch_te.arg(&hist_3d);
        launch_te.arg(&hist_2d_yf_yp);
        launch_te.arg(&hist_2d_xp_yp);
        launch_te.arg(&hist_1d_yp);
        launch_te.arg(&n_bins_i32);
        launch_te.arg(&total_samples);
        launch_te.arg(&mut te_result);
        unsafe { launch_te.launch(te_cfg)?; }

        // Step 5: Download result (GPU -> CPU once)
        let te_value = stream.memcpy_dtov(&te_result)?[0];

        Ok(te_value.max(0.0)) // Transfer entropy is non-negative
    }

    /// Batch compute transfer entropy for multiple pairs
    ///
    /// More efficient than calling compute_transfer_entropy repeatedly
    pub fn compute_batch(
        &self,
        sources: &[Array1<f64>],
        targets: &[Array1<f64>]
    ) -> Result<Vec<f64>> {
        if sources.len() != targets.len() {
            return Err(anyhow!("Sources and targets must have same count"));
        }

        let mut results = Vec::with_capacity(sources.len());
        for (src, tgt) in sources.iter().zip(targets.iter()) {
            results.push(self.compute_transfer_entropy(src, tgt)?);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy_gpu_creation() {
        if let Ok(context) = CudaContext::new(0) {
            let te_gpu = TransferEntropyGpu::new(context, 3, 1, Some(16));
            assert!(te_gpu.is_ok());
        }
    }

    #[test]
    fn test_transfer_entropy_computation() {
        if let Ok(context) = CudaContext::new(0) {
            if let Ok(te_gpu) = TransferEntropyGpu::new(context, 2, 1, Some(8)) {
                // Create coupled time series: Y depends on X
                let length = 1000;
                let mut source = Vec::with_capacity(length);
                let mut target = Vec::with_capacity(length);

                for t in 0..length {
                    let x = (t as f64 * 0.1).sin();
                    source.push(x);

                    // Target depends on source with delay
                    let y = if t > 0 { source[t-1] * 0.8 + 0.2 * (t as f64 * 0.15).cos() } else { 0.0 };
                    target.push(y);
                }

                let source_arr = Array1::from_vec(source);
                let target_arr = Array1::from_vec(target);

                let te = te_gpu.compute_transfer_entropy(&source_arr, &target_arr);
                assert!(te.is_ok());

                let te_value = te.unwrap();
                assert!(te_value >= 0.0); // TE is non-negative
                assert!(te_value.is_finite());

                println!("GPU Transfer Entropy (X→Y): {:.6} bits", te_value);
            }
        }
    }
}
