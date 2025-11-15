//! GPU-accelerated Linear Layer (using simple implementation)

use crate::gpu::gpu_enabled::{SimpleGpuLinear, SimpleGpuTensor};
use anyhow::Result;

/// GPU Linear layer (wraps SimpleGpuLinear)
pub struct GpuLinear {
    inner: SimpleGpuLinear,
    pub in_features: usize,
    pub out_features: usize,
}

impl GpuLinear {
    /// Create new GPU Linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        _pool: std::sync::Arc<super::super::gpu_enabled::SimpleGpuContext>,
    ) -> Result<Self> {
        let inner = SimpleGpuLinear::new(in_features, out_features)?;
        Ok(Self {
            inner,
            in_features,
            out_features,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &SimpleGpuTensor) -> Result<SimpleGpuTensor> {
        self.inner.forward(input)
    }

    /// Get configuration
    pub fn config(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }
}

/// Builder pattern for GpuLinear
pub struct GpuLinearBuilder {
    in_features: Option<usize>,
    out_features: Option<usize>,
}

impl GpuLinearBuilder {
    pub fn new() -> Self {
        Self {
            in_features: None,
            out_features: None,
        }
    }

    pub fn in_features(mut self, size: usize) -> Self {
        self.in_features = Some(size);
        self
    }

    pub fn out_features(mut self, size: usize) -> Self {
        self.out_features = Some(size);
        self
    }

    pub fn pool(self, _pool: std::sync::Arc<super::super::gpu_enabled::SimpleGpuContext>) -> Self {
        // Pool is ignored in simple implementation
        self
    }

    pub fn build(self) -> Result<GpuLinear> {
        let in_features = self
            .in_features
            .ok_or_else(|| anyhow::anyhow!("in_features not specified"))?;
        let out_features = self
            .out_features
            .ok_or_else(|| anyhow::anyhow!("out_features not specified"))?;

        let inner = SimpleGpuLinear::new(in_features, out_features)?;
        Ok(GpuLinear {
            inner,
            in_features,
            out_features,
        })
    }
}
