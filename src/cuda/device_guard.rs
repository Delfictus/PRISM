use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use anyhow::Result;

static DEVICE_CAPS: OnceLock<DeviceCapabilities> = OnceLock::new();

/// Simplified device capability snapshot used for memory feasibility checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub name: String,
    pub total_memory_mb: usize,
    pub available_memory_mb: usize,
    pub tensor_cores: bool,
    pub fp16_support: bool,
    pub warp_size: usize,
    pub memory_clock_mhz: usize,
    pub memory_bus_width: usize,
}

impl DeviceCapabilities {
    pub fn detect() -> Result<Self> {
        // In lieu of direct CUDA bindings, allow overriding via environment variables
        // or fall back to conservative defaults approximating an A100 class device.
        let default_total = 80_000; // 80 GB
        let default_available = 64_000;
        let default_tensor = true;
        let default_fp16 = true;

        Ok(Self {
            name: std::env::var("PRISM_GPU_NAME").unwrap_or_else(|_| "Simulated GPU".into()),
            total_memory_mb: std::env::var("PRISM_GPU_MEMORY_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_total),
            available_memory_mb: std::env::var("PRISM_GPU_AVAILABLE_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_available),
            tensor_cores: std::env::var("PRISM_GPU_TENSOR_CORES")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(default_tensor),
            fp16_support: std::env::var("PRISM_GPU_FP16")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(default_fp16),
            warp_size: 32,
            memory_clock_mhz: std::env::var("PRISM_GPU_MEM_CLOCK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_500_000 / 1000),
            memory_bus_width: std::env::var("PRISM_GPU_BUS_WIDTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
        })
    }

    pub fn get_cached() -> &'static DeviceCapabilities {
        DEVICE_CAPS.get_or_init(|| Self::detect().expect("device detection"))
    }

    pub fn estimate_bandwidth_gbps(&self) -> f32 {
        // theoretical peak = 2 * memory_clock * bus_width / 8
        let clock_hz = (self.memory_clock_mhz as f32) * 1e6;
        let bits = 2.0 * clock_hz * self.memory_bus_width as f32;
        bits / 8.0 / 1e9
    }
}
