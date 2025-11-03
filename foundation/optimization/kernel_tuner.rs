//! Kernel-Level Configuration and Occupancy Analysis
//!
//! This module provides hardware-aware kernel configuration by querying GPU properties
//! and calculating theoretical occupancy to guide the auto-tuner's search space.
//!
//! # Occupancy Theory
//!
//! Occupancy is defined as the ratio of active warps per SM to maximum warps:
//!
//! ```text
//! Occupancy = active_warps_per_sm / max_warps_per_sm
//! ```
//!
//! Occupancy is limited by:
//! 1. **Threads per block**: Must be multiple of warp size (32)
//! 2. **Registers per thread**: Limited by SM register file
//! 3. **Shared memory per block**: Limited by SM shared memory
//!
//! # Occupancy Calculation
//!
//! Given a kernel configuration:
//! - threads_per_block = block_size
//! - registers_per_thread (from PTX/SASS analysis)
//! - shared_memory_per_block
//!
//! Calculate:
//! ```text
//! warps_per_block = ceil(threads_per_block / 32)
//! blocks_per_sm = min(
//!     max_blocks_per_sm,
//!     max_warps_per_sm / warps_per_block,
//!     shared_memory_per_sm / shared_memory_per_block,
//!     registers_per_sm / (registers_per_thread * threads_per_block)
//! )
//! active_warps = blocks_per_sm * warps_per_block
//! occupancy = active_warps / max_warps_per_sm
//! ```

use cudarc::driver::sys as cuda;

/// GPU device properties
#[derive(Debug, Clone)]
pub struct GpuProperties {
    /// Device name
    pub name: String,
    /// Compute capability (major.minor)
    pub compute_capability: (i32, i32),
    /// Number of streaming multiprocessors
    pub sm_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Maximum threads per SM
    pub max_threads_per_sm: i32,
    /// Maximum warps per SM
    pub max_warps_per_sm: i32,
    /// Maximum blocks per SM
    pub max_blocks_per_sm: i32,
    /// Shared memory per SM (bytes)
    pub shared_memory_per_sm: usize,
    /// Shared memory per block (bytes)
    pub shared_memory_per_block: usize,
    /// Registers per SM
    pub registers_per_sm: i32,
    /// Warp size (always 32 for NVIDIA GPUs)
    pub warp_size: i32,
}

/// Kernel configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Threads per block (block size)
    pub block_size: u32,
    /// Number of blocks (grid size)
    pub grid_size: u32,
    /// Dynamic shared memory per block (bytes)
    pub shared_memory: usize,
    /// Estimated registers per thread
    pub registers_per_thread: i32,
}

/// Occupancy analysis result
#[derive(Debug, Clone)]
pub struct OccupancyInfo {
    /// Theoretical occupancy (0.0 to 1.0)
    pub occupancy: f64,
    /// Active warps per SM
    pub active_warps_per_sm: i32,
    /// Blocks per SM
    pub blocks_per_sm: i32,
    /// Limiting factor
    pub limiting_factor: LimitingFactor,
}

/// Factor limiting occupancy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingFactor {
    /// Limited by warp count
    Warps,
    /// Limited by shared memory
    SharedMemory,
    /// Limited by register usage
    Registers,
    /// Limited by block count
    Blocks,
}

/// Kernel tuner for occupancy analysis
pub struct KernelTuner {
    /// GPU device properties
    properties: GpuProperties,
}

impl KernelTuner {
    /// Create new kernel tuner by querying device 0
    pub fn new() -> Result<Self, String> {
        Self::for_device(0)
    }

    /// Create kernel tuner for specific device
    pub fn for_device(device_id: i32) -> Result<Self, String> {
        unsafe {
            // Get device properties
            let mut props: cuda::CUdevprop = std::mem::zeroed();
            let result = cuda::cuDeviceGetProperties(&mut props, device_id);
            if result != cuda::CUresult::CUDA_SUCCESS {
                return Err(format!("Failed to get device properties: {:?}", result));
            }

            // Get device name
            let mut name_buf = [0i8; 256];
            cuda::cuDeviceGetName(name_buf.as_mut_ptr(), 256, device_id);
            let name = std::ffi::CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            // Get compute capability
            let mut major = 0;
            let mut minor = 0;
            cuda::cuDeviceGetAttribute(
                &mut major,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device_id,
            );
            cuda::cuDeviceGetAttribute(
                &mut minor,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device_id,
            );

            // Get SM count
            let mut sm_count = 0;
            cuda::cuDeviceGetAttribute(
                &mut sm_count,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device_id,
            );

            // Get max threads per block
            let mut max_threads_per_block = 0;
            cuda::cuDeviceGetAttribute(
                &mut max_threads_per_block,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                device_id,
            );

            // Get max threads per SM (compute capability dependent)
            let max_threads_per_sm = match (major, minor) {
                (7, 5) => 1024, // Turing
                (8, 0) => 2048, // Ampere
                (8, 6) => 1536, // Ampere (RTX 30xx)
                (8, 9) => 1536, // Ada Lovelace (RTX 40xx)
                (9, 0) => 2048, // Hopper
                _ => 2048,      // Default
            };

            let max_warps_per_sm = max_threads_per_sm / 32;

            // Get shared memory per SM
            let mut shared_memory_per_sm = 0;
            cuda::cuDeviceGetAttribute(
                &mut shared_memory_per_sm,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                device_id,
            );

            // Get shared memory per block
            let mut shared_memory_per_block = 0;
            cuda::cuDeviceGetAttribute(
                &mut shared_memory_per_block,
                cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                device_id,
            );

            // Get registers per SM (compute capability dependent)
            let registers_per_sm = match (major, minor) {
                (7, 5) => 65536,  // Turing
                (8, 0) => 65536,  // Ampere
                (8, 6) => 65536,  // Ampere (RTX 30xx)
                (8, 9) => 65536,  // Ada Lovelace (RTX 40xx)
                (9, 0) => 65536,  // Hopper
                _ => 65536,       // Default
            };

            // Max blocks per SM (hardware limit)
            let max_blocks_per_sm = 16; // Common across architectures

            let properties = GpuProperties {
                name,
                compute_capability: (major, minor),
                sm_count,
                max_threads_per_block,
                max_threads_per_sm,
                max_warps_per_sm,
                max_blocks_per_sm,
                shared_memory_per_sm: shared_memory_per_sm as usize,
                shared_memory_per_block: shared_memory_per_block as usize,
                registers_per_sm,
                warp_size: 32,
            };

            Ok(Self { properties })
        }
    }

    /// Get GPU properties
    pub fn get_properties(&self) -> &GpuProperties {
        &self.properties
    }

    /// Calculate theoretical occupancy for a kernel configuration
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// warps_per_block = ceil(block_size / warp_size)
    /// blocks_per_sm = min(
    ///     limit_blocks,
    ///     limit_warps,
    ///     limit_shared_mem,
    ///     limit_registers
    /// )
    /// occupancy = (blocks_per_sm * warps_per_block) / max_warps_per_sm
    /// ```
    pub fn calculate_occupancy(&self, config: &KernelConfig) -> OccupancyInfo {
        let props = &self.properties;

        // Calculate warps per block
        let warps_per_block = ((config.block_size + props.warp_size as u32 - 1)
            / props.warp_size as u32) as i32;

        // Limit by max blocks per SM
        let limit_blocks = props.max_blocks_per_sm;

        // Limit by warp count
        let limit_warps = if warps_per_block > 0 {
            props.max_warps_per_sm / warps_per_block
        } else {
            0
        };

        // Limit by shared memory
        let limit_shared_mem = if config.shared_memory > 0 {
            (props.shared_memory_per_sm / config.shared_memory) as i32
        } else {
            props.max_blocks_per_sm
        };

        // Limit by registers
        let registers_per_block = config.registers_per_thread * config.block_size as i32;
        let limit_registers = if registers_per_block > 0 {
            props.registers_per_sm / registers_per_block
        } else {
            props.max_blocks_per_sm
        };

        // Find minimum (most restrictive limit)
        let blocks_per_sm = limit_blocks
            .min(limit_warps)
            .min(limit_shared_mem)
            .min(limit_registers)
            .max(0);

        // Determine limiting factor
        let limiting_factor = if blocks_per_sm == limit_blocks && blocks_per_sm < limit_warps {
            LimitingFactor::Blocks
        } else if blocks_per_sm == limit_warps {
            LimitingFactor::Warps
        } else if blocks_per_sm == limit_shared_mem {
            LimitingFactor::SharedMemory
        } else {
            LimitingFactor::Registers
        };

        // Calculate occupancy
        let active_warps_per_sm = blocks_per_sm * warps_per_block;
        let occupancy = if props.max_warps_per_sm > 0 {
            (active_warps_per_sm as f64) / (props.max_warps_per_sm as f64)
        } else {
            0.0
        };

        OccupancyInfo {
            occupancy: occupancy.clamp(0.0, 1.0),
            active_warps_per_sm,
            blocks_per_sm,
            limiting_factor,
        }
    }

    /// Generate recommended configurations for a workload size
    ///
    /// Returns a list of KernelConfig candidates with good occupancy
    pub fn recommend_configs(&self, workload_size: usize) -> Vec<KernelConfig> {
        let mut configs = Vec::new();

        // Common block sizes (powers of 2 and multiples of warp size)
        let block_sizes = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

        for &block_size in &block_sizes {
            if block_size > self.properties.max_threads_per_block as u32 {
                continue;
            }

            // Calculate grid size to cover workload
            let grid_size = ((workload_size + block_size as usize - 1) / block_size as usize) as u32;

            // Estimate registers per thread (conservative)
            let registers_per_thread = 32;

            let config = KernelConfig {
                block_size,
                grid_size,
                shared_memory: 0, // No shared memory by default
                registers_per_thread,
            };

            // Only include configs with decent occupancy
            let occ = self.calculate_occupancy(&config);
            if occ.occupancy >= 0.5 {
                configs.push(config);
            }
        }

        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_tuner_creation() {
        // This test requires a CUDA-capable GPU
        if let Ok(tuner) = KernelTuner::new() {
            let props = tuner.get_properties();
            println!("GPU: {}", props.name);
            println!("Compute Capability: {}.{}", props.compute_capability.0, props.compute_capability.1);
            println!("SM Count: {}", props.sm_count);
            assert!(props.sm_count > 0);
            assert!(props.max_threads_per_block > 0);
            assert_eq!(props.warp_size, 32);
        }
    }

    #[test]
    fn test_occupancy_calculation() {
        if let Ok(tuner) = KernelTuner::new() {
            let config = KernelConfig {
                block_size: 256,
                grid_size: 100,
                shared_memory: 0,
                registers_per_thread: 32,
            };

            let occ = tuner.calculate_occupancy(&config);
            println!("Occupancy: {:.2}%", occ.occupancy * 100.0);
            println!("Active warps per SM: {}", occ.active_warps_per_sm);
            println!("Blocks per SM: {}", occ.blocks_per_sm);
            println!("Limiting factor: {:?}", occ.limiting_factor);

            assert!(occ.occupancy > 0.0);
            assert!(occ.occupancy <= 1.0);
        }
    }

    #[test]
    fn test_recommend_configs() {
        if let Ok(tuner) = KernelTuner::new() {
            let workload_size = 10000;
            let configs = tuner.recommend_configs(workload_size);

            println!("Recommended configs for workload size {}:", workload_size);
            for (i, config) in configs.iter().enumerate() {
                let occ = tuner.calculate_occupancy(config);
                println!("  Config {}: block={}, grid={}, occ={:.2}%",
                    i, config.block_size, config.grid_size, occ.occupancy * 100.0);
            }

            assert!(!configs.is_empty(), "Should recommend at least one config");
        }
    }
}
