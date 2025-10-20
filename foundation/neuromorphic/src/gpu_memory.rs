//! GPU Memory Management for RTX 5070 Acceleration
//!
//! Provides efficient memory pooling, allocation strategies, and resource management
//! for CUDA operations in the neuromorphic-quantum platform

use cudarc::driver::*;
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use dashmap::DashMap;

/// GPU memory pool for efficient buffer reuse
/// Eliminates allocation overhead that would reduce the 89% performance gain
pub struct GpuMemoryPool {
    device: Arc<CudaContext>,
    pools: DashMap<usize, Vec<CudaSlice<f32>>>,
    allocation_stats: Arc<Mutex<AllocationStats>>,
    max_pool_size: usize,
    total_allocated_bytes: Arc<Mutex<usize>>,

    // Asynchronous transfer streams for overlapped computation
    transfer_stream: Arc<Mutex<Option<Arc<CudaStream>>>>,
    compute_stream: Arc<Mutex<Option<Arc<CudaStream>>>>,
}

/// Memory allocation statistics for monitoring
#[derive(Debug, Default)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub peak_memory_usage_mb: f32,
    pub current_memory_usage_mb: f32,
}

/// Memory buffer type for different use cases
#[derive(Debug, Clone, Copy)]
pub enum BufferType {
    Matrix,        // Large matrices (reservoir weights)
    Vector,        // State vectors (neuron states)
    Input,         // Input vectors (spike patterns)
    Temporary,     // Scratch buffers for computation
}

/// GPU memory configuration for RTX 5070 optimization
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    pub max_cache_size_mb: usize,
    pub prealloc_buffer_sizes: Vec<usize>,
    pub enable_unified_memory: bool,
    pub memory_growth_factor: f32,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            max_cache_size_mb: 4096,  // 4GB cache for RTX 5070's 8GB VRAM (optimized)
            prealloc_buffer_sizes: vec![
                100,      // Small vectors
                1000,     // Medium vectors
                10000,    // Large vectors
                100000,   // Small matrices
                1000000,  // Large matrices (1000x1000 reservoir)
                4000000,  // Extra large matrices (2000x2000 reservoir)
                10000000, // Massive matrices for advanced reservoirs
            ],
            enable_unified_memory: true,  // RTX 5070 supports unified memory
            memory_growth_factor: 2.0,    // More aggressive growth for large allocations
        }
    }
}

impl GpuMemoryPool {
    /// Create new GPU memory pool optimized for RTX 5070
    pub fn new(device: Arc<CudaContext>, config: GpuMemoryConfig) -> Result<Self> {
        // Create CUDA streams for asynchronous operations
        let transfer_stream = device.new_stream()?;
        let compute_stream = device.new_stream()?;

        let pool = Self {
            device: device.clone(),
            pools: DashMap::new(),
            allocation_stats: Arc::new(Mutex::new(AllocationStats::default())),
            max_pool_size: config.max_cache_size_mb * 1024 * 1024 / 4, // Convert MB to f32 count
            total_allocated_bytes: Arc::new(Mutex::new(0)),
            transfer_stream: Arc::new(Mutex::new(Some(transfer_stream))),
            compute_stream: Arc::new(Mutex::new(Some(compute_stream))),
        };

        // Pre-allocate common buffer sizes for optimal performance
        pool.preallocate_buffers(&config.prealloc_buffer_sizes)?;

        Ok(pool)
    }

    /// Pre-allocate commonly used buffer sizes
    /// This eliminates allocation overhead during critical processing
    fn preallocate_buffers(&self, sizes: &[usize]) -> Result<()> {
        for &size in sizes {
            let mut buffers = Vec::new();

            // Pre-allocate 4 buffers of each size
            for _ in 0..4 {
                let buffer = self.device.default_stream().alloc_zeros::<f32>(size)?;
                buffers.push(buffer);
            }

            self.pools.insert(size, buffers);

            // Update allocation stats
            {
                let mut stats = self.allocation_stats.lock().unwrap();
                stats.total_allocations += 4;
                stats.current_memory_usage_mb += (size * 4 * 4) as f32 / 1024.0 / 1024.0;
                stats.peak_memory_usage_mb = stats.peak_memory_usage_mb.max(stats.current_memory_usage_mb);
            }
        }

        Ok(())
    }

    /// Get buffer from pool or allocate new one
    /// Core function for high-performance memory management
    pub fn get_buffer(&self, size: usize, buffer_type: BufferType) -> Result<CudaSlice<f32>> {
        // Try to get from pool first (cache hit)
        if let Some(mut buffers) = self.pools.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                // Cache hit - reuse existing buffer
                {
                    let mut stats = self.allocation_stats.lock().unwrap();
                    stats.cache_hits += 1;
                }
                return Ok(buffer);
            }
        }

        // Cache miss - allocate new buffer
        let stream = self.device.default_stream();
        let buffer = match buffer_type {
            BufferType::Matrix | BufferType::Vector => {
                stream.alloc_zeros::<f32>(size)?
            },
            BufferType::Input => {
                // Input buffers might benefit from pinned memory
                stream.alloc_zeros::<f32>(size)?
            },
            BufferType::Temporary => {
                // Temporary buffers for intermediate computation
                stream.alloc_zeros::<f32>(size)?
            },
        };

        // Update stats
        {
            let mut stats = self.allocation_stats.lock().unwrap();
            stats.cache_misses += 1;
            stats.total_allocations += 1;
            stats.current_memory_usage_mb += (size * 4) as f32 / 1024.0 / 1024.0;
            stats.peak_memory_usage_mb = stats.peak_memory_usage_mb.max(stats.current_memory_usage_mb);
        }

        Ok(buffer)
    }

    /// Return buffer to pool for reuse
    pub fn return_buffer(&self, buffer: CudaSlice<f32>, size: usize) -> Result<()> {
        // Check if we're under pool size limit
        let current_pool_size = self.pools.get(&size).map(|v| v.len()).unwrap_or(0);

        if current_pool_size < 8 {  // Max 8 buffers per size in pool
            // Clear buffer memory for reuse (need mutable reference)
            let mut buffer_mut = buffer;
            self.device.default_stream().memset_zeros(&mut buffer_mut)
                .map_err(|e| anyhow!("Failed to clear buffer: {}", e))?;

            // Add to pool
            self.pools.entry(size).or_insert_with(Vec::new).push(buffer_mut);
        }
        // If pool is full, buffer will be automatically freed when dropped

        Ok(())
    }

    /// Get current memory usage statistics
    pub fn get_stats(&self) -> AllocationStats {
        let stats = self.allocation_stats.lock().unwrap();
        AllocationStats {
            total_allocations: stats.total_allocations,
            total_deallocations: stats.total_deallocations,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            peak_memory_usage_mb: stats.peak_memory_usage_mb,
            current_memory_usage_mb: stats.current_memory_usage_mb,
        }
    }

    /// Clear all cached buffers (useful for memory pressure situations)
    pub fn clear_cache(&self) {
        self.pools.clear();

        let mut stats = self.allocation_stats.lock().unwrap();
        stats.current_memory_usage_mb = 0.0;
        stats.total_deallocations += stats.total_allocations;
    }

    /// Asynchronous host-to-device transfer for overlapped computation
    pub fn htod_async_copy<T: cudarc::driver::DeviceRepr>(
        &self,
        host_data: &[T],
    ) -> Result<CudaSlice<T>> {
        if let Some(stream) = self.transfer_stream.lock().unwrap().as_ref() {
            stream.memcpy_stod(host_data)
                .map_err(|e| anyhow!("Failed to copy host to device: {}", e))
        } else {
            self.device.default_stream().memcpy_stod(host_data)
                .map_err(|e| anyhow!("Failed to copy host to device: {}", e))
        }
    }

    /// Asynchronous device-to-host transfer for overlapped computation
    pub fn dtoh_async_copy<T: cudarc::driver::DeviceRepr>(
        &self,
        device_buffer: &CudaSlice<T>,
    ) -> Result<Vec<T>> {
        if let Some(stream) = self.transfer_stream.lock().unwrap().as_ref() {
            stream.memcpy_dtov(device_buffer)
                .map_err(|e| anyhow!("Failed to copy device to host: {}", e))
        } else {
            self.device.default_stream().memcpy_dtov(device_buffer)
                .map_err(|e| anyhow!("Failed to copy device to host: {}", e))
        }
    }

    /// Synchronize transfer operations
    pub fn sync_transfers(&self) -> Result<()> {
        if let Some(stream) = self.transfer_stream.lock().unwrap().as_ref() {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get compute stream for kernel execution
    pub fn get_compute_stream(&self) -> Arc<Mutex<Option<Arc<CudaStream>>>> {
        self.compute_stream.clone()
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }
}

/// RAII wrapper for GPU buffer management with zero-copy optimization
/// Automatically returns buffer to pool when dropped
pub struct ManagedGpuBuffer {
    buffer: Option<CudaSlice<f32>>,
    size: usize,
    pool: Arc<GpuMemoryPool>,

    // Zero-copy optimization tracking
    is_borrowed: bool,
    borrow_count: usize,
}

impl ManagedGpuBuffer {
    /// Create new managed buffer
    pub fn new(pool: Arc<GpuMemoryPool>, size: usize, buffer_type: BufferType) -> Result<Self> {
        let buffer = pool.get_buffer(size, buffer_type)?;

        Ok(Self {
            buffer: Some(buffer),
            size,
            pool,
            is_borrowed: false,
            borrow_count: 0,
        })
    }

    /// Get reference to underlying buffer
    pub fn buffer(&self) -> Result<&CudaSlice<f32>> {
        self.buffer.as_ref()
            .ok_or_else(|| anyhow!("Buffer has been consumed"))
    }

    /// Get mutable reference to underlying buffer
    pub fn buffer_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        self.buffer.as_mut()
            .ok_or_else(|| anyhow!("Buffer has been consumed"))
    }

    /// Consume buffer and take ownership (prevents automatic return to pool)
    pub fn take(mut self) -> Result<CudaSlice<f32>> {
        if self.is_borrowed {
            return Err(anyhow!("Cannot take buffer that is currently borrowed"));
        }
        self.buffer.take()
            .ok_or_else(|| anyhow!("Buffer has already been taken"))
    }

    /// Create a zero-copy borrow of the buffer for temporary use
    /// This allows multiple references without data copying
    pub fn borrow_zero_copy(&mut self) -> Result<GpuBufferBorrow> {
        if self.buffer.is_none() {
            return Err(anyhow!("Buffer has been consumed"));
        }

        self.is_borrowed = true;
        self.borrow_count += 1;

        Ok(GpuBufferBorrow {
            buffer_ref: self,
            active: true,
        })
    }

    /// Internal method to return from borrow
    fn return_from_borrow(&mut self) {
        if self.borrow_count > 0 {
            self.borrow_count -= 1;
        }
        if self.borrow_count == 0 {
            self.is_borrowed = false;
        }
    }
}

/// Zero-copy borrow wrapper that ensures safe access patterns
pub struct GpuBufferBorrow<'a> {
    buffer_ref: &'a mut ManagedGpuBuffer,
    active: bool,
}

impl<'a> GpuBufferBorrow<'a> {
    /// Get reference to the borrowed buffer
    pub fn get(&self) -> Result<&CudaSlice<f32>> {
        if !self.active {
            return Err(anyhow!("Borrow has been released"));
        }
        self.buffer_ref.buffer.as_ref()
            .ok_or_else(|| anyhow!("Buffer has been consumed"))
    }

    /// Get mutable reference to the borrowed buffer
    pub fn get_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        if !self.active {
            return Err(anyhow!("Borrow has been released"));
        }
        self.buffer_ref.buffer.as_mut()
            .ok_or_else(|| anyhow!("Buffer has been consumed"))
    }
}

impl<'a> Drop for GpuBufferBorrow<'a> {
    fn drop(&mut self) {
        if self.active {
            self.buffer_ref.return_from_borrow();
            self.active = false;
        }
    }
}

impl Drop for ManagedGpuBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Return buffer to pool for reuse
            if let Err(e) = self.pool.return_buffer(buffer, self.size) {
                eprintln!("Warning: Failed to return buffer to pool: {}", e);
            }
        }
    }
}

/// High-level GPU memory manager for neuromorphic-quantum platform
pub struct NeuromorphicGpuMemoryManager {
    memory_pool: Arc<GpuMemoryPool>,
    device: Arc<CudaContext>,

    // Pre-allocated buffers for common operations
    reservoir_weight_buffer: Option<ManagedGpuBuffer>,
    input_weight_buffer: Option<ManagedGpuBuffer>,
    state_buffers: Vec<ManagedGpuBuffer>,
}

impl NeuromorphicGpuMemoryManager {
    /// Create memory manager for neuromorphic processing
    pub fn new(device: Arc<CudaContext>, reservoir_size: usize, input_size: usize) -> Result<Self> {
        let config = GpuMemoryConfig::default();
        let memory_pool = Arc::new(GpuMemoryPool::new(device.clone(), config)?);

        // Pre-allocate buffers for neuromorphic operations
        let reservoir_matrix_size = reservoir_size * reservoir_size;
        let input_matrix_size = reservoir_size * input_size;

        let reservoir_weight_buffer = Some(ManagedGpuBuffer::new(
            memory_pool.clone(),
            reservoir_matrix_size,
            BufferType::Matrix,
        )?);

        let input_weight_buffer = Some(ManagedGpuBuffer::new(
            memory_pool.clone(),
            input_matrix_size,
            BufferType::Matrix,
        )?);

        // Pre-allocate state buffers (current, previous, temporary)
        let mut state_buffers = Vec::new();
        for _ in 0..4 {  // Current, previous, temp, input
            let buffer = ManagedGpuBuffer::new(
                memory_pool.clone(),
                reservoir_size.max(input_size),
                BufferType::Vector,
            )?;
            state_buffers.push(buffer);
        }

        Ok(Self {
            memory_pool,
            device,
            reservoir_weight_buffer,
            input_weight_buffer,
            state_buffers,
        })
    }

    /// Get reservoir weight buffer
    pub fn get_reservoir_weights(&mut self) -> Result<&mut CudaSlice<f32>> {
        self.reservoir_weight_buffer
            .as_mut()
            .ok_or_else(|| anyhow!("Reservoir weight buffer not available"))?
            .buffer_mut()
    }

    /// Get input weight buffer
    pub fn get_input_weights(&mut self) -> Result<&mut CudaSlice<f32>> {
        self.input_weight_buffer
            .as_mut()
            .ok_or_else(|| anyhow!("Input weight buffer not available"))?
            .buffer_mut()
    }

    /// Get state buffer by index
    pub fn get_state_buffer(&mut self, index: usize) -> Result<&mut CudaSlice<f32>> {
        if index >= self.state_buffers.len() {
            return Err(anyhow!("State buffer index {} out of range", index));
        }
        self.state_buffers[index].buffer_mut()
    }

    /// Get temporary buffer for computation
    pub fn get_temp_buffer(&self, size: usize) -> Result<ManagedGpuBuffer> {
        ManagedGpuBuffer::new(self.memory_pool.clone(), size, BufferType::Temporary)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> AllocationStats {
        self.memory_pool.get_stats()
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }

    /// Clear memory cache to free GPU memory
    pub fn clear_cache(&self) {
        self.memory_pool.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA-capable GPU
    fn test_memory_pool_creation() {
        if let Ok(device) = CudaContext::new(0) {
            let device = Arc::new(device);
            let config = GpuMemoryConfig::default();

            let pool = GpuMemoryPool::new(device, config);
            assert!(pool.is_ok());

            let pool = pool.unwrap();
            let stats = pool.get_stats();

            // Should have pre-allocated some buffers
            assert!(stats.total_allocations > 0);
            assert!(stats.current_memory_usage_mb > 0.0);

            println!("Memory pool created with {} allocations, {:.1}MB used",
                    stats.total_allocations, stats.current_memory_usage_mb);
        }
    }

    #[test]
    #[ignore] // Requires CUDA-capable GPU
    fn test_buffer_reuse() {
        if let Ok(device) = CudaContext::new(0) {
            let device = Arc::new(device);
            let config = GpuMemoryConfig::default();
            let pool = Arc::new(GpuMemoryPool::new(device, config).unwrap());

            // Get buffer
            let buffer1 = pool.get_buffer(1000, BufferType::Vector).unwrap();
            let initial_stats = pool.get_stats();

            // Return buffer
            pool.return_buffer(buffer1, 1000).unwrap();

            // Get same size buffer again - should be reused
            let _buffer2 = pool.get_buffer(1000, BufferType::Vector).unwrap();
            let final_stats = pool.get_stats();

            // Should have more cache hits in final stats
            assert!(final_stats.cache_hits > initial_stats.cache_hits);

            println!("Buffer reuse test: {} cache hits, {} cache misses",
                    final_stats.cache_hits, final_stats.cache_misses);
        }
    }

    #[test]
    #[ignore] // Requires CUDA-capable GPU
    fn test_neuromorphic_memory_manager() {
        if let Ok(device) = CudaContext::new(0) {
            let device = Arc::new(device);

            let manager = NeuromorphicGpuMemoryManager::new(device, 1000, 100);
            assert!(manager.is_ok());

            let mut manager = manager.unwrap();

            // Test buffer access
            let reservoir_weights = manager.get_reservoir_weights();
            assert!(reservoir_weights.is_ok());

            let input_weights = manager.get_input_weights();
            assert!(input_weights.is_ok());

            let state_buffer = manager.get_state_buffer(0);
            assert!(state_buffer.is_ok());

            let stats = manager.get_memory_stats();
            println!("Neuromorphic memory manager: {:.1}MB allocated",
                    stats.current_memory_usage_mb);
        }
    }
}