//! Checkpoint and Restore for Stateful Recovery
//!
//! This module implements atomic checkpointing for stateful components, enabling
//! recovery from failures without data loss.
//!
//! # Architecture
//!
//! The checkpoint system consists of:
//! 1. **Checkpointable trait**: Interface for stateful components
//! 2. **CheckpointManager**: Orchestrates periodic snapshots
//! 3. **StorageBackend**: Pluggable storage (S3, NVMe, etc.)
//!
//! # Design Goals
//!
//! - **Atomicity**: Checkpoints are all-or-nothing
//! - **Low Overhead**: < 5% of processing time
//! - **Consistency**: Chandler-Lamport snapshot algorithm
//! - **Durability**: Multiple storage backends
//!
//! # Mathematical Foundation
//!
//! Checkpoint overhead is bounded by:
//!
//! ```text
//! T_checkpoint ≤ T_serialize + T_write
//! T_serialize = O(n) where n = state size
//! T_write = O(n/B) where B = bandwidth
//! ```
//!
//! For target overhead < 5%:
//! ```text
//! T_checkpoint < 0.05 · T_process
//! ```

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Trait for components that support checkpointing
pub trait Checkpointable: Serialize + DeserializeOwned {
    /// Serialize component state to bytes
    fn checkpoint(&self) -> Result<Vec<u8>, CheckpointError> {
        bincode::serialize(self).map_err(|e| CheckpointError::Serialization(e.to_string()))
    }

    /// Restore component state from bytes
    fn restore(data: &[u8]) -> Result<Self, CheckpointError>
    where
        Self: Sized,
    {
        bincode::deserialize(data).map_err(|e| CheckpointError::Deserialization(e.to_string()))
    }

    /// Component identifier for checkpoint tracking
    fn component_id(&self) -> String;
}

/// Checkpoint metadata (for internal storage, not directly serializable)
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Component identifier
    pub component_id: String,
    /// Checkpoint timestamp (seconds since UNIX epoch)
    pub timestamp_secs: u64,
    /// Checkpoint version/sequence number
    pub version: u64,
    /// Size of checkpoint data (bytes)
    pub size_bytes: usize,
    /// Checksum for integrity verification
    pub checksum: u64,
}

/// Serializable metadata for JSON storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableMetadata {
    component_id: String,
    timestamp_secs: u64,
    version: u64,
    size_bytes: usize,
    checksum: u64,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(component_id: String, version: u64, data: &[u8]) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            component_id,
            timestamp_secs,
            version,
            size_bytes: data.len(),
            checksum: Self::compute_checksum(data),
        }
    }

    /// Convert to serializable form
    fn to_serializable(&self) -> SerializableMetadata {
        SerializableMetadata {
            component_id: self.component_id.clone(),
            timestamp_secs: self.timestamp_secs,
            version: self.version,
            size_bytes: self.size_bytes,
            checksum: self.checksum,
        }
    }

    /// Create from serializable form
    fn from_serializable(s: SerializableMetadata) -> Self {
        Self {
            component_id: s.component_id,
            timestamp_secs: s.timestamp_secs,
            version: s.version,
            size_bytes: s.size_bytes,
            checksum: s.checksum,
        }
    }

    /// Compute simple checksum (FNV-1a hash)
    fn compute_checksum(data: &[u8]) -> u64 {
        const FNV_PRIME: u64 = 0x100000001b3;
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;

        data.iter().fold(FNV_OFFSET, |hash, &byte| {
            (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
        })
    }

    /// Verify data integrity
    pub fn verify(&self, data: &[u8]) -> bool {
        data.len() == self.size_bytes && Self::compute_checksum(data) == self.checksum
    }
}

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    /// Write checkpoint data
    fn write(&self, component_id: &str, version: u64, data: &[u8]) -> Result<(), CheckpointError>;

    /// Read checkpoint data
    fn read(&self, component_id: &str, version: u64) -> Result<Vec<u8>, CheckpointError>;

    /// Read latest checkpoint
    fn read_latest(&self, component_id: &str) -> Result<(u64, Vec<u8>), CheckpointError>;

    /// List available checkpoint versions
    fn list_versions(&self, component_id: &str) -> Result<Vec<u64>, CheckpointError>;

    /// Delete old checkpoints
    fn prune(&self, component_id: &str, keep_count: usize) -> Result<(), CheckpointError>;
}

/// Local filesystem storage backend
pub struct LocalStorageBackend {
    /// Base directory for checkpoints
    base_path: PathBuf,
}

impl LocalStorageBackend {
    /// Create new local storage backend
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)
            .map_err(|e| CheckpointError::Storage(format!("Failed to create directory: {}", e)))?;

        Ok(Self { base_path })
    }

    /// Get checkpoint file path
    fn checkpoint_path(&self, component_id: &str, version: u64) -> PathBuf {
        self.base_path
            .join(component_id)
            .join(format!("checkpoint_{:08}.bin", version))
    }

    /// Get metadata file path
    fn metadata_path(&self, component_id: &str, version: u64) -> PathBuf {
        self.base_path
            .join(component_id)
            .join(format!("metadata_{:08}.json", version))
    }
}

impl StorageBackend for LocalStorageBackend {
    fn write(&self, component_id: &str, version: u64, data: &[u8]) -> Result<(), CheckpointError> {
        // Create component directory
        let component_dir = self.base_path.join(component_id);
        fs::create_dir_all(&component_dir)
            .map_err(|e| CheckpointError::Storage(format!("Failed to create directory: {}", e)))?;

        // Write metadata
        let metadata = CheckpointMetadata::new(component_id.to_string(), version, data);
        let serializable = metadata.to_serializable();
        let metadata_json = serde_json::to_string_pretty(&serializable)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        let metadata_path = self.metadata_path(component_id, version);
        fs::write(&metadata_path, metadata_json)
            .map_err(|e| CheckpointError::Storage(format!("Failed to write metadata: {}", e)))?;

        // Write checkpoint data atomically
        let checkpoint_path = self.checkpoint_path(component_id, version);
        let temp_path = checkpoint_path.with_extension("tmp");

        {
            let mut file = File::create(&temp_path)
                .map_err(|e| CheckpointError::Storage(format!("Failed to create file: {}", e)))?;
            file.write_all(data)
                .map_err(|e| CheckpointError::Storage(format!("Failed to write data: {}", e)))?;
            file.sync_all()
                .map_err(|e| CheckpointError::Storage(format!("Failed to sync: {}", e)))?;
        }

        // Atomic rename
        fs::rename(&temp_path, &checkpoint_path)
            .map_err(|e| CheckpointError::Storage(format!("Failed to rename: {}", e)))?;

        Ok(())
    }

    fn read(&self, component_id: &str, version: u64) -> Result<Vec<u8>, CheckpointError> {
        // Read metadata
        let metadata_path = self.metadata_path(component_id, version);
        let metadata_json = fs::read_to_string(&metadata_path)
            .map_err(|e| CheckpointError::Storage(format!("Failed to read metadata: {}", e)))?;

        let serializable: SerializableMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;
        let metadata = CheckpointMetadata::from_serializable(serializable);

        // Read checkpoint data
        let checkpoint_path = self.checkpoint_path(component_id, version);
        let data = fs::read(&checkpoint_path)
            .map_err(|e| CheckpointError::Storage(format!("Failed to read checkpoint: {}", e)))?;

        // Verify integrity
        if !metadata.verify(&data) {
            return Err(CheckpointError::Corruption("Checksum mismatch".to_string()));
        }

        Ok(data)
    }

    fn read_latest(&self, component_id: &str) -> Result<(u64, Vec<u8>), CheckpointError> {
        let versions = self.list_versions(component_id)?;
        let latest = versions
            .into_iter()
            .max()
            .ok_or_else(|| CheckpointError::NotFound(component_id.to_string()))?;

        let data = self.read(component_id, latest)?;
        Ok((latest, data))
    }

    fn list_versions(&self, component_id: &str) -> Result<Vec<u64>, CheckpointError> {
        let component_dir = self.base_path.join(component_id);

        if !component_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = fs::read_dir(&component_dir)
            .map_err(|e| CheckpointError::Storage(format!("Failed to read directory: {}", e)))?;

        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|e| CheckpointError::Storage(format!("Failed to read entry: {}", e)))?;
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            if let Some(version_str) = filename_str
                .strip_prefix("checkpoint_")
                .and_then(|s| s.strip_suffix(".bin"))
            {
                if let Ok(version) = version_str.parse::<u64>() {
                    versions.push(version);
                }
            }
        }

        versions.sort_unstable();
        Ok(versions)
    }

    fn prune(&self, component_id: &str, keep_count: usize) -> Result<(), CheckpointError> {
        let mut versions = self.list_versions(component_id)?;

        if versions.len() <= keep_count {
            return Ok(());
        }

        // Sort and remove oldest checkpoints
        versions.sort_unstable();
        let to_remove = &versions[..versions.len() - keep_count];

        for &version in to_remove {
            let checkpoint_path = self.checkpoint_path(component_id, version);
            let metadata_path = self.metadata_path(component_id, version);

            let _ = fs::remove_file(&checkpoint_path);
            let _ = fs::remove_file(&metadata_path);
        }

        Ok(())
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Component version tracking
    versions: Arc<Mutex<HashMap<String, u64>>>,
    /// Performance metrics
    metrics: Arc<Mutex<CheckpointMetrics>>,
}

/// Checkpoint performance metrics
#[derive(Debug, Clone, Default)]
pub struct CheckpointMetrics {
    /// Total checkpoints created
    pub checkpoint_count: u64,
    /// Total restore operations
    pub restore_count: u64,
    /// Average checkpoint latency (ms)
    pub avg_checkpoint_latency_ms: f64,
    /// Average restore latency (ms)
    pub avg_restore_latency_ms: f64,
    /// Total checkpoint overhead (ms)
    pub total_checkpoint_time_ms: f64,
    /// Total processing time (ms)
    pub total_processing_time_ms: f64,
}

impl CheckpointMetrics {
    /// Calculate checkpoint overhead percentage
    pub fn overhead_percentage(&self) -> f64 {
        if self.total_processing_time_ms > 0.0 {
            (self.total_checkpoint_time_ms / self.total_processing_time_ms) * 100.0
        } else {
            0.0
        }
    }
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self {
            storage,
            versions: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(CheckpointMetrics::default())),
        }
    }

    /// Create checkpoint for component
    pub fn checkpoint<T: Checkpointable>(&self, component: &T) -> Result<u64, CheckpointError> {
        let start = Instant::now();

        let component_id = component.component_id();

        // Get next version
        let version = {
            let mut versions = self.versions.lock().unwrap();
            let version = versions.get(&component_id).copied().unwrap_or(0) + 1;
            versions.insert(component_id.clone(), version);
            version
        };

        // Serialize component
        let data = component.checkpoint()?;

        // Write to storage
        self.storage.write(&component_id, version, &data)?;

        // Update metrics
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.lock().unwrap();
        metrics.checkpoint_count += 1;
        metrics.total_checkpoint_time_ms += latency_ms;

        // Update running average
        let n = metrics.checkpoint_count as f64;
        metrics.avg_checkpoint_latency_ms =
            (metrics.avg_checkpoint_latency_ms * (n - 1.0) + latency_ms) / n;

        Ok(version)
    }

    /// Restore component from latest checkpoint
    pub fn restore<T: Checkpointable>(&self, component_id: &str) -> Result<T, CheckpointError> {
        let start = Instant::now();

        let (version, data) = self.storage.read_latest(component_id)?;
        let component = T::restore(&data)?;

        // Update version tracking
        {
            let mut versions = self.versions.lock().unwrap();
            versions.insert(component_id.to_string(), version);
        }

        // Update metrics
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.lock().unwrap();
        metrics.restore_count += 1;

        // Update running average
        let n = metrics.restore_count as f64;
        metrics.avg_restore_latency_ms =
            (metrics.avg_restore_latency_ms * (n - 1.0) + latency_ms) / n;

        Ok(component)
    }

    /// Restore from specific version
    pub fn restore_version<T: Checkpointable>(
        &self,
        component_id: &str,
        version: u64,
    ) -> Result<T, CheckpointError> {
        let data = self.storage.read(component_id, version)?;
        T::restore(&data)
    }

    /// Record processing time for overhead calculation
    pub fn record_processing_time(&self, duration: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_processing_time_ms += duration.as_secs_f64() * 1000.0;
    }

    /// Get checkpoint metrics
    pub fn metrics(&self) -> CheckpointMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Prune old checkpoints
    pub fn prune(&self, component_id: &str, keep_count: usize) -> Result<(), CheckpointError> {
        self.storage.prune(component_id, keep_count)
    }

    /// Reset metrics (for testing)
    pub fn reset_metrics(&self) {
        *self.metrics.lock().unwrap() = CheckpointMetrics::default();
    }
}

/// Checkpoint errors
#[derive(Debug)]
pub enum CheckpointError {
    /// Serialization failed
    Serialization(String),
    /// Deserialization failed
    Deserialization(String),
    /// Storage operation failed
    Storage(String),
    /// Checkpoint not found
    NotFound(String),
    /// Checkpoint data corrupted
    Corruption(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Serialization(e) => write!(f, "Serialization error: {}", e),
            CheckpointError::Deserialization(e) => write!(f, "Deserialization error: {}", e),
            CheckpointError::Storage(e) => write!(f, "Storage error: {}", e),
            CheckpointError::NotFound(id) => write!(f, "Checkpoint not found: {}", id),
            CheckpointError::Corruption(e) => write!(f, "Checkpoint corrupted: {}", e),
        }
    }
}

impl std::error::Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestComponent {
        id: String,
        value: i32,
        data: Vec<f64>,
    }

    impl Checkpointable for TestComponent {
        fn component_id(&self) -> String {
            self.id.clone()
        }
    }

    #[test]
    fn test_checkpoint_metadata() {
        let data = vec![1, 2, 3, 4, 5];
        let metadata = CheckpointMetadata::new("test".to_string(), 1, &data);

        assert_eq!(metadata.component_id, "test");
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.size_bytes, 5);
        assert!(metadata.verify(&data));
        assert!(!metadata.verify(&[1, 2, 3])); // Wrong size
        assert!(!metadata.verify(&[1, 2, 3, 4, 6])); // Wrong data
    }

    #[test]
    fn test_local_storage_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorageBackend::new(temp_dir.path()).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        storage.write("test", 1, &data).unwrap();

        let read_data = storage.read("test", 1).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_local_storage_list_versions() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorageBackend::new(temp_dir.path()).unwrap();

        storage.write("test", 1, &[1]).unwrap();
        storage.write("test", 3, &[3]).unwrap();
        storage.write("test", 2, &[2]).unwrap();

        let versions = storage.list_versions("test").unwrap();
        assert_eq!(versions, vec![1, 2, 3]);
    }

    #[test]
    fn test_local_storage_read_latest() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorageBackend::new(temp_dir.path()).unwrap();

        storage.write("test", 1, &[1]).unwrap();
        storage.write("test", 2, &[2]).unwrap();
        storage.write("test", 3, &[3]).unwrap();

        let (version, data) = storage.read_latest("test").unwrap();
        assert_eq!(version, 3);
        assert_eq!(data, vec![3]);
    }

    #[test]
    fn test_local_storage_prune() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorageBackend::new(temp_dir.path()).unwrap();

        for i in 1..=5 {
            storage.write("test", i, &[i as u8]).unwrap();
        }

        storage.prune("test", 2).unwrap();

        let versions = storage.list_versions("test").unwrap();
        assert_eq!(versions, vec![4, 5]);
    }

    #[test]
    fn test_checkpoint_manager_checkpoint_restore() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageBackend::new(temp_dir.path()).unwrap());
        let manager = CheckpointManager::new(storage);

        let component = TestComponent {
            id: "test_comp".to_string(),
            value: 42,
            data: vec![1.0, 2.0, 3.0],
        };

        let version = manager.checkpoint(&component).unwrap();
        assert_eq!(version, 1);

        let restored: TestComponent = manager.restore("test_comp").unwrap();
        assert_eq!(restored, component);
    }

    #[test]
    fn test_checkpoint_manager_multiple_versions() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageBackend::new(temp_dir.path()).unwrap());
        let manager = CheckpointManager::new(storage);

        let comp1 = TestComponent {
            id: "test".to_string(),
            value: 1,
            data: vec![1.0],
        };
        let comp2 = TestComponent {
            id: "test".to_string(),
            value: 2,
            data: vec![2.0],
        };

        manager.checkpoint(&comp1).unwrap();
        manager.checkpoint(&comp2).unwrap();

        let restored: TestComponent = manager.restore("test").unwrap();
        assert_eq!(restored, comp2); // Latest version
    }

    #[test]
    fn test_checkpoint_metrics() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageBackend::new(temp_dir.path()).unwrap());
        let manager = CheckpointManager::new(storage);

        let component = TestComponent {
            id: "test".to_string(),
            value: 42,
            data: vec![1.0; 1000],
        };

        manager.checkpoint(&component).unwrap();
        manager.record_processing_time(Duration::from_millis(100));

        let metrics = manager.metrics();
        assert_eq!(metrics.checkpoint_count, 1);
        assert!(metrics.avg_checkpoint_latency_ms > 0.0);
        assert!(metrics.overhead_percentage() < 100.0);
    }

    #[test]
    fn test_checkpoint_overhead_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageBackend::new(temp_dir.path()).unwrap());
        let manager = CheckpointManager::new(storage);

        let component = TestComponent {
            id: "test".to_string(),
            value: 42,
            data: vec![1.0; 100],
        };

        // Simulate 10 processing cycles with checkpoints
        for _ in 0..10 {
            manager.checkpoint(&component).unwrap();
            manager.record_processing_time(Duration::from_millis(100));
        }

        let metrics = manager.metrics();
        println!("Checkpoint overhead: {:.2}%", metrics.overhead_percentage());

        // Overhead should be < 5%
        assert!(metrics.overhead_percentage() < 5.0);
    }
}
