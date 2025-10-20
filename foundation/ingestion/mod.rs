//! Real-Time Data Ingestion Framework
//!
//! Provides async, high-throughput data ingestion from multiple sources
//! with buffering, backpressure handling, and <10ms latency targets.

pub mod buffer;
pub mod config;
pub mod engine;
pub mod error;
pub mod health;
pub mod types;

pub use buffer::CircularBuffer;
pub use config::{CircuitBreakerConfig, EngineConfig, IngestionConfig, RetryConfig, SourceConfig};
pub use engine::{IngestionEngine, IngestionStats};
pub use error::{CircuitBreaker, CircuitBreakerState, IngestionError, RetryPolicy};
pub use health::{ComponentHealth, HealthMetrics, HealthReport, HealthStatus};
pub use types::{DataPoint, DataSource, SourceInfo};
