//! Resilience and Fault Tolerance Module
//!
//! This module implements enterprise-grade reliability features for the Active Inference Platform:
//! - Health monitoring and graceful degradation
//! - Circuit breakers for cascading failure prevention
//! - Checkpoint/restore for stateful recovery
//!
//! # Architecture
//!
//! The resilience system consists of three main components:
//!
//! 1. **HealthMonitor**: Tracks component health and system state
//! 2. **CircuitBreaker**: Isolates failing components to prevent cascades
//! 3. **CheckpointManager**: Provides state snapshots for recovery
//!
//! # Design Principles
//!
//! - **Fail-Safe**: System degrades gracefully under failure
//! - **Self-Healing**: Automated recovery without human intervention
//! - **Observability**: Rich metrics for monitoring and debugging
//! - **Low Overhead**: Checkpoint overhead < 5% of processing time
//!
//! # Constitution Compliance
//!
//! This module implements Phase 4 Task 4.1 requirements:
//! - MTBF > 1000 hours with random failure injection
//! - Cascading failure prevention via circuit breakers
//! - State integrity through atomic checkpointing
//! - Production-grade error handling throughout

pub mod checkpoint_manager;
pub mod circuit_breaker;
pub mod fault_tolerance;

pub use fault_tolerance::{ComponentHealth, HealthMonitor, HealthStatus, SystemState};

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
};

pub use checkpoint_manager::{
    CheckpointError, CheckpointManager, CheckpointMetadata, Checkpointable, StorageBackend,
};
