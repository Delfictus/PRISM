use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEntry {
    pub id: Uuid,
    pub timestamp_us: u128,
    pub component: ComponentId,
    pub level: EventLevel,
    pub event: EventData,
    pub correlation_id: Option<Uuid>,
    pub metrics: Option<Metrics>,
}

impl TelemetryEntry {
    pub fn new(
        component: ComponentId,
        level: EventLevel,
        event: EventData,
        correlation_id: Option<Uuid>,
        metrics: Option<Metrics>,
    ) -> Self {
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros())
            .unwrap_or_default();
        Self {
            id: Uuid::new_v4(),
            timestamp_us,
            component,
            level,
            event,
            correlation_id,
            metrics,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentId {
    CMAAdapter,
    GPUColoring,
    DensePathGuard,
    Orchestrator,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventLevel {
    Debug,
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventData {
    AdapterStarted {
        description: String,
    },
    AdapterProcessed {
        items: usize,
        duration_ms: f64,
    },
    AdapterFailed {
        error: String,
    },
    PathDecision {
        details: String,
    },
    AdapterStopped {
        reason: String,
    },
    ProcessingStarted {
        graph_size: usize,
        edges: usize,
        strategy: String,
    },
    ProcessingCompleted {
        colors_used: u32,
        duration_ms: f64,
        iterations: usize,
    },
    ProcessingFailed {
        error: String,
        recoverable: bool,
    },
    ConsensusProposed {
        vertex: usize,
        proposed_color: u32,
        confidence: f64,
    },
    ConsensusReached {
        vertex: usize,
        final_color: u32,
        agreement_score: f64,
    },
    ConsensusConflict {
        vertex: usize,
        proposals: Vec<(ComponentId, u32)>,
    },
    MemoryAllocation {
        bytes: usize,
        purpose: String,
    },
    MemoryPressure {
        used_mb: usize,
        available_mb: usize,
    },
    PerformanceCheckpoint {
        phase: String,
        elapsed_ms: f64,
        progress_pct: f64,
    },
    Custom {
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    pub cpu_usage_pct: Option<f32>,
    pub gpu_usage_pct: Option<f32>,
    pub memory_mb: Option<usize>,
    pub gpu_memory_mb: Option<usize>,
    pub throughput_per_sec: Option<f64>,
}
