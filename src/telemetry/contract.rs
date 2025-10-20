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
    AdapterStarted { description: String },
    AdapterProcessed { items: usize, duration_ms: f64 },
    AdapterFailed { error: String },
    PathDecision { details: String },
    Custom { payload: serde_json::Value },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    pub cpu_usage_pct: Option<f32>,
    pub memory_mb: Option<usize>,
    pub gpu_memory_mb: Option<usize>,
    pub throughput_per_sec: Option<f64>,
}
