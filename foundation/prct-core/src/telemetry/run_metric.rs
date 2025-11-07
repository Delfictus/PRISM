//! Runtime Telemetry Metrics
//!
//! Captures fine-grained performance and progress data during pipeline execution.
//! Metrics are streamed to JSONL for real-time monitoring and post-run analysis.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Pipeline phase identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseName {
    Reservoir,
    TransferEntropy,
    ActiveInference,
    Thermodynamic,
    Quantum,
    Memetic,
    Ensemble,
    Validation,
}

impl fmt::Display for PhaseName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseName::Reservoir => write!(f, "RESERVOIR"),
            PhaseName::TransferEntropy => write!(f, "TE"),
            PhaseName::ActiveInference => write!(f, "AI"),
            PhaseName::Thermodynamic => write!(f, "THERMO"),
            PhaseName::Quantum => write!(f, "QUANTUM"),
            PhaseName::Memetic => write!(f, "MEMETIC"),
            PhaseName::Ensemble => write!(f, "ENSEMBLE"),
            PhaseName::Validation => write!(f, "VALID"),
        }
    }
}

/// Execution mode for phase
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PhaseExecMode {
    /// GPU execution succeeded
    GpuSuccess {
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_id: Option<usize>,
    },

    /// CPU fallback (with reason)
    CpuFallback {
        reason: String,
    },

    /// CPU-only (GPU disabled in config)
    CpuDisabled,
}

impl PhaseExecMode {
    pub fn gpu_success(stream_id: Option<usize>) -> Self {
        PhaseExecMode::GpuSuccess { stream_id }
    }

    pub fn cpu_fallback(reason: impl Into<String>) -> Self {
        PhaseExecMode::CpuFallback { reason: reason.into() }
    }

    pub fn cpu_disabled() -> Self {
        PhaseExecMode::CpuDisabled
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, PhaseExecMode::GpuSuccess { .. })
    }
}

impl fmt::Display for PhaseExecMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseExecMode::GpuSuccess { stream_id: Some(id) } => write!(f, "GPU[stream={}]", id),
            PhaseExecMode::GpuSuccess { stream_id: None } => write!(f, "GPU"),
            PhaseExecMode::CpuFallback { reason } => write!(f, "CPU[fallback: {}]", reason),
            PhaseExecMode::CpuDisabled => write!(f, "CPU[disabled]"),
        }
    }
}

/// Single telemetry metric for a phase step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetric {
    /// ISO8601 timestamp
    pub timestamp: String,

    /// Pipeline phase
    pub phase: PhaseName,

    /// Step description (e.g., "temp_5", "replica_swap", "qubo_iteration_100")
    pub step: String,

    /// Current chromatic number
    pub chromatic_number: usize,

    /// Current conflict count
    pub conflicts: usize,

    /// Step duration in milliseconds
    pub duration_ms: f64,

    /// Execution mode (GPU/CPU)
    pub gpu_mode: PhaseExecMode,

    /// Phase-specific parameters (JSON object)
    pub parameters: serde_json::Value,

    /// Optional notes/warnings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl RunMetric {
    /// Create new metric with current timestamp
    pub fn new(
        phase: PhaseName,
        step: impl Into<String>,
        chromatic_number: usize,
        conflicts: usize,
        duration_ms: f64,
        gpu_mode: PhaseExecMode,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            phase,
            step: step.into(),
            chromatic_number,
            conflicts,
            duration_ms,
            gpu_mode,
            parameters: serde_json::Value::Null,
            notes: None,
        }
    }

    /// Add parameters as JSON
    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }

    /// Add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Format for terminal display
    pub fn format_terminal(&self) -> String {
        format!(
            "[{}][{}] {} | colors={} conflicts={} | {:.2}ms",
            self.phase,
            self.gpu_mode,
            self.step,
            self.chromatic_number,
            self.conflicts,
            self.duration_ms
        )
    }
}

/// Summary of entire run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Run identifier
    pub run_id: String,

    /// Graph name
    pub graph_name: String,

    /// Total runtime in seconds
    pub total_runtime_sec: f64,

    /// Final chromatic number
    pub final_chromatic: usize,

    /// Final conflicts
    pub final_conflicts: usize,

    /// Total metrics recorded
    pub metric_count: usize,

    /// Phase breakdown
    pub phase_stats: Vec<PhaseStats>,

    /// GPU usage summary
    pub gpu_summary: GpuUsageSummary,
}

/// Statistics for single phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseStats {
    pub phase: PhaseName,
    pub total_time_ms: f64,
    pub step_count: usize,
    pub gpu_steps: usize,
    pub cpu_steps: usize,
    pub best_chromatic: usize,
}

/// GPU usage summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsageSummary {
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    pub gpu_percentage: f64,
    pub streams_used: Vec<usize>,
    pub stream_mode: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_serialization() {
        let metric = RunMetric::new(
            PhaseName::Thermodynamic,
            "temp_5",
            115,
            0,
            123.45,
            PhaseExecMode::gpu_success(Some(2)),
        )
        .with_parameters(serde_json::json!({"temp": 0.5, "replicas": 56}));

        let json = serde_json::to_string(&metric).expect("Failed to serialize");
        let _deserialized: RunMetric = serde_json::from_str(&json).expect("Failed to deserialize");
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", PhaseName::Thermodynamic), "THERMO");
        assert_eq!(format!("{}", PhaseName::ActiveInference), "AI");
    }
}
