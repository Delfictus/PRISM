use crate::cuda::device_guard::DeviceCapabilities;
use crate::telemetry::TelemetryLogger;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub enum PathDecision {
    Dense {
        use_tensor_cores: bool,
        estimated_speedup: f32,
    },
    Sparse {
        reason: String,
        fallback: SparseFallback,
    },
}

#[derive(Debug, Clone, Serialize)]
pub enum SparseFallback {
    Csr,
    Coo,
    EdgeList,
}

#[derive(Debug, Serialize)]
pub enum TelemetryEvent {
    PreFlightCheck {
        graph_size: usize,
        edges: usize,
        required_memory_mb: usize,
        available_memory_mb: usize,
        tensor_cores_available: bool,
    },
    PathDecision {
        decision: PathDecision,
        check_duration_us: u64,
    },
}

pub struct DensePathGuard {
    caps: &'static DeviceCapabilities,
    telemetry: TelemetryLogger,
}

impl DensePathGuard {
    pub fn new() -> Self {
        Self {
            caps: DeviceCapabilities::get_cached(),
            telemetry: TelemetryLogger::new("dense_path_guard"),
        }
    }

    pub fn check_feasibility(&self, n: usize, edges: usize) -> PathDecision {
        let start = Instant::now();
        let adjacency_bytes = n * n * 2; // FP16 approximation
        let workspace_bytes = n * 64 * 4; // heuristic workspace estimate
        let required_mb = (adjacency_bytes + workspace_bytes) / (1024 * 1024);

        self.telemetry.log(TelemetryEvent::PreFlightCheck {
            graph_size: n,
            edges,
            required_memory_mb: required_mb,
            available_memory_mb: self.caps.available_memory_mb,
            tensor_cores_available: self.caps.tensor_cores,
        });

        let decision = if !self.caps.fp16_support {
            PathDecision::Sparse {
                reason: "Device lacks FP16 support".into(),
                fallback: SparseFallback::Csr,
            }
        } else if required_mb > self.caps.available_memory_mb {
            PathDecision::Sparse {
                reason: format!(
                    "Insufficient memory: need {}MB, have {}MB",
                    required_mb, self.caps.available_memory_mb
                ),
                fallback: SparseFallback::Csr,
            }
        } else {
            let density = if n > 0 {
                edges as f32 / (n * n) as f32
            } else {
                0.0
            };
            if density < 0.1 {
                PathDecision::Sparse {
                    reason: format!("Graph too sparse ({:.2}% density)", density * 100.0),
                    fallback: SparseFallback::Csr,
                }
            } else {
                PathDecision::Dense {
                    use_tensor_cores: self.caps.tensor_cores,
                    estimated_speedup: self.estimate_speedup(density),
                }
            }
        };

        self.telemetry.log(TelemetryEvent::PathDecision {
            decision: decision.clone(),
            check_duration_us: start.elapsed().as_micros() as u64,
        });

        decision
    }

    fn estimate_speedup(&self, density: f32) -> f32 {
        let bandwidth_factor = self.caps.estimate_bandwidth_gbps() / 500.0;
        let tensor_boost = if self.caps.tensor_cores { 8.0 } else { 1.0 };
        (density * tensor_boost * bandwidth_factor).max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_when_density_low() {
        let guard = DensePathGuard::new();
        let decision = guard.check_feasibility(100, 50); // density < 0.1
        match decision {
            PathDecision::Sparse { .. } => {}
            _ => panic!("expected sparse decision"),
        }
    }
}
