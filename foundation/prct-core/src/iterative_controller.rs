//! Iterative Pipeline Controller
//!
//! Runs multiple pipeline passes with warm starts to iteratively
//! improve chromatic number until convergence.

use crate::errors::*;
use crate::world_record_pipeline::{WorldRecordPipeline, WorldRecordConfig};
use crate::telemetry::{TelemetryHandle, PhaseName, PhaseExecMode, RunMetric};
use shared_types::{ColoringSolution, Graph};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for iterative refinement
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IterativeConfig {
    /// Maximum number of passes
    #[serde(default = "default_max_passes")]
    pub max_passes: usize,

    /// Minimum improvement required to continue (colors)
    #[serde(default = "default_min_delta")]
    pub min_delta: usize,

    /// Enable warm starts (use previous best as initial)
    #[serde(default = "default_true")]
    pub enable_warm_start: bool,

    /// Use telemetry for monitoring
    #[serde(default = "default_true")]
    pub enable_telemetry: bool,
}

fn default_max_passes() -> usize { 3 }
fn default_min_delta() -> usize { 1 }
fn default_true() -> bool { true }

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            max_passes: default_max_passes(),
            min_delta: default_min_delta(),
            enable_warm_start: true,
            enable_telemetry: true,
        }
    }
}

/// Run iterative pipeline with convergence detection
pub fn run_iterative_pipeline(
    graph: &Graph,
    config: &WorldRecordConfig,
    iterative_config: &IterativeConfig,
    telemetry: Option<Arc<TelemetryHandle>>,
) -> Result<ColoringSolution> {
    let mut best_solution: Option<ColoringSolution> = None;
    let mut pass = 0;
    let total_start = Instant::now();

    while pass < iterative_config.max_passes {
        let pass_start = Instant::now();

        eprintln!("\n=== Iterative Pass {}/{} ===", pass + 1, iterative_config.max_passes);

        // Create pipeline for this pass
        let mut pipeline = WorldRecordPipeline::new(graph.clone(), config.clone())?;

        // If warm start enabled, use previous best as initial solution
        // (This would require modifying WorldRecordPipeline to accept initial solution)
        // For now, each pass runs from scratch

        // Run pipeline pass
        let solution = pipeline.run()?;

        let pass_duration = pass_start.elapsed().as_secs_f64();

        // Record in telemetry
        if let Some(ref t) = telemetry {
            let metric = RunMetric::new(
                PhaseName::Ensemble,
                format!("iterative_pass_{}", pass + 1),
                solution.chromatic_number,
                solution.conflicts,
                pass_duration * 1000.0,
                PhaseExecMode::cpu_disabled(),
            )
            .with_parameters(serde_json::json!({
                "pass": pass + 1,
                "max_passes": iterative_config.max_passes,
            }))
            .with_notes(format!(
                "Pass {} complete: {} colors, {} conflicts",
                pass + 1,
                solution.chromatic_number,
                solution.conflicts
            ));

            t.record(metric);
        }

        // Check improvement
        let improvement = if let Some(ref prev_best) = best_solution {
            let delta = prev_best.chromatic_number as i32 - solution.chromatic_number as i32;

            eprintln!(
                "Pass {}: {} colors (prev: {}, delta: {})",
                pass + 1,
                solution.chromatic_number,
                prev_best.chromatic_number,
                delta
            );

            if delta < iterative_config.min_delta as i32 {
                eprintln!("Convergence detected: improvement {} < min_delta {}", delta, iterative_config.min_delta);
                break;
            }

            delta
        } else {
            eprintln!("Pass {}: {} colors (initial)", pass + 1, solution.chromatic_number);
            0
        };

        // Update best
        if let Some(ref best) = best_solution {
            if solution.chromatic_number < best.chromatic_number {
                best_solution = Some(solution);
            }
        } else {
            best_solution = Some(solution);
        }

        pass += 1;
    }

    let total_duration = total_start.elapsed().as_secs_f64();

    let final_solution = best_solution.ok_or_else(|| {
        PRCTError::ColoringFailed("No solution found in iterative pipeline".to_string())
    })?;

    eprintln!(
        "\n=== Iterative Pipeline Complete ===\nTotal passes: {}\nTotal time: {:.2}s\nFinal chromatic: {}\nFinal conflicts: {}",
        pass,
        total_duration,
        final_solution.chromatic_number,
        final_solution.conflicts
    );

    Ok(final_solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterative_config_defaults() {
        let config = IterativeConfig::default();
        assert_eq!(config.max_passes, 3);
        assert_eq!(config.min_delta, 1);
        assert!(config.enable_warm_start);
    }
}
