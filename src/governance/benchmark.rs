use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Manifest describing benchmark artifacts and performance thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub version: String,
    pub generated: String,
    pub artifacts: HashMap<String, BenchmarkArtifact>,
    pub performance_thresholds: PerformanceThresholds,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkArtifact {
    pub path: String,
    pub sha256: String,
    pub vertices: usize,
    pub edges: usize,
    pub expected_colors: ExpectedColors,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedColors {
    pub world_record: u32,
    pub baseline: u32,
    pub target: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub regression_tolerance: f64,
    pub improvement_target: f64,
    pub variance_limit: f64,
    pub timeout_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub min_runs: usize,
    pub percentile: usize,
    pub warmup_runs: usize,
    pub cooldown_ms: u64,
}

impl BenchmarkManifest {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read_to_string(&path)?;
        let manifest: BenchmarkManifest = serde_json::from_str(&data)?;
        manifest.verify_checksums()?;
        Ok(manifest)
    }

    pub fn verify_checksums(&self) -> Result<()> {
        for (name, artifact) in &self.artifacts {
            let file_path = Path::new(&artifact.path);
            let data = fs::read(file_path)
                .map_err(|e| anyhow!("Failed to read {} ({}): {}", name, artifact.path, e))?;
            let mut hasher = Sha256::new();
            hasher.update(data);
            let digest = hex::encode(hasher.finalize());
            if digest != artifact.sha256 {
                bail!(
                    "Checksum mismatch for {}: expected {}, got {}",
                    name,
                    artifact.sha256,
                    digest
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub graph_name: String,
    pub colors_achieved: u32,
    pub time_ms: f64,
    pub memory_peak_mb: f64,
    pub determinism_hash: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub colors: u32,
    pub time_ms: f64,
    pub memory_mb: f64,
    pub improvement: i32,
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    Blocker,
    Critical,
    Warning,
}

#[derive(Debug, Clone)]
pub enum GateDecision {
    Pass { metrics: PerformanceMetrics },
    Fail { reason: String, severity: Severity },
}

/// Evaluates benchmark results against manifest thresholds.
pub struct PerformanceGate {
    manifest: BenchmarkManifest,
    results: Vec<BenchmarkResult>,
    results_path: Option<PathBuf>,
}

impl PerformanceGate {
    pub fn load(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest = BenchmarkManifest::load_from_path(manifest_path)?;
        Ok(Self {
            manifest,
            results: Vec::new(),
            results_path: None,
        })
    }

    pub fn with_results_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.results_path = Some(path.into());
        self
    }

    pub fn manifest(&self) -> &BenchmarkManifest {
        &self.manifest
    }

    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    pub fn validate_result(&mut self, result: BenchmarkResult) -> Result<GateDecision> {
        let artifact = self
            .manifest
            .artifacts
            .get(&result.graph_name)
            .ok_or_else(|| anyhow!("Unknown benchmark: {}", result.graph_name))?;

        if result.colors_achieved > artifact.expected_colors.baseline {
            return Ok(GateDecision::Fail {
                reason: format!(
                    "Color regression: {} > {} (baseline)",
                    result.colors_achieved, artifact.expected_colors.baseline
                ),
                severity: Severity::Blocker,
            });
        }

        let timeout_limit =
            artifact.timeout_ms as f64 * self.manifest.performance_thresholds.timeout_multiplier;
        if result.time_ms > timeout_limit {
            return Ok(GateDecision::Fail {
                reason: format!(
                    "Timeout exceeded: {:.1}ms > {:.1}ms",
                    result.time_ms, timeout_limit
                ),
                severity: Severity::Critical,
            });
        }

        if let Some(baseline) = self.median_for(&result.graph_name) {
            let regression = (result.time_ms - baseline) / baseline;
            if regression > self.manifest.performance_thresholds.regression_tolerance {
                return Ok(GateDecision::Fail {
                    reason: format!("Performance regression {:.1}%", regression * 100.0),
                    severity: Severity::Critical,
                });
            }
        }

        self.results.push(result.clone());
        let improvement = artifact.expected_colors.baseline as i32 - result.colors_achieved as i32;

        Ok(GateDecision::Pass {
            metrics: PerformanceMetrics {
                colors: result.colors_achieved,
                time_ms: result.time_ms,
                memory_mb: result.memory_peak_mb,
                improvement,
            },
        })
    }

    pub fn persist_results(&self) -> Result<()> {
        if let Some(path) = &self.results_path {
            let json = serde_json::to_string_pretty(&self.results)?;
            fs::write(path, json)?;
        }
        Ok(())
    }

    fn median_for(&self, graph_name: &str) -> Option<f64> {
        let mut times: Vec<f64> = self
            .results
            .iter()
            .filter(|r| r.graph_name == graph_name)
            .map(|r| r.time_ms)
            .collect();
        if times.len() < self.manifest.validation.min_runs {
            return None;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = times.len() / 2;
        if times.len() % 2 == 0 {
            Some((times[mid - 1] + times[mid]) / 2.0)
        } else {
            Some(times[mid])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_manifest(tmp: &tempfile::TempDir) -> PathBuf {
        let manifest_path = tmp.path().join("bench_manifest.json");
        let mut data = serde_json::json!({
            "version": "1.0.0",
            "generated": "2025-01-19T00:00:00Z",
            "artifacts": {
                "sample": {
                    "path": tmp.path().join("sample.col").to_string_lossy(),
                    "sha256": "",
                    "vertices": 4,
                    "edges": 6,
                    "expected_colors": {
                        "world_record": 4,
                        "baseline": 4,
                        "target": 4
                    },
                    "timeout_ms": 1000
                }
            },
            "performance_thresholds": {
                "regression_tolerance": 0.1,
                "improvement_target": 0.05,
                "variance_limit": 0.08,
                "timeout_multiplier": 1.5
            },
            "validation": {
                "min_runs": 1,
                "percentile": 50,
                "warmup_runs": 0,
                "cooldown_ms": 0
            }
        });

        // compute hash for dummy artifact
        let artifact_path = tmp.path().join("sample.col");
        fs::write(&artifact_path, b"1234").unwrap();
        let digest = hex::encode(Sha256::digest(b"1234"));
        if let Some(sample) = data
            .get_mut("artifacts")
            .and_then(|artifacts| artifacts.get_mut("sample"))
        {
            if let Some(obj) = sample.as_object_mut() {
                obj.insert("sha256".to_string(), serde_json::Value::String(digest));
            }
        }

        let json_str = serde_json::to_string_pretty(&data).unwrap();
        fs::write(&manifest_path, json_str).unwrap();
        manifest_path
    }

    #[test]
    fn validate_result_passes_when_within_threshold() {
        let tmp = tempfile::TempDir::new().unwrap();
        let manifest_path = fixture_manifest(&tmp);
        let mut gate = PerformanceGate::load(&manifest_path).unwrap();

        let decision = gate
            .validate_result(BenchmarkResult {
                graph_name: "sample".into(),
                colors_achieved: 4,
                time_ms: 500.0,
                memory_peak_mb: 128.0,
                determinism_hash: "abc".into(),
            })
            .unwrap();

        matches!(decision, GateDecision::Pass { .. });
    }
}
