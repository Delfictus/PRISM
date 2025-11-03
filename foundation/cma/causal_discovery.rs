//! Causal Structure Discovery (Stage 2 of CMA)
//!
//! # Purpose
//! Discovers causal manifold structure from thermodynamic ensemble
//! using REAL transfer entropy with KSG estimator and FDR control.
//!
//! # Constitution Reference
//! Phase 6, Task 6.1, Stage 2 - Causal Structure Discovery
//! Phase 6 Implementation Constitution - Sprint 1.2

use ndarray::Array2;
use std::collections::HashMap;

use super::transfer_entropy_ksg::{KSGEstimator, TimeSeries};
use super::transfer_entropy_gpu::GpuKSGEstimator;

/// Causal manifold discovery with false discovery rate control
pub struct CausalManifoldDiscovery {
    fdr_threshold: f64,
    ksg_neighbors: usize,
    min_transfer_entropy: f64,
    /// Real KSG estimator (CPU)
    ksg_estimator: KSGEstimator,
    /// GPU-accelerated KSG (optional)
    gpu_ksg: Option<GpuKSGEstimator>,
}

impl CausalManifoldDiscovery {
    /// Create new causal discovery engine with REAL KSG estimator
    pub fn new(fdr_threshold: f64) -> Self {
        let ksg_neighbors = 4;
        let embed_dim = 3;
        let delay = 1;

        // Initialize real KSG estimator
        let ksg_estimator = KSGEstimator::new(ksg_neighbors, embed_dim, delay);

        // Try to initialize GPU version
        let gpu_ksg = GpuKSGEstimator::new(ksg_neighbors, embed_dim, delay).ok();

        if gpu_ksg.is_some() {
            println!("✓ Causal discovery initialized with GPU-accelerated KSG");
        } else {
            println!("✓ Causal discovery initialized with CPU KSG");
        }

        Self {
            fdr_threshold,
            ksg_neighbors,
            min_transfer_entropy: 0.01,
            ksg_estimator,
            gpu_ksg,
        }
    }

    /// Discover causal manifold from ensemble
    pub fn discover(&self, ensemble: &super::Ensemble) -> super::CausalManifold {
        // Step 1: Compute pairwise transfer entropies
        let te_matrix = self.compute_transfer_entropies(ensemble);

        // Step 2: Apply Benjamini-Hochberg FDR control
        let significant_edges = self.apply_fdr_control(&te_matrix);

        // Step 3: Estimate manifold dimension
        let intrinsic_dim = self.estimate_dimension(&significant_edges);

        // Step 4: Compute metric tensor
        let metric_tensor = self.compute_metric_tensor(&significant_edges, intrinsic_dim);

        super::CausalManifold {
            edges: significant_edges,
            intrinsic_dim,
            metric_tensor,
        }
    }

    fn compute_transfer_entropies(&self, ensemble: &super::Ensemble) -> HashMap<(usize, usize), (f64, f64)> {
        let mut te_matrix = HashMap::new();
        let n_vars = ensemble.solutions[0].data.len();

        // Extract time series from ensemble
        let time_series_data = self.extract_time_series(ensemble);

        // Convert to TimeSeries objects for REAL KSG estimator
        let time_series: Vec<TimeSeries> = time_series_data.into_iter()
            .enumerate()
            .map(|(i, data)| TimeSeries::new(data, format!("var_{}", i)))
            .collect();

        println!("Computing transfer entropies with REAL KSG estimator...");

        // Compute pairwise transfer entropies using REAL implementation
        for i in 0..n_vars {
            for j in 0..n_vars {
                if i != j {
                    // Use GPU if available, otherwise CPU
                    let result = if let Some(ref gpu_ksg) = self.gpu_ksg {
                        gpu_ksg.compute_te_gpu(&time_series[i], &time_series[j]).ok()
                    } else {
                        self.ksg_estimator.compute_te(&time_series[i], &time_series[j]).ok()
                    };

                    if let Some(te_result) = result {
                        if te_result.te_value > self.min_transfer_entropy {
                            te_matrix.insert((i, j), (te_result.te_value, te_result.p_value));

                            if te_result.significant {
                                println!("  Found significant causal edge: {} → {} (TE={:.4}, p={:.4})",
                                         i, j, te_result.te_value, te_result.p_value);
                            }
                        }
                    }
                }
            }
        }

        println!("✓ Computed {} transfer entropy pairs", te_matrix.len());
        te_matrix
    }

    fn ksg_transfer_entropy(&self, source: &[f64], target: &[f64]) -> (f64, f64) {
        // KSG estimator implementation
        // TE_KSG(X→Y) = ψ(k) - ⟨ψ(n_y + 1) + ψ(n_xz + 1) - ψ(n_z + 1)⟩

        let n = source.len();
        if n < 10 {
            return (0.0, 1.0);
        }

        // Build delay embeddings
        let delay = 1;
        let embed_dim = 3;

        let mut te_sum = 0.0;
        let mut valid_points = 0;

        for t in embed_dim..n-1 {
            // Current state: Y(t), Past of Y: Y(t-1:t-embed_dim), Past of X: X(t-1:t-embed_dim)
            let y_curr = target[t];
            let y_past = &target[t-embed_dim..t];
            let x_past = &source[t-embed_dim..t];

            // Find k-nearest neighbors in joint space
            let mut distances = Vec::new();
            for s in embed_dim..n-1 {
                if s != t {
                    let dist = self.compute_distance(
                        y_curr, target[s],
                        y_past, &target[s-embed_dim..s],
                        x_past, &source[s-embed_dim..s]
                    );
                    distances.push(dist);
                }
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if distances.len() > self.ksg_neighbors {
                let epsilon = distances[self.ksg_neighbors - 1];

                // Count neighbors in marginal spaces
                let n_y = self.count_neighbors_marginal(
                    y_curr, y_past,
                    target, embed_dim, epsilon
                );
                let n_xz = self.count_neighbors_joint(
                    y_past, x_past,
                    target, source, embed_dim, epsilon
                );
                let n_z = self.count_neighbors_marginal_past(
                    y_past,
                    target, embed_dim, epsilon
                );

                // Digamma function approximation
                te_sum += self.digamma(self.ksg_neighbors as f64)
                    - self.digamma(n_y as f64 + 1.0)
                    - self.digamma(n_xz as f64 + 1.0)
                    + self.digamma(n_z as f64 + 1.0);
                valid_points += 1;
            }
        }

        let te_value = if valid_points > 0 {
            te_sum / valid_points as f64
        } else {
            0.0
        };

        // Bootstrap for p-value
        let p_value = self.bootstrap_p_value(source, target, te_value);

        (te_value.max(0.0), p_value)
    }

    fn apply_fdr_control(&self, te_matrix: &HashMap<(usize, usize), (f64, f64)>) -> Vec<super::CausalEdge> {
        // Benjamini-Hochberg procedure
        let mut p_values: Vec<_> = te_matrix.iter()
            .map(|(&(i, j), &(te, p))| (i, j, te, p))
            .collect();

        // Sort by p-value
        p_values.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

        // Find threshold
        let m = p_values.len();
        let mut threshold_idx = 0;

        for (i, &(_, _, _, p)) in p_values.iter().enumerate() {
            let adjusted_threshold = self.fdr_threshold * (i + 1) as f64 / m as f64;
            if p <= adjusted_threshold {
                threshold_idx = i;
            }
        }

        // Create edges for significant connections
        p_values.iter()
            .take(threshold_idx + 1)
            .map(|&(source, target, te, p)| super::CausalEdge {
                source,
                target,
                transfer_entropy: te,
                p_value: p,
            })
            .collect()
    }

    fn estimate_dimension(&self, edges: &[super::CausalEdge]) -> usize {
        // Use local dimension estimation based on causal graph topology
        if edges.is_empty() {
            return 2; // Minimum dimension
        }

        // Count unique nodes
        let mut nodes = std::collections::HashSet::new();
        for edge in edges {
            nodes.insert(edge.source);
            nodes.insert(edge.target);
        }

        // Estimate based on connectivity
        let avg_degree = (edges.len() * 2) as f64 / nodes.len() as f64;
        let estimated_dim = (avg_degree.log2() + 1.0).ceil() as usize;

        estimated_dim.max(2).min(nodes.len())
    }

    fn compute_metric_tensor(&self, edges: &[super::CausalEdge], dim: usize) -> Array2<f64> {
        // Initialize metric as identity
        let mut metric: Array2<f64> = Array2::eye(dim);

        // Weight by causal strengths
        for edge in edges {
            let i = edge.source % dim;
            let j = edge.target % dim;

            // Off-diagonal elements weighted by transfer entropy
            if i != j {
                metric[[i, j]] += edge.transfer_entropy * 0.1;
                metric[[j, i]] += edge.transfer_entropy * 0.1;
            }
        }

        // Ensure positive definiteness
        for i in 0..dim {
            metric[[i, i]] = metric.row(i).sum().abs() + 1.0;
        }

        metric
    }

    // Helper functions
    fn extract_time_series(&self, ensemble: &super::Ensemble) -> Vec<Vec<f64>> {
        let n_vars = ensemble.solutions[0].data.len();
        let mut series = vec![Vec::new(); n_vars];

        for solution in &ensemble.solutions {
            for (i, &val) in solution.data.iter().enumerate() {
                series[i].push(val);
            }
        }

        series
    }

    fn compute_distance(
        &self,
        y_curr1: f64, y_curr2: f64,
        y_past1: &[f64], y_past2: &[f64],
        x_past1: &[f64], x_past2: &[f64]
    ) -> f64 {
        let mut dist = (y_curr1 - y_curr2).powi(2);

        for i in 0..y_past1.len() {
            dist += (y_past1[i] - y_past2[i]).powi(2);
        }

        for i in 0..x_past1.len() {
            dist += (x_past1[i] - x_past2[i]).powi(2);
        }

        dist.sqrt()
    }

    fn count_neighbors_marginal(
        &self,
        y_curr: f64, y_past: &[f64],
        target: &[f64], embed_dim: usize, epsilon: f64
    ) -> usize {
        let mut count = 0;
        let n = target.len();

        for t in embed_dim..n-1 {
            let dist_curr = (target[t] - y_curr).abs();
            let mut dist_past = 0.0;

            for i in 0..y_past.len() {
                dist_past += (target[t-embed_dim+i] - y_past[i]).powi(2);
            }

            if dist_curr + dist_past.sqrt() < epsilon {
                count += 1;
            }
        }

        count
    }

    fn count_neighbors_joint(
        &self,
        y_past: &[f64], x_past: &[f64],
        target: &[f64], source: &[f64],
        embed_dim: usize, epsilon: f64
    ) -> usize {
        let mut count = 0;
        let n = target.len();

        for t in embed_dim..n-1 {
            let mut dist = 0.0;

            for i in 0..y_past.len() {
                dist += (target[t-embed_dim+i] - y_past[i]).powi(2);
                dist += (source[t-embed_dim+i] - x_past[i]).powi(2);
            }

            if dist.sqrt() < epsilon {
                count += 1;
            }
        }

        count
    }

    fn count_neighbors_marginal_past(
        &self,
        y_past: &[f64],
        target: &[f64], embed_dim: usize, epsilon: f64
    ) -> usize {
        let mut count = 0;
        let n = target.len();

        for t in embed_dim..n-1 {
            let mut dist = 0.0;

            for i in 0..y_past.len() {
                dist += (target[t-embed_dim+i] - y_past[i]).powi(2);
            }

            if dist.sqrt() < epsilon {
                count += 1;
            }
        }

        count
    }

    fn digamma(&self, x: f64) -> f64 {
        // Asymptotic approximation for digamma function
        if x < 1.0 {
            -1.0 / x - 0.5772156649 // Euler-Mascheroni constant
        } else {
            x.ln() - 0.5 / x - 1.0 / (12.0 * x * x)
        }
    }

    fn bootstrap_p_value(&self, source: &[f64], target: &[f64], observed_te: f64) -> f64 {
        let n_bootstrap = 100;
        let mut greater_count = 0;

        for _ in 0..n_bootstrap {
            // Shuffle source to break causal relationship
            let mut shuffled = source.to_vec();
            fastrand::shuffle(&mut shuffled);

            let (surrogate_te, _) = self.ksg_transfer_entropy(&shuffled, target);

            if surrogate_te >= observed_te {
                greater_count += 1;
            }
        }

        (greater_count as f64 + 1.0) / (n_bootstrap as f64 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_discovery_creation() {
        let discoverer = CausalManifoldDiscovery::new(0.05);
        assert_eq!(discoverer.fdr_threshold, 0.05);
        assert_eq!(discoverer.ksg_neighbors, 4);
    }

    #[test]
    fn test_dimension_estimation() {
        let discoverer = CausalManifoldDiscovery::new(0.05);

        let edges = vec![
            super::super::CausalEdge {
                source: 0,
                target: 1,
                transfer_entropy: 0.5,
                p_value: 0.01,
            },
            super::super::CausalEdge {
                source: 1,
                target: 2,
                transfer_entropy: 0.3,
                p_value: 0.02,
            },
        ];

        let dim = discoverer.estimate_dimension(&edges);
        assert!(dim >= 2);
        assert!(dim <= 3);
    }

    #[test]
    fn test_metric_tensor_positive_definite() {
        let discoverer = CausalManifoldDiscovery::new(0.05);

        let edges = vec![
            super::super::CausalEdge {
                source: 0,
                target: 1,
                transfer_entropy: 0.5,
                p_value: 0.01,
            },
        ];

        let metric = discoverer.compute_metric_tensor(&edges, 3);

        // Check diagonal dominance (sufficient for positive definiteness)
        for i in 0..3 {
            let diag = metric[[i, i]];
            let off_diag_sum: f64 = (0..3)
                .filter(|&j| j != i)
                .map(|j| metric[[i, j]].abs())
                .sum();
            assert!(diag > off_diag_sum);
        }
    }
}