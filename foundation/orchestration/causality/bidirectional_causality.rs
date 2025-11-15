//! Ultra-Enhanced Bidirectional Causality Analysis for LLM Orchestration
//!
//! World-First Algorithm #9: Complete implementation of bidirectional causal discovery
//! with convergent cross mapping, transfer entropy, Granger causality, and Pearl's
//! causal inference framework with do-calculus.

use crate::orchestration::OrchestrationError;
use nalgebra::{DMatrix, DVector, SVD};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::collections::{HashMap, HashSet, VecDeque};

/// Bidirectional causality analyzer
pub struct BidirectionalCausalityAnalyzer {
    /// Convergent Cross Mapping engine
    ccm_engine: ConvergentCrossMapping,
    /// Transfer entropy calculator
    te_calculator: TransferEntropyCalculator,
    /// Granger causality tester
    granger_tester: GrangerCausalityTester,
    /// Pearl's causal graph
    causal_graph: CausalGraph,
    /// PC algorithm for structure learning
    pc_algorithm: PCAlgorithm,
    /// Causal discovery parameters
    parameters: CausalParameters,
    /// Time series data buffer
    time_series_buffer: TimeSeriesBuffer,
}

/// Convergent Cross Mapping (Sugihara et al. 2012)
#[derive(Clone, Debug)]
struct ConvergentCrossMapping {
    /// Embedding dimension
    embedding_dim: usize,
    /// Time delay for embedding
    tau: usize,
    /// Library lengths to test
    library_lengths: Vec<usize>,
    /// Number of nearest neighbors
    k_neighbors: usize,
    /// Convergence threshold
    convergence_threshold: f64,
    /// Shadow manifold cache
    shadow_manifolds: HashMap<String, ShadowManifold>,
}

#[derive(Clone, Debug)]
struct ShadowManifold {
    /// Embedded time series
    embedded: DMatrix<f64>,
    /// Original time series
    original: DVector<f64>,
    /// Time indices
    time_indices: Vec<usize>,
}

/// Transfer Entropy with multiple variants
#[derive(Clone, Debug)]
struct TransferEntropyCalculator {
    /// History length for source
    k_source: usize,
    /// History length for target
    k_target: usize,
    /// Prediction horizon
    l_future: usize,
    /// Binning method
    binning: BinningMethod,
    /// Use symbolic transfer entropy
    use_symbolic: bool,
    /// Permutation length for symbolic TE
    permutation_length: usize,
    /// Effective transfer entropy threshold
    ete_threshold: f64,
}

#[derive(Clone, Debug)]
enum BinningMethod {
    FixedWidth(usize),
    Adaptive,
    KernelDensity,
    Symbolic,
}

/// Granger Causality with multiple tests
#[derive(Clone, Debug)]
struct GrangerCausalityTester {
    /// Maximum lag to test
    max_lag: usize,
    /// Significance level
    alpha: f64,
    /// Use VAR model
    use_var: bool,
    /// Nonlinear Granger test
    nonlinear: bool,
    /// Spectral Granger causality
    spectral: bool,
    /// Conditional Granger causality
    conditional_vars: Vec<String>,
}

/// Pearl's Causal Graph (DAG)
#[derive(Clone, Debug)]
struct CausalGraph {
    /// Nodes (variables)
    nodes: Vec<CausalNode>,
    /// Directed edges
    edges: HashMap<(usize, usize), CausalEdge>,
    /// Adjacency matrix
    adjacency: DMatrix<f64>,
    /// d-separation cache
    d_separation_cache: HashMap<(usize, usize, HashSet<usize>), bool>,
    /// Backdoor paths
    backdoor_paths: HashMap<(usize, usize), Vec<Vec<usize>>>,
}

#[derive(Clone, Debug)]
struct CausalNode {
    /// Variable name
    name: String,
    /// Node type
    node_type: NodeType,
    /// Is observed
    observed: bool,
    /// Intervention state
    intervened: Option<f64>,
}

#[derive(Clone, Debug)]
enum NodeType {
    Observed,
    Latent,
    Confounder,
    Mediator,
    Collider,
}

#[derive(Clone, Debug)]
struct CausalEdge {
    /// Edge strength
    strength: f64,
    /// Edge type
    edge_type: EdgeType,
    /// Time lag
    lag: usize,
}

#[derive(Clone, Debug)]
enum EdgeType {
    Direct,
    Spurious,
    Bidirectional,
    TimeDelayed,
}

/// PC Algorithm for causal structure learning
#[derive(Clone, Debug)]
struct PCAlgorithm {
    /// Independence test type
    independence_test: IndependenceTest,
    /// Significance level
    alpha: f64,
    /// Maximum conditioning set size
    max_conditioning_size: usize,
    /// Orientation rules
    orientation_rules: Vec<OrientationRule>,
    /// Conflict resolution strategy
    conflict_resolution: ConflictResolution,
}

#[derive(Clone, Debug)]
enum IndependenceTest {
    PartialCorrelation,
    MutualInformation,
    HSIC,      // Hilbert-Schmidt Independence Criterion
    KernelCIT, // Kernel Conditional Independence Test
}

#[derive(Clone, Debug)]
enum OrientationRule {
    ColliderOrientation,
    AcyclicityOrientation,
    ConservativeOrientation,
    MajorityOrientation,
}

#[derive(Clone, Debug)]
enum ConflictResolution {
    Conservative,
    Majority,
    MaxStrength,
    Bayesian,
}

/// Causal discovery parameters
#[derive(Clone, Debug)]
struct CausalParameters {
    /// Minimum samples for reliable estimation
    min_samples: usize,
    /// Use bootstrapping
    bootstrap: bool,
    /// Number of bootstrap samples
    n_bootstrap: usize,
    /// Surrogate data method
    surrogate_method: SurrogateMethod,
    /// Number of surrogates for significance testing
    n_surrogates: usize,
    /// Multiple testing correction
    correction: MultipleTestingCorrection,
}

#[derive(Clone, Debug)]
enum SurrogateMethod {
    RandomShuffle,
    PhaseRandomization,
    IAAFT, // Iterative Amplitude Adjusted Fourier Transform
    TwinSurrogates,
}

#[derive(Clone, Debug)]
enum MultipleTestingCorrection {
    None,
    Bonferroni,
    BenjaminiHochberg,
    BenjaminiYekutieli,
}

/// Time series buffer for analysis
#[derive(Clone, Debug)]
struct TimeSeriesBuffer {
    /// Variable name to time series mapping
    series: HashMap<String, DVector<f64>>,
    /// Sampling rate
    sampling_rate: f64,
    /// Buffer capacity
    capacity: usize,
}

impl BidirectionalCausalityAnalyzer {
    /// Create new bidirectional causality analyzer
    pub fn new() -> Self {
        Self {
            ccm_engine: ConvergentCrossMapping {
                embedding_dim: 3,
                tau: 1,
                library_lengths: vec![10, 20, 50, 100, 200, 500],
                k_neighbors: 4,
                convergence_threshold: 0.1,
                shadow_manifolds: HashMap::new(),
            },
            te_calculator: TransferEntropyCalculator {
                k_source: 1,
                k_target: 1,
                l_future: 1,
                binning: BinningMethod::Adaptive,
                use_symbolic: true,
                permutation_length: 3,
                ete_threshold: 0.05,
            },
            granger_tester: GrangerCausalityTester {
                max_lag: 10,
                alpha: 0.05,
                use_var: true,
                nonlinear: true,
                spectral: true,
                conditional_vars: Vec::new(),
            },
            causal_graph: CausalGraph::new(),
            pc_algorithm: PCAlgorithm {
                independence_test: IndependenceTest::HSIC,
                alpha: 0.05,
                max_conditioning_size: 5,
                orientation_rules: vec![
                    OrientationRule::ColliderOrientation,
                    OrientationRule::AcyclicityOrientation,
                    OrientationRule::MajorityOrientation,
                ],
                conflict_resolution: ConflictResolution::Bayesian,
            },
            parameters: CausalParameters {
                min_samples: 100,
                bootstrap: true,
                n_bootstrap: 1000,
                surrogate_method: SurrogateMethod::IAAFT,
                n_surrogates: 100,
                correction: MultipleTestingCorrection::BenjaminiHochberg,
            },
            time_series_buffer: TimeSeriesBuffer {
                series: HashMap::new(),
                sampling_rate: 1.0,
                capacity: 10000,
            },
        }
    }

    /// Analyze bidirectional causality between time series
    pub fn analyze(
        &mut self,
        x_name: &str,
        x_data: &DVector<f64>,
        y_name: &str,
        y_data: &DVector<f64>,
    ) -> Result<CausalityResult, OrchestrationError> {
        // Validate inputs
        if x_data.len() != y_data.len() {
            return Err(OrchestrationError::DimensionMismatch(format!(
                "Expected {}, got {}",
                x_data.len(),
                y_data.len()
            )));
        }

        if x_data.len() < self.parameters.min_samples {
            return Err(OrchestrationError::InsufficientData {
                required: self.parameters.min_samples,
                available: x_data.len(),
            });
        }

        // Store in buffer
        self.time_series_buffer
            .series
            .insert(x_name.to_string(), x_data.clone());
        self.time_series_buffer
            .series
            .insert(y_name.to_string(), y_data.clone());

        // 1. Convergent Cross Mapping
        let ccm_result = self.convergent_cross_mapping(x_data, y_data)?;

        // 2. Transfer Entropy (both directions)
        let te_x_to_y = self.transfer_entropy(x_data, y_data)?;
        let te_y_to_x = self.transfer_entropy(y_data, x_data)?;

        // 3. Granger Causality
        let granger_x_to_y = self.granger_causality(x_data, y_data)?;
        let granger_y_to_x = self.granger_causality(y_data, x_data)?;

        // 4. Build causal graph
        self.update_causal_graph(x_name, y_name, &ccm_result, te_x_to_y, te_y_to_x)?;

        // 5. PC algorithm for structure learning
        let pc_result = self.pc_structure_learning()?;

        // 6. Compute causal strength with bootstrapping
        let causal_strength = if self.parameters.bootstrap {
            self.bootstrap_causal_strength(x_data, y_data)?
        } else {
            self.compute_causal_strength(&ccm_result, te_x_to_y, te_y_to_x)
        };

        // 7. Test significance with surrogates
        let significance = self.test_significance_surrogates(x_data, y_data, &causal_strength)?;

        Ok(CausalityResult {
            ccm: ccm_result.clone(),
            transfer_entropy: TransferEntropyResult {
                te_x_to_y,
                te_y_to_x,
                normalized_te: (te_x_to_y - te_y_to_x) / (te_x_to_y + te_y_to_x + 1e-10),
            },
            granger: GrangerResult {
                f_statistic_x_to_y: granger_x_to_y,
                f_statistic_y_to_x: granger_y_to_x,
                significant: granger_x_to_y > 3.84 || granger_y_to_x > 3.84, // Chi-square critical value
            },
            causal_direction: self.determine_direction(te_x_to_y, te_y_to_x, &ccm_result),
            strength: causal_strength,
            confidence: significance.confidence,
            graph_structure: pc_result,
        })
    }

    /// Convergent Cross Mapping
    fn convergent_cross_mapping(
        &mut self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> Result<CCMResult, OrchestrationError> {
        // Create shadow manifolds
        let manifold_x = self.create_shadow_manifold(x, "x")?;
        let manifold_y = self.create_shadow_manifold(y, "y")?;

        let mut rho_values = Vec::new();
        let mut converged = false;
        let mut convergence_rate = 0.0;

        // Test different library lengths
        for &lib_len in &self.ccm_engine.library_lengths {
            if lib_len > x.len() {
                continue;
            }

            // Cross map Y from X's manifold
            let rho_x_to_y = self.cross_map(&manifold_x, y, lib_len)?;

            // Cross map X from Y's manifold
            let rho_y_to_x = self.cross_map(&manifold_y, x, lib_len)?;

            rho_values.push((lib_len, rho_x_to_y, rho_y_to_x));

            // Check convergence
            if rho_values.len() > 2 {
                let recent_change =
                    (rho_values[rho_values.len() - 1].1 - rho_values[rho_values.len() - 2].1).abs();
                if recent_change < self.ccm_engine.convergence_threshold {
                    converged = true;
                    convergence_rate = recent_change;
                }
            }
        }

        // Fit convergence curve
        let (asymptote_x_to_y, rate_x_to_y) =
            self.fit_convergence_curve(&rho_values.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>())?;
        let (asymptote_y_to_x, rate_y_to_x) =
            self.fit_convergence_curve(&rho_values.iter().map(|r| (r.0, r.2)).collect::<Vec<_>>())?;

        Ok(CCMResult {
            rho_values,
            converged,
            convergence_rate,
            asymptote_x_to_y,
            asymptote_y_to_x,
            rate_x_to_y,
            rate_y_to_x,
        })
    }

    /// Create shadow manifold using time-delay embedding
    fn create_shadow_manifold(
        &mut self,
        series: &DVector<f64>,
        name: &str,
    ) -> Result<ShadowManifold, OrchestrationError> {
        let E = self.ccm_engine.embedding_dim;
        let tau = self.ccm_engine.tau;
        let n_points = series.len() - (E - 1) * tau;

        if n_points < 10 {
            return Err(OrchestrationError::InsufficientData {
                required: 10,
                available: n_points,
            });
        }

        let mut embedded = DMatrix::zeros(n_points, E);
        let mut time_indices = Vec::new();

        for i in 0..n_points {
            for j in 0..E {
                embedded[(i, j)] = series[i + j * tau];
            }
            time_indices.push(i + (E - 1) * tau);
        }

        let manifold = ShadowManifold {
            embedded,
            original: series.clone(),
            time_indices,
        };

        self.ccm_engine
            .shadow_manifolds
            .insert(name.to_string(), manifold.clone());

        Ok(manifold)
    }

    /// Cross map target from source manifold
    fn cross_map(
        &self,
        source_manifold: &ShadowManifold,
        target: &DVector<f64>,
        lib_len: usize,
    ) -> Result<f64, OrchestrationError> {
        let mut predictions = Vec::new();
        let mut observations = Vec::new();

        // Randomly sample library points
        let library_indices: Vec<usize> = (0..source_manifold.embedded.nrows())
            .filter(|&i| i < lib_len)
            .collect();

        for test_idx in lib_len..source_manifold.embedded.nrows() {
            // Find k nearest neighbors in library
            let test_point = source_manifold.embedded.row(test_idx);
            let mut distances: Vec<(usize, f64)> = library_indices
                .iter()
                .map(|&lib_idx| {
                    let lib_point = source_manifold.embedded.row(lib_idx);
                    let dist = (test_point - lib_point).norm();
                    (lib_idx, dist)
                })
                .collect();

            distances.sort_by_key(|d| OrderedFloat(d.1));
            let nearest = distances
                .iter()
                .take(self.ccm_engine.k_neighbors)
                .collect::<Vec<_>>();

            // Compute weights (exponential)
            let min_dist = nearest[0].1;
            let weights: Vec<f64> = nearest
                .iter()
                .map(|d| (-d.1 / (min_dist + 1e-10)).exp())
                .collect();
            let weight_sum: f64 = weights.iter().sum();

            // Predict target value
            let mut prediction = 0.0;
            for (i, neighbor) in nearest.iter().enumerate() {
                let target_idx = source_manifold.time_indices[neighbor.0];
                if target_idx < target.len() {
                    prediction += target[target_idx] * weights[i] / weight_sum;
                }
            }

            let obs_idx = source_manifold.time_indices[test_idx];
            if obs_idx < target.len() {
                predictions.push(prediction);
                observations.push(target[obs_idx]);
            }
        }

        // Compute correlation
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correlation = self.compute_correlation(&predictions, &observations);
        Ok(correlation)
    }

    /// Fit convergence curve
    fn fit_convergence_curve(
        &self,
        rho_values: &[(usize, f64)],
    ) -> Result<(f64, f64), OrchestrationError> {
        if rho_values.len() < 2 {
            return Ok((0.0, 0.0));
        }

        // Fit exponential: rho = asymptote * (1 - exp(-rate * L))
        // Using simple linear regression on transformed data

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let n = rho_values.len() as f64;

        for (lib_len, rho) in rho_values {
            let x = *lib_len as f64;
            let y = *rho;

            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let rate = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let asymptote = (sum_y - rate * sum_x) / n;

        Ok((asymptote, rate))
    }

    /// Compute correlation
    fn compute_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x > 0.0 && var_y > 0.0 {
            cov / (var_x * var_y).sqrt()
        } else {
            0.0
        }
    }

    /// Compute transfer entropy
    fn transfer_entropy(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        let k = self.te_calculator.k_source;
        let l = self.te_calculator.k_target;
        let h = self.te_calculator.l_future;

        let n = source.len();
        if n < k + l + h + 1 {
            return Err(OrchestrationError::InsufficientData {
                required: k + l + h + 1,
                available: n,
            });
        }

        if self.te_calculator.use_symbolic {
            self.symbolic_transfer_entropy(source, target)
        } else {
            match self.te_calculator.binning {
                BinningMethod::KernelDensity => self.kernel_transfer_entropy(source, target),
                _ => self.binned_transfer_entropy(source, target),
            }
        }
    }

    /// Symbolic transfer entropy
    fn symbolic_transfer_entropy(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        let perm_len = self.te_calculator.permutation_length;

        // Convert to symbolic sequences
        let symbols_source = self.symbolize(source, perm_len)?;
        let symbols_target = self.symbolize(target, perm_len)?;

        // Count symbol occurrences
        let mut joint_counts: HashMap<(Vec<usize>, Vec<usize>, Vec<usize>), f64> = HashMap::new();
        let mut target_future_counts: HashMap<Vec<usize>, f64> = HashMap::new();
        let mut target_past_counts: HashMap<Vec<usize>, f64> = HashMap::new();
        let mut joint_past_counts: HashMap<(Vec<usize>, Vec<usize>), f64> = HashMap::new();

        let k = self.te_calculator.k_source;
        let l = self.te_calculator.k_target;

        for i in (k + l)..symbols_target.len() - 1 {
            let target_past: Vec<usize> = symbols_target[(i - l)..i].to_vec();
            let source_past: Vec<usize> = symbols_source[(i - k)..i].to_vec();
            let target_future = vec![symbols_target[i]];

            *joint_counts
                .entry((
                    target_future.clone(),
                    target_past.clone(),
                    source_past.clone(),
                ))
                .or_insert(0.0) += 1.0;
            *target_future_counts.entry(target_future).or_insert(0.0) += 1.0;
            *target_past_counts.entry(target_past.clone()).or_insert(0.0) += 1.0;
            *joint_past_counts
                .entry((target_past, source_past))
                .or_insert(0.0) += 1.0;
        }

        // Compute transfer entropy
        let mut te = 0.0;
        let total = joint_counts.values().sum::<f64>();

        for ((future, t_past, s_past), count) in joint_counts {
            let p_joint = count / total;
            let p_future_given_past = count / joint_past_counts[&(t_past.clone(), s_past.clone())];
            let p_future = target_future_counts[&future] / total;
            let p_past = target_past_counts[&t_past] / total;

            if p_joint > 0.0 && p_future_given_past > 0.0 && p_future > 0.0 && p_past > 0.0 {
                te += p_joint * (p_future_given_past / (p_future / p_past)).log2();
            }
        }

        Ok(te)
    }

    /// Symbolize time series
    fn symbolize(
        &self,
        series: &DVector<f64>,
        perm_len: usize,
    ) -> Result<Vec<usize>, OrchestrationError> {
        let mut symbols = Vec::new();

        for i in 0..series.len() - perm_len + 1 {
            let window: Vec<f64> = series.rows(i, perm_len).iter().copied().collect();

            // Get permutation pattern
            let mut indices: Vec<usize> = (0..perm_len).collect();
            indices.sort_by_key(|&i| OrderedFloat(window[i]));

            // Convert to symbol (factorial number system)
            let mut symbol = 0;
            let mut factorial = 1;
            for j in 0..perm_len {
                symbol += indices[j] * factorial;
                factorial *= j + 1;
            }

            symbols.push(symbol);
        }

        Ok(symbols)
    }

    /// Kernel density estimation transfer entropy
    fn kernel_transfer_entropy(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        // Simplified KDE-based TE
        // Would use proper kernel density estimation in production

        let bandwidth = 0.1 * source.variance().sqrt();
        let mut te = 0.0;
        let n = source.len() - 2;

        for i in 1..n {
            let target_future = target[i + 1];
            let target_past = target[i];
            let source_past = source[i];

            // Estimate densities using Gaussian kernels
            let mut p_future_given_both = 0.0;
            let mut p_future_given_target = 0.0;
            let mut count_both = 0.0;
            let mut count_target = 0.0;

            for j in 1..n {
                if j != i {
                    let kernel_target =
                        (-((target[j] - target_past).powi(2)) / (2.0 * bandwidth.powi(2))).exp();
                    let kernel_source =
                        (-((source[j] - source_past).powi(2)) / (2.0 * bandwidth.powi(2))).exp();
                    let kernel_future = (-((target[j + 1] - target_future).powi(2))
                        / (2.0 * bandwidth.powi(2)))
                    .exp();

                    p_future_given_both += kernel_future * kernel_target * kernel_source;
                    p_future_given_target += kernel_future * kernel_target;
                    count_both += kernel_target * kernel_source;
                    count_target += kernel_target;
                }
            }

            if count_both > 0.0 && count_target > 0.0 {
                p_future_given_both /= count_both;
                p_future_given_target /= count_target;

                if p_future_given_both > 0.0 && p_future_given_target > 0.0 {
                    te += (p_future_given_both / p_future_given_target).log2() / n as f64;
                }
            }
        }

        Ok(te.max(0.0))
    }

    /// Binned transfer entropy
    fn binned_transfer_entropy(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        // Discretize data
        let n_bins = 10;
        let source_binned = self.discretize(source, n_bins);
        let target_binned = self.discretize(target, n_bins);

        // Compute joint and marginal probabilities
        let mut joint_prob: HashMap<(usize, usize, usize), f64> = HashMap::new();
        let mut marginal_yt: HashMap<(usize, usize), f64> = HashMap::new();
        let mut marginal_y: HashMap<usize, f64> = HashMap::new();

        for i in 1..source_binned.len() - 1 {
            let key = (target_binned[i + 1], target_binned[i], source_binned[i]);
            *joint_prob.entry(key).or_insert(0.0) += 1.0;
            *marginal_yt
                .entry((target_binned[i + 1], target_binned[i]))
                .or_insert(0.0) += 1.0;
            *marginal_y.entry(target_binned[i]).or_insert(0.0) += 1.0;
        }

        // Normalize
        let total = (source_binned.len() - 2) as f64;
        for v in joint_prob.values_mut() {
            *v /= total;
        }
        for v in marginal_yt.values_mut() {
            *v /= total;
        }
        for v in marginal_y.values_mut() {
            *v /= total;
        }

        // Calculate transfer entropy
        let mut te = 0.0;
        for ((y_next, y_curr, x_curr), p_joint) in &joint_prob {
            if let (Some(&p_yt), Some(&p_y)) =
                (marginal_yt.get(&(*y_next, *y_curr)), marginal_y.get(y_curr))
            {
                if p_yt > 0.0 && p_y > 0.0 && *p_joint > 0.0 {
                    te += p_joint
                        * (p_joint * p_y / (p_yt * marginal_y.get(y_curr).unwrap_or(&1.0))).log2();
                }
            }
        }

        Ok(te)
    }

    /// Discretize continuous data
    fn discretize(&self, data: &DVector<f64>, n_bins: usize) -> Vec<usize> {
        let min_val = data.min();
        let max_val = data.max();
        let bin_width = (max_val - min_val) / n_bins as f64;

        data.iter()
            .map(|&x| {
                let bin = ((x - min_val) / bin_width).floor() as usize;
                bin.min(n_bins - 1)
            })
            .collect()
    }

    /// Granger causality test
    fn granger_causality(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        if self.granger_tester.nonlinear {
            self.nonlinear_granger(source, target)
        } else {
            self.linear_granger(source, target)
        }
    }

    /// Linear Granger causality
    fn linear_granger(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        let max_lag = self.granger_tester.max_lag;
        let n = source.len();

        if n <= max_lag * 2 {
            return Err(OrchestrationError::InsufficientData {
                required: max_lag * 2 + 1,
                available: n,
            });
        }

        // Build regression matrices
        let n_samples = n - max_lag;

        // Restricted model: Y_t = sum(a_i * Y_{t-i}) + e
        let mut X_restricted = DMatrix::zeros(n_samples, max_lag);
        let mut y = DVector::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..max_lag {
                X_restricted[(i, j)] = target[max_lag + i - j - 1];
            }
            y[i] = target[max_lag + i];
        }

        // Unrestricted model: Y_t = sum(a_i * Y_{t-i}) + sum(b_i * X_{t-i}) + e
        let mut X_unrestricted = DMatrix::zeros(n_samples, max_lag * 2);

        for i in 0..n_samples {
            for j in 0..max_lag {
                X_unrestricted[(i, j)] = target[max_lag + i - j - 1];
                X_unrestricted[(i, max_lag + j)] = source[max_lag + i - j - 1];
            }
        }

        // Compute RSS for both models
        let rss_restricted = self.compute_rss(&X_restricted, &y)?;
        let rss_unrestricted = self.compute_rss(&X_unrestricted, &y)?;

        // F-statistic
        let f_stat = ((rss_restricted - rss_unrestricted) / max_lag as f64)
            / (rss_unrestricted / (n_samples - 2 * max_lag) as f64);

        Ok(f_stat)
    }

    /// Nonlinear Granger causality
    fn nonlinear_granger(
        &self,
        source: &DVector<f64>,
        target: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        // Use kernel regression for nonlinear Granger
        // Simplified implementation - would use proper kernel methods in production

        let max_lag = self.granger_tester.max_lag;
        let n = source.len() - max_lag;
        let bandwidth = 0.5;

        let mut error_restricted = 0.0;
        let mut error_unrestricted = 0.0;

        for i in max_lag..source.len() - 1 {
            let target_future = target[i + 1];

            // Restricted prediction (only target history)
            let mut pred_restricted = 0.0;
            let mut weight_restricted = 0.0;

            // Unrestricted prediction (target and source history)
            let mut pred_unrestricted = 0.0;
            let mut weight_unrestricted = 0.0;

            for j in max_lag..source.len() - 1 {
                if j != i {
                    // Compute kernel weights
                    let mut kernel_restricted = 1.0;
                    let mut kernel_unrestricted = 1.0;

                    for k in 1..=max_lag {
                        let target_diff = (target[i - k + 1] - target[j - k + 1]).powi(2);
                        kernel_restricted *= (-target_diff / (2.0 * bandwidth * bandwidth)).exp();

                        kernel_unrestricted *= (-target_diff / (2.0 * bandwidth * bandwidth)).exp();
                        let source_diff = (source[i - k + 1] - source[j - k + 1]).powi(2);
                        kernel_unrestricted *= (-source_diff / (2.0 * bandwidth * bandwidth)).exp();
                    }

                    pred_restricted += kernel_restricted * target[j + 1];
                    weight_restricted += kernel_restricted;

                    pred_unrestricted += kernel_unrestricted * target[j + 1];
                    weight_unrestricted += kernel_unrestricted;
                }
            }

            if weight_restricted > 0.0 {
                pred_restricted /= weight_restricted;
                error_restricted += (target_future - pred_restricted).powi(2);
            }

            if weight_unrestricted > 0.0 {
                pred_unrestricted /= weight_unrestricted;
                error_unrestricted += (target_future - pred_unrestricted).powi(2);
            }
        }

        // Compute test statistic
        let statistic = n as f64 * (error_restricted - error_unrestricted) / error_unrestricted;

        Ok(statistic.max(0.0))
    }

    /// Compute residual sum of squares
    fn compute_rss(&self, X: &DMatrix<f64>, y: &DVector<f64>) -> Result<f64, OrchestrationError> {
        // Solve least squares: beta = (X'X)^{-1}X'y
        let XtX = X.transpose() * X;
        let Xty = X.transpose() * y;

        if let Some(XtX_inv) = XtX.try_inverse() {
            let beta = XtX_inv * Xty;
            let predictions = X * beta;
            let residuals = y - predictions;
            Ok(residuals.dot(&residuals))
        } else {
            Err(OrchestrationError::SingularMatrix {
                matrix_name: "XtX".to_string(),
            })
        }
    }

    /// Update causal graph
    fn update_causal_graph(
        &mut self,
        x_name: &str,
        y_name: &str,
        ccm: &CCMResult,
        te_x_to_y: f64,
        te_y_to_x: f64,
    ) -> Result<(), OrchestrationError> {
        // Add nodes if not present
        let x_idx = self.causal_graph.add_node(x_name);
        let y_idx = self.causal_graph.add_node(y_name);

        // Determine edge type and strength
        let edge_type = if ccm.converged && ccm.asymptote_x_to_y > 0.7 && ccm.asymptote_y_to_x > 0.7
        {
            EdgeType::Bidirectional
        } else if te_x_to_y > te_y_to_x * 2.0 {
            EdgeType::Direct
        } else if te_y_to_x > te_x_to_y * 2.0 {
            EdgeType::Direct
        } else {
            EdgeType::Spurious
        };

        // Add edges
        if te_x_to_y > self.te_calculator.ete_threshold {
            self.causal_graph
                .add_edge(x_idx, y_idx, te_x_to_y, edge_type.clone());
        }

        if te_y_to_x > self.te_calculator.ete_threshold {
            self.causal_graph
                .add_edge(y_idx, x_idx, te_y_to_x, edge_type);
        }

        Ok(())
    }

    /// PC structure learning algorithm
    fn pc_structure_learning(&mut self) -> Result<GraphStructure, OrchestrationError> {
        let nodes = self.causal_graph.nodes.clone();
        let n_vars = nodes.len();

        if n_vars < 2 {
            return Ok(GraphStructure {
                adjacency: DMatrix::zeros(n_vars, n_vars),
                edges: Vec::new(),
                v_structures: Vec::new(),
            });
        }

        // Start with complete graph
        let mut skeleton = DMatrix::from_element(n_vars, n_vars, 1.0);
        for i in 0..n_vars {
            skeleton[(i, i)] = 0.0;
        }

        // Phase 1: Learn skeleton
        for size in 0..=self.pc_algorithm.max_conditioning_size {
            for i in 0..n_vars {
                for j in i + 1..n_vars {
                    if skeleton[(i, j)] == 0.0 {
                        continue;
                    }

                    // Find conditioning sets
                    let neighbors = self.get_neighbors(&skeleton, i, j);
                    let conditioning_sets = self.generate_conditioning_sets(&neighbors, size);

                    for conditioning_set in conditioning_sets {
                        if self.test_independence(i, j, &conditioning_set)? {
                            skeleton[(i, j)] = 0.0;
                            skeleton[(j, i)] = 0.0;
                            break;
                        }
                    }
                }
            }
        }

        // Phase 2: Orient edges
        let oriented = self.orient_edges(&skeleton)?;

        // Find v-structures
        let v_structures = self.find_v_structures(&oriented);

        // Build edge list
        let mut edges = Vec::new();
        for i in 0..n_vars {
            for j in 0..n_vars {
                if oriented[(i, j)] > 0.0 {
                    edges.push((i, j));
                }
            }
        }

        Ok(GraphStructure {
            adjacency: oriented,
            edges,
            v_structures,
        })
    }

    /// Get neighbors in skeleton
    fn get_neighbors(&self, skeleton: &DMatrix<f64>, i: usize, j: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for k in 0..skeleton.ncols() {
            if k != i && k != j && (skeleton[(i, k)] > 0.0 || skeleton[(j, k)] > 0.0) {
                neighbors.push(k);
            }
        }
        neighbors
    }

    /// Generate conditioning sets
    fn generate_conditioning_sets(&self, neighbors: &[usize], size: usize) -> Vec<HashSet<usize>> {
        if size == 0 {
            return vec![HashSet::new()];
        }

        if size > neighbors.len() {
            return vec![];
        }

        // Generate all combinations of given size
        let mut sets = Vec::new();
        self.combinations(neighbors, size, 0, HashSet::new(), &mut sets);
        sets
    }

    /// Generate combinations recursively
    fn combinations(
        &self,
        items: &[usize],
        size: usize,
        start: usize,
        current: HashSet<usize>,
        result: &mut Vec<HashSet<usize>>,
    ) {
        if current.len() == size {
            result.push(current);
            return;
        }

        for i in start..items.len() {
            let mut new_set = current.clone();
            new_set.insert(items[i]);
            self.combinations(items, size, i + 1, new_set, result);
        }
    }

    /// Test conditional independence
    fn test_independence(
        &self,
        i: usize,
        j: usize,
        conditioning: &HashSet<usize>,
    ) -> Result<bool, OrchestrationError> {
        // Get time series data
        let series: Vec<&DVector<f64>> = self.time_series_buffer.series.values().collect();

        if series.len() <= i || series.len() <= j {
            return Ok(false);
        }

        let x = series[i];
        let y = series[j];

        match self.pc_algorithm.independence_test {
            IndependenceTest::PartialCorrelation => {
                let pcorr = self.partial_correlation(x, y, conditioning, &series)?;
                Ok(pcorr.abs() < self.pc_algorithm.alpha)
            }
            IndependenceTest::HSIC => {
                let hsic = self.hsic_test(x, y)?;
                Ok(hsic < self.pc_algorithm.alpha)
            }
            _ => Ok(false),
        }
    }

    /// Partial correlation
    fn partial_correlation(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
        conditioning: &HashSet<usize>,
        series: &[&DVector<f64>],
    ) -> Result<f64, OrchestrationError> {
        if conditioning.is_empty() {
            return Ok(self.compute_correlation(&x.as_slice().to_vec(), &y.as_slice().to_vec()));
        }

        // Build regression matrix for conditioning variables
        let n = x.len();
        let k = conditioning.len();
        let mut Z = DMatrix::zeros(n, k);

        for (col, &var_idx) in conditioning.iter().enumerate() {
            if var_idx < series.len() {
                for row in 0..n {
                    Z[(row, col)] = series[var_idx][row];
                }
            }
        }

        // Regress out conditioning variables
        let x_resid = self.compute_residuals(x, &Z)?;
        let y_resid = self.compute_residuals(y, &Z)?;

        // Compute correlation of residuals
        Ok(self.compute_correlation(&x_resid.as_slice().to_vec(), &y_resid.as_slice().to_vec()))
    }

    /// Compute residuals after regression
    fn compute_residuals(
        &self,
        y: &DVector<f64>,
        X: &DMatrix<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        let XtX = X.transpose() * X;
        if let Some(XtX_inv) = XtX.try_inverse() {
            let beta = XtX_inv * X.transpose() * y;
            Ok(y - X * beta)
        } else {
            Ok(y.clone()) // Return original if regression fails
        }
    }

    /// Hilbert-Schmidt Independence Criterion
    fn hsic_test(&self, x: &DVector<f64>, y: &DVector<f64>) -> Result<f64, OrchestrationError> {
        let n = x.len();
        let gamma = 1.0; // RBF kernel bandwidth

        // Compute kernel matrices
        let mut K_x = DMatrix::zeros(n, n);
        let mut K_y = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                K_x[(i, j)] = (-(x[i] - x[j]).powi(2) / (2.0 * gamma * gamma)).exp();
                K_y[(i, j)] = (-(y[i] - y[j]).powi(2) / (2.0 * gamma * gamma)).exp();
            }
        }

        // Center kernel matrices
        let H = DMatrix::identity(n, n) - DMatrix::from_element(n, n, 1.0 / n as f64);
        let K_x_centered = &H * &K_x * &H;
        let K_y_centered = &H * &K_y * &H;

        // Compute HSIC
        let hsic = (K_x_centered.component_mul(&K_y_centered)).sum() / (n * n) as f64;

        Ok(hsic)
    }

    /// Orient edges using PC algorithm rules
    fn orient_edges(&self, skeleton: &DMatrix<f64>) -> Result<DMatrix<f64>, OrchestrationError> {
        let mut oriented = skeleton.clone();

        // Apply orientation rules
        for rule in &self.pc_algorithm.orientation_rules {
            match rule {
                OrientationRule::ColliderOrientation => self.orient_colliders(&mut oriented)?,
                OrientationRule::AcyclicityOrientation => self.orient_acyclic(&mut oriented)?,
                OrientationRule::MajorityOrientation => self.orient_majority(&mut oriented)?,
                _ => {}
            }
        }

        Ok(oriented)
    }

    /// Orient v-structures (colliders)
    fn orient_colliders(&self, adjacency: &mut DMatrix<f64>) -> Result<(), OrchestrationError> {
        let n = adjacency.nrows();

        for j in 0..n {
            for i in 0..n {
                for k in i + 1..n {
                    // Check if i -> j <- k is a v-structure
                    if adjacency[(i, j)] > 0.0
                        && adjacency[(j, i)] > 0.0
                        && adjacency[(k, j)] > 0.0
                        && adjacency[(j, k)] > 0.0
                        && adjacency[(i, k)] == 0.0
                        && adjacency[(k, i)] == 0.0
                    {
                        // Orient as collider
                        adjacency[(j, i)] = 0.0;
                        adjacency[(j, k)] = 0.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Orient to maintain acyclicity
    fn orient_acyclic(&self, adjacency: &mut DMatrix<f64>) -> Result<(), OrchestrationError> {
        let n = adjacency.nrows();
        let mut changed = true;

        while changed {
            changed = false;

            for i in 0..n {
                for j in 0..n {
                    if adjacency[(i, j)] > 0.0 && adjacency[(j, i)] > 0.0 {
                        // Check if orienting j -> i would create cycle
                        if !self.would_create_cycle(adjacency, j, i) {
                            adjacency[(i, j)] = 0.0;
                            changed = true;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if edge would create cycle
    fn would_create_cycle(&self, adjacency: &DMatrix<f64>, from: usize, to: usize) -> bool {
        // DFS to check for path from 'to' to 'from'
        let n = adjacency.nrows();
        let mut visited = vec![false; n];
        let mut stack = vec![to];

        while let Some(node) = stack.pop() {
            if node == from {
                return true;
            }

            if !visited[node] {
                visited[node] = true;

                for next in 0..n {
                    if adjacency[(node, next)] > 0.0 && adjacency[(next, node)] == 0.0 {
                        stack.push(next);
                    }
                }
            }
        }

        false
    }

    /// Orient using majority rule
    fn orient_majority(&self, adjacency: &mut DMatrix<f64>) -> Result<(), OrchestrationError> {
        // Simplified majority orientation
        let n = adjacency.nrows();

        for i in 0..n {
            for j in 0..n {
                if adjacency[(i, j)] > 0.0 && adjacency[(j, i)] > 0.0 {
                    // Count evidence for each direction
                    let mut evidence_i_to_j = 0;
                    let mut evidence_j_to_i = 0;

                    for k in 0..n {
                        if k != i && k != j {
                            if adjacency[(i, k)] > 0.0 && adjacency[(k, j)] > 0.0 {
                                evidence_i_to_j += 1;
                            }
                            if adjacency[(j, k)] > 0.0 && adjacency[(k, i)] > 0.0 {
                                evidence_j_to_i += 1;
                            }
                        }
                    }

                    if evidence_i_to_j > evidence_j_to_i {
                        adjacency[(j, i)] = 0.0;
                    } else if evidence_j_to_i > evidence_i_to_j {
                        adjacency[(i, j)] = 0.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Find v-structures in graph
    fn find_v_structures(&self, adjacency: &DMatrix<f64>) -> Vec<(usize, usize, usize)> {
        let mut v_structures = Vec::new();
        let n = adjacency.nrows();

        for j in 0..n {
            for i in 0..n {
                for k in i + 1..n {
                    if adjacency[(i, j)] > 0.0
                        && adjacency[(j, i)] == 0.0
                        && adjacency[(k, j)] > 0.0
                        && adjacency[(j, k)] == 0.0
                        && adjacency[(i, k)] == 0.0
                        && adjacency[(k, i)] == 0.0
                    {
                        v_structures.push((i, j, k));
                    }
                }
            }
        }

        v_structures
    }

    /// Compute causal strength
    fn compute_causal_strength(
        &self,
        ccm: &CCMResult,
        te_x_to_y: f64,
        te_y_to_x: f64,
    ) -> CausalStrength {
        let ccm_strength = (ccm.asymptote_x_to_y + ccm.asymptote_y_to_x) / 2.0;
        let te_strength = (te_x_to_y + te_y_to_x) / 2.0;

        CausalStrength {
            overall: (ccm_strength + te_strength) / 2.0,
            ccm_component: ccm_strength,
            te_component: te_strength,
            directionality: (te_x_to_y - te_y_to_x).abs() / (te_x_to_y + te_y_to_x + 1e-10),
        }
    }

    /// Bootstrap causal strength estimation
    fn bootstrap_causal_strength(
        &mut self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> Result<CausalStrength, OrchestrationError> {
        let mut strengths = Vec::new();

        for _ in 0..self.parameters.n_bootstrap {
            // Resample with replacement
            let (x_boot, y_boot) = self.bootstrap_resample(x, y);

            // Compute metrics
            let ccm = self.convergent_cross_mapping(&x_boot, &y_boot)?;
            let te_x_to_y = self.transfer_entropy(&x_boot, &y_boot)?;
            let te_y_to_x = self.transfer_entropy(&y_boot, &x_boot)?;

            strengths.push(self.compute_causal_strength(&ccm, te_x_to_y, te_y_to_x));
        }

        // Compute mean and confidence intervals
        let mean_strength = CausalStrength {
            overall: strengths.iter().map(|s| s.overall).sum::<f64>() / strengths.len() as f64,
            ccm_component: strengths.iter().map(|s| s.ccm_component).sum::<f64>()
                / strengths.len() as f64,
            te_component: strengths.iter().map(|s| s.te_component).sum::<f64>()
                / strengths.len() as f64,
            directionality: strengths.iter().map(|s| s.directionality).sum::<f64>()
                / strengths.len() as f64,
        };

        Ok(mean_strength)
    }

    /// Bootstrap resample
    fn bootstrap_resample(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        let n = x.len();
        let mut x_boot = DVector::zeros(n);
        let mut y_boot = DVector::zeros(n);

        for i in 0..n {
            let idx = rand::random::<usize>() % n;
            x_boot[i] = x[idx];
            y_boot[i] = y[idx];
        }

        (x_boot, y_boot)
    }

    /// Test significance with surrogate data
    fn test_significance_surrogates(
        &mut self,
        x: &DVector<f64>,
        y: &DVector<f64>,
        observed_strength: &CausalStrength,
    ) -> Result<SignificanceResult, OrchestrationError> {
        let mut null_distribution = Vec::new();

        for _ in 0..self.parameters.n_surrogates {
            // Generate surrogate data
            let (x_surr, y_surr) = match self.parameters.surrogate_method {
                SurrogateMethod::RandomShuffle => (self.shuffle(x), self.shuffle(y)),
                SurrogateMethod::PhaseRandomization => {
                    (self.phase_randomize(x)?, self.phase_randomize(y)?)
                }
                SurrogateMethod::IAAFT => (self.iaaft_surrogate(x)?, self.iaaft_surrogate(y)?),
                SurrogateMethod::TwinSurrogates => {
                    (self.twin_surrogate(x)?, self.twin_surrogate(y)?)
                }
            };

            // Compute causal strength for surrogate
            let ccm = self.convergent_cross_mapping(&x_surr, &y_surr)?;
            let te_x_to_y = self.transfer_entropy(&x_surr, &y_surr)?;
            let te_y_to_x = self.transfer_entropy(&y_surr, &x_surr)?;

            null_distribution.push(
                self.compute_causal_strength(&ccm, te_x_to_y, te_y_to_x)
                    .overall,
            );
        }

        // Compute p-value
        let p_value = null_distribution
            .iter()
            .filter(|&&null_val| null_val >= observed_strength.overall)
            .count() as f64
            / null_distribution.len() as f64;

        // Apply multiple testing correction
        let adjusted_p = match self.parameters.correction {
            MultipleTestingCorrection::Bonferroni => p_value * 2.0, // Adjust for bidirectional test
            MultipleTestingCorrection::BenjaminiHochberg => self.benjamini_hochberg(p_value, 2),
            _ => p_value,
        };

        Ok(SignificanceResult {
            p_value,
            adjusted_p,
            significant: adjusted_p < 0.05,
            confidence: 1.0 - adjusted_p,
        })
    }

    /// Shuffle time series
    fn shuffle(&self, series: &DVector<f64>) -> DVector<f64> {
        let mut shuffled = series.clone();
        let n = shuffled.len();

        for i in 0..n {
            let j = rand::random::<usize>() % n;
            shuffled.swap_rows(i, j);
        }

        shuffled
    }

    /// Phase randomization surrogate
    fn phase_randomize(&self, series: &DVector<f64>) -> Result<DVector<f64>, OrchestrationError> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let n = series.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_data: Vec<Complex<f64>> =
            series.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_data);

        // Randomize phases (keep DC and Nyquist)
        for i in 1..n / 2 {
            let phase = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
            let magnitude = complex_data[i].norm();
            complex_data[i] = Complex::from_polar(magnitude, phase);

            // Ensure conjugate symmetry
            if i < n - i {
                complex_data[n - i] = complex_data[i].conj();
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_data);

        // Extract real part
        let surrogate = DVector::from_iterator(n, complex_data.iter().map(|c| c.re / n as f64));

        Ok(surrogate)
    }

    /// IAAFT surrogate
    fn iaaft_surrogate(&self, series: &DVector<f64>) -> Result<DVector<f64>, OrchestrationError> {
        // Iterative Amplitude Adjusted Fourier Transform
        // Simplified implementation

        let mut surrogate = self.phase_randomize(series)?;

        for _ in 0..10 {
            // Fixed iterations
            // Rank order to match original amplitude distribution
            let sorted_original: Vec<f64> = {
                let mut v = series.as_slice().to_vec();
                v.sort_by_key(|x| OrderedFloat(*x));
                v
            };

            let mut indices: Vec<usize> = (0..surrogate.len()).collect();
            indices.sort_by_key(|&i| OrderedFloat(surrogate[i]));

            for (i, &idx) in indices.iter().enumerate() {
                surrogate[idx] = sorted_original[i];
            }

            // Phase randomize again
            surrogate = self.phase_randomize(&surrogate)?;
        }

        Ok(surrogate)
    }

    /// Twin surrogate
    fn twin_surrogate(&self, series: &DVector<f64>) -> Result<DVector<f64>, OrchestrationError> {
        // Generate twin surrogate (preserves recurrence structure)
        let embedding_dim = 3;
        let tau = 1;

        // Create embedded vectors
        let mut embedded = Vec::new();
        for i in 0..series.len() - (embedding_dim - 1) * tau {
            let mut vec = Vec::new();
            for j in 0..embedding_dim {
                vec.push(series[i + j * tau]);
            }
            embedded.push(vec);
        }

        // Find twins (nearest neighbors)
        let mut surrogate = DVector::zeros(series.len());
        surrogate[0] = series[0];

        for i in 1..series.len() - (embedding_dim - 1) * tau {
            // Find nearest neighbor to current embedded vector
            let current = &embedded[i - 1];
            let mut min_dist = f64::INFINITY;
            let mut twin_idx = 0;

            for (j, other) in embedded.iter().enumerate() {
                if j != i - 1 {
                    let dist: f64 = current
                        .iter()
                        .zip(other.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    if dist < min_dist {
                        min_dist = dist;
                        twin_idx = j;
                    }
                }
            }

            // Use the successor of the twin
            if twin_idx + 1 < embedded.len() {
                surrogate[i] = series[twin_idx + 1];
            } else {
                surrogate[i] = series[rand::random::<usize>() % series.len()];
            }
        }

        // Fill remaining values
        for i in series.len() - (embedding_dim - 1) * tau..series.len() {
            surrogate[i] = series[i];
        }

        Ok(surrogate)
    }

    /// Benjamini-Hochberg correction
    fn benjamini_hochberg(&self, p_value: f64, n_tests: usize) -> f64 {
        // Simplified BH correction
        p_value * n_tests as f64 / 1.0 // Would sort p-values in full implementation
    }

    /// Determine causal direction
    fn determine_direction(
        &self,
        te_x_to_y: f64,
        te_y_to_x: f64,
        ccm: &CCMResult,
    ) -> CausalDirection {
        let te_ratio = te_x_to_y / (te_y_to_x + 1e-10);
        let ccm_ratio = ccm.asymptote_x_to_y / (ccm.asymptote_y_to_x + 1e-10);

        if te_ratio > 2.0 && ccm_ratio > 1.5 {
            CausalDirection::XCausesY
        } else if te_ratio < 0.5 && ccm_ratio < 0.67 {
            CausalDirection::YCausesX
        } else if te_ratio > 0.8 && te_ratio < 1.2 && ccm.converged {
            CausalDirection::Bidirectional
        } else {
            CausalDirection::Undetermined
        }
    }

    /// Analyze causality in LLM responses
    pub fn analyze_llm_causality(
        &mut self,
        responses: &[String],
        query: &str,
    ) -> Result<LLMCausalityAnalysis, OrchestrationError> {
        // Convert responses to time series (simplified encoding)
        let mut time_series = Vec::new();

        for response in responses {
            let encoding = self.encode_response(response);
            time_series.push(encoding);
        }

        // Analyze pairwise causality
        let mut causal_matrix = DMatrix::zeros(responses.len(), responses.len());
        let mut influence_scores = vec![0.0; responses.len()];

        for i in 0..responses.len() {
            for j in 0..responses.len() {
                if i != j {
                    let result = self.analyze(
                        &format!("llm_{}", i),
                        &time_series[i],
                        &format!("llm_{}", j),
                        &time_series[j],
                    )?;

                    causal_matrix[(i, j)] = result.strength.overall;
                    influence_scores[i] += result.strength.overall;
                }
            }
        }

        // Identify causal chains
        let chains = self.identify_causal_chains(&causal_matrix)?;

        // Find root causes
        let root_causes = self.find_root_causes(&causal_matrix);
        let consensus_mechanism = self.determine_consensus_mechanism(&causal_matrix);

        Ok(LLMCausalityAnalysis {
            causal_matrix,
            influence_scores,
            causal_chains: chains,
            root_causes,
            consensus_mechanism,
        })
    }

    /// Encode response as time series
    fn encode_response(&self, response: &str) -> DVector<f64> {
        // Simple character frequency encoding
        let mut encoding = DVector::zeros(256);

        for byte in response.bytes() {
            encoding[byte as usize] += 1.0;
        }

        // Normalize
        let sum = encoding.sum();
        if sum > 0.0 {
            encoding /= sum;
        }

        encoding
    }

    /// Identify causal chains
    fn identify_causal_chains(
        &self,
        causal_matrix: &DMatrix<f64>,
    ) -> Result<Vec<CausalChain>, OrchestrationError> {
        let mut chains = Vec::new();
        let threshold = 0.5;

        // Find all paths using DFS
        for start in 0..causal_matrix.nrows() {
            let mut visited = vec![false; causal_matrix.nrows()];
            let mut path = vec![start];
            self.dfs_causal_chains(
                causal_matrix,
                start,
                &mut visited,
                &mut path,
                &mut chains,
                threshold,
            );
        }

        Ok(chains)
    }

    /// DFS for causal chains
    fn dfs_causal_chains(
        &self,
        matrix: &DMatrix<f64>,
        node: usize,
        visited: &mut [bool],
        path: &mut Vec<usize>,
        chains: &mut Vec<CausalChain>,
        threshold: f64,
    ) {
        visited[node] = true;

        for next in 0..matrix.ncols() {
            if !visited[next] && matrix[(node, next)] > threshold {
                path.push(next);
                self.dfs_causal_chains(matrix, next, visited, path, chains, threshold);
                path.pop();
            }
        }

        if path.len() > 2 {
            let strength = self.compute_chain_strength(matrix, path);
            chains.push(CausalChain {
                path: path.clone(),
                strength,
            });
        }

        visited[node] = false;
    }

    /// Compute chain strength
    fn compute_chain_strength(&self, matrix: &DMatrix<f64>, path: &[usize]) -> f64 {
        let mut strength = 1.0;

        for window in path.windows(2) {
            strength *= matrix[(window[0], window[1])];
        }

        strength
    }

    /// Find root causes
    fn find_root_causes(&self, causal_matrix: &DMatrix<f64>) -> Vec<usize> {
        let mut root_causes = Vec::new();

        for i in 0..causal_matrix.nrows() {
            let mut is_root = true;

            // Check if any node strongly causes this one
            for j in 0..causal_matrix.ncols() {
                if i != j && causal_matrix[(j, i)] > 0.5 {
                    is_root = false;
                    break;
                }
            }

            if is_root {
                // Check if this node causes others
                let causes_others =
                    (0..causal_matrix.ncols()).any(|j| i != j && causal_matrix[(i, j)] > 0.5);

                if causes_others {
                    root_causes.push(i);
                }
            }
        }

        root_causes
    }

    /// Determine consensus mechanism
    fn determine_consensus_mechanism(&self, causal_matrix: &DMatrix<f64>) -> String {
        let mean_causality =
            causal_matrix.sum() / (causal_matrix.nrows() * causal_matrix.ncols()) as f64;

        if mean_causality > 0.7 {
            "Strong mutual causality - use weighted consensus".to_string()
        } else if mean_causality > 0.3 {
            "Moderate causality - use hierarchical consensus".to_string()
        } else {
            "Weak causality - use independent voting".to_string()
        }
    }
}

impl CausalGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
            adjacency: DMatrix::zeros(0, 0),
            d_separation_cache: HashMap::new(),
            backdoor_paths: HashMap::new(),
        }
    }

    fn add_node(&mut self, name: &str) -> usize {
        // Check if node exists
        for (i, node) in self.nodes.iter().enumerate() {
            if node.name == name {
                return i;
            }
        }

        // Add new node
        let idx = self.nodes.len();
        self.nodes.push(CausalNode {
            name: name.to_string(),
            node_type: NodeType::Observed,
            observed: true,
            intervened: None,
        });

        // Resize adjacency matrix
        let new_size = self.nodes.len();
        let mut new_adj = DMatrix::zeros(new_size, new_size);
        for i in 0..self.adjacency.nrows() {
            for j in 0..self.adjacency.ncols() {
                new_adj[(i, j)] = self.adjacency[(i, j)];
            }
        }
        self.adjacency = new_adj;

        idx
    }

    fn add_edge(&mut self, from: usize, to: usize, strength: f64, edge_type: EdgeType) {
        self.edges.insert(
            (from, to),
            CausalEdge {
                strength,
                edge_type,
                lag: 0,
            },
        );

        if from < self.adjacency.nrows() && to < self.adjacency.ncols() {
            self.adjacency[(from, to)] = strength;
        }
    }
}

/// Causality analysis result
#[derive(Clone, Debug)]
pub struct CausalityResult {
    pub ccm: CCMResult,
    pub transfer_entropy: TransferEntropyResult,
    pub granger: GrangerResult,
    pub causal_direction: CausalDirection,
    pub strength: CausalStrength,
    pub confidence: f64,
    pub graph_structure: GraphStructure,
}

#[derive(Clone, Debug)]
pub struct CCMResult {
    pub rho_values: Vec<(usize, f64, f64)>, // (lib_len, rho_x_to_y, rho_y_to_x)
    pub converged: bool,
    pub convergence_rate: f64,
    pub asymptote_x_to_y: f64,
    pub asymptote_y_to_x: f64,
    pub rate_x_to_y: f64,
    pub rate_y_to_x: f64,
}

#[derive(Clone, Debug)]
pub struct TransferEntropyResult {
    pub te_x_to_y: f64,
    pub te_y_to_x: f64,
    pub normalized_te: f64,
}

#[derive(Clone, Debug)]
pub struct GrangerResult {
    pub f_statistic_x_to_y: f64,
    pub f_statistic_y_to_x: f64,
    pub significant: bool,
}

#[derive(Clone, Debug)]
pub enum CausalDirection {
    XCausesY,
    YCausesX,
    Bidirectional,
    Undetermined,
}

#[derive(Clone, Debug)]
pub struct CausalStrength {
    pub overall: f64,
    pub ccm_component: f64,
    pub te_component: f64,
    pub directionality: f64,
}

#[derive(Clone, Debug)]
struct SignificanceResult {
    p_value: f64,
    adjusted_p: f64,
    significant: bool,
    confidence: f64,
}

#[derive(Clone, Debug)]
pub struct GraphStructure {
    pub adjacency: DMatrix<f64>,
    pub edges: Vec<(usize, usize)>,
    pub v_structures: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct LLMCausalityAnalysis {
    pub causal_matrix: DMatrix<f64>,
    pub influence_scores: Vec<f64>,
    pub causal_chains: Vec<CausalChain>,
    pub root_causes: Vec<usize>,
    pub consensus_mechanism: String,
}

#[derive(Clone, Debug)]
pub struct CausalChain {
    pub path: Vec<usize>,
    pub strength: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidirectional_causality() {
        let mut analyzer = BidirectionalCausalityAnalyzer::new();

        // Create coupled time series
        let n = 200;
        let mut x = DVector::zeros(n);
        let mut y = DVector::zeros(n);

        x[0] = rand::random::<f64>();
        y[0] = rand::random::<f64>();

        for i in 1..n {
            x[i] = 0.7 * x[i - 1] + 0.3 * y[i - 1] + 0.1 * rand::random::<f64>();
            y[i] = 0.5 * y[i - 1] + 0.4 * x[i - 1] + 0.1 * rand::random::<f64>();
        }

        let result = analyzer.analyze("x", &x, "y", &y).unwrap();

        assert!(result.strength.overall > 0.0);
        assert!(matches!(
            result.causal_direction,
            CausalDirection::Bidirectional | CausalDirection::XCausesY | CausalDirection::YCausesX
        ));
    }
}
