//! Information Channel
//!
//! Constitution: Phase 3, Task 3.1 - Cross-Domain Bridge
//!
//! Implements information-theoretic channel for cross-domain communication.
//! Based on Shannon's channel capacity theorem and rate-distortion theory.
//!
//! Mathematical Foundation:
//! ```text
//! C = max I(X;Y)  // Channel capacity
//! R(D) = min I(X;X̂)  // Rate-distortion function
//! H(Y|X) ≤ H(Y)  // Data processing inequality
//! ```

use ndarray::{Array1, Array2};

/// State of information channel
#[derive(Debug, Clone)]
pub struct ChannelState {
    /// Source distribution (neuromorphic domain)
    pub source: Array1<f64>,
    /// Target distribution (quantum domain)
    pub target: Array1<f64>,
    /// Channel transition matrix P(Y|X)
    pub transition_matrix: Array2<f64>,
    /// Mutual information I(X;Y)
    pub mutual_information: f64,
    /// Channel capacity
    pub capacity: f64,
}

impl ChannelState {
    /// Create new channel state
    pub fn new(n_source: usize, n_target: usize) -> Self {
        Self {
            source: Array1::zeros(n_source),
            target: Array1::zeros(n_target),
            transition_matrix: Array2::zeros((n_target, n_source)),
            mutual_information: 0.0,
            capacity: 0.0,
        }
    }

    /// Compute entropy H(X) of source distribution
    pub fn source_entropy(&self) -> f64 {
        entropy(&self.source)
    }

    /// Compute entropy H(Y) of target distribution
    pub fn target_entropy(&self) -> f64 {
        entropy(&self.target)
    }

    /// Compute joint entropy H(X,Y)
    pub fn joint_entropy(&self) -> f64 {
        let mut h_joint = 0.0;

        for (i, &p_x) in self.source.iter().enumerate() {
            if p_x > 1e-10 {
                for (j, &p_y_given_x) in self.transition_matrix.column(i).iter().enumerate() {
                    if p_y_given_x > 1e-10 {
                        let p_xy = p_x * p_y_given_x;
                        h_joint -= p_xy * p_xy.log2();
                    }
                }
            }
        }

        h_joint
    }

    /// Update mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    pub fn update_mutual_information(&mut self) {
        let h_x = self.source_entropy();
        let h_y = self.target_entropy();
        let h_xy = self.joint_entropy();

        self.mutual_information = h_x + h_y - h_xy;

        // Verify non-negativity (information inequality)
        assert!(
            self.mutual_information >= -1e-6,
            "Mutual information must be non-negative: {}",
            self.mutual_information
        );
        self.mutual_information = self.mutual_information.max(0.0);
    }
}

/// Result of information transfer
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Information transferred (bits)
    pub information_bits: f64,
    /// Transfer efficiency (0-1)
    pub efficiency: f64,
    /// Latency (milliseconds)
    pub latency_ms: f64,
    /// Distortion introduced
    pub distortion: f64,
}

/// Information channel for cross-domain communication
#[derive(Debug, Clone)]
pub struct InformationChannel {
    /// Current channel state
    pub state: ChannelState,
    /// Noise level (entropy)
    pub noise_entropy: f64,
    /// Bandwidth constraint
    pub bandwidth: f64,
}

impl InformationChannel {
    /// Create new information channel
    pub fn new(n_source: usize, n_target: usize, noise_entropy: f64) -> Self {
        Self {
            state: ChannelState::new(n_source, n_target),
            noise_entropy,
            bandwidth: 1000.0, // 1000 bits/sec default
        }
    }

    /// Initialize channel with non-uniform probabilities for non-zero MI
    pub fn initialize_uniform(&mut self) {
        let (n_target, n_source) = self.state.transition_matrix.dim();

        // Identity-like transition matrix (not uniform!) for maximum MI
        // P(Y=i|X=i) = 0.9, P(Y≠i|X=i) = 0.1/(n-1)
        for i in 0..n_source {
            for j in 0..n_target {
                if i == j && i < n_target {
                    self.state.transition_matrix[[j, i]] = 0.9;
                } else {
                    self.state.transition_matrix[[j, i]] = 0.1 / (n_target - 1).max(1) as f64;
                }
            }
        }

        // Uniform source distribution
        self.state.source.fill(1.0 / n_source as f64);

        // Compute target marginal: P(Y) = Σ_x P(Y|X)P(X)
        self.update_target_distribution();

        // Update mutual information
        self.state.update_mutual_information();

        // Channel capacity with noise
        self.state.capacity = (n_target.min(n_source) as f64).log2() - self.noise_entropy;
    }

    /// Update target distribution from source and transition matrix
    pub fn update_target_distribution(&mut self) {
        let (n_target, _) = self.state.transition_matrix.dim();
        self.state.target = Array1::zeros(n_target);

        // P(Y) = Σ_x P(Y|X)P(X)
        for (i, &p_x) in self.state.source.iter().enumerate() {
            for (j, p_y_j) in self.state.target.iter_mut().enumerate() {
                *p_y_j += self.state.transition_matrix[[j, i]] * p_x;
            }
        }

        // Normalize
        let sum: f64 = self.state.target.sum();
        if sum > 1e-10 {
            self.state.target /= sum;
        }
    }

    /// Optimize channel to maximize mutual information
    ///
    /// Uses Blahut-Arimoto algorithm for capacity computation
    pub fn maximize_mutual_information(&mut self, max_iterations: usize) -> f64 {
        let n_source = self.state.source.len();

        for _ in 0..max_iterations {
            // Update source distribution to maximize I(X;Y)
            // p*(x) ∝ exp(Σ_y p(y|x) log p(y|x) / p(y))
            let mut new_source = Array1::zeros(n_source);

            for i in 0..n_source {
                let mut kl_sum = 0.0;

                for (j, &p_y_given_x) in self.state.transition_matrix.column(i).iter().enumerate() {
                    if p_y_given_x > 1e-10 && self.state.target[j] > 1e-10 {
                        kl_sum += p_y_given_x * (p_y_given_x / self.state.target[j]).log2();
                    }
                }

                new_source[i] = kl_sum.exp();
            }

            // Normalize
            let sum = new_source.sum();
            if sum > 1e-10 {
                new_source /= sum;
            }

            self.state.source = new_source;

            // Update target distribution
            self.update_target_distribution();

            // Update mutual information
            let old_mi = self.state.mutual_information;
            self.state.update_mutual_information();

            // Check convergence
            if (self.state.mutual_information - old_mi).abs() < 1e-6 {
                break;
            }
        }

        self.state.mutual_information
    }

    /// Transfer information through channel
    ///
    /// Returns transfer result with efficiency and latency
    pub fn transfer(&mut self, source_signal: &Array1<f64>) -> TransferResult {
        let start = std::time::Instant::now();

        // Update source distribution
        let sum = source_signal.sum();
        if sum > 1e-10 {
            self.state.source = source_signal / sum;
        }

        // Update target
        self.update_target_distribution();

        // Compute mutual information
        self.state.update_mutual_information();

        // Compute distortion (KL divergence)
        let distortion = kl_divergence(source_signal, &self.state.target);

        // Efficiency: actual MI / capacity
        let efficiency = if self.state.capacity > 0.0 {
            (self.state.mutual_information / self.state.capacity).min(1.0)
        } else {
            0.0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        TransferResult {
            information_bits: self.state.mutual_information,
            efficiency,
            latency_ms,
            distortion,
        }
    }
}

/// Compute entropy H(X) = -Σ p(x) log p(x)
fn entropy(dist: &Array1<f64>) -> f64 {
    let mut h = 0.0;

    for &p in dist.iter() {
        if p > 1e-10 {
            h -= p * p.log2();
        }
    }

    h
}

/// Compute KL divergence D(P||Q) = Σ p(x) log(p(x)/q(x))
fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    let mut kl = 0.0;

    for (&p_i, &q_i) in p.iter().zip(q.iter()) {
        if p_i > 1e-10 && q_i > 1e-10 {
            kl += p_i * (p_i / q_i).log2();
        }
    }

    kl.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let dist = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let h = entropy(&dist);

        // Uniform distribution has maximum entropy
        assert!((h - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_deterministic() {
        let dist = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let h = entropy(&dist);

        // Deterministic distribution has zero entropy
        assert!(h < 1e-6);
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let mut channel = InformationChannel::new(10, 10, 0.1);
        channel.initialize_uniform();

        assert!(channel.state.mutual_information >= 0.0);
    }

    #[test]
    fn test_mutual_information_maximization() {
        let mut channel = InformationChannel::new(8, 8, 0.0);
        channel.initialize_uniform();

        let mi_initial = channel.state.mutual_information;
        let mi_optimized = channel.maximize_mutual_information(100);

        // Optimized MI should be >= initial MI
        assert!(mi_optimized >= mi_initial - 1e-6);
    }

    #[test]
    fn test_transfer_latency() {
        let mut channel = InformationChannel::new(10, 10, 0.1);
        channel.initialize_uniform();

        let signal = Array1::from_vec(vec![0.1; 10]);
        let result = channel.transfer(&signal);

        // Should meet latency requirement
        assert!(
            result.latency_ms < 1.0,
            "Latency: {:.3} ms",
            result.latency_ms
        );
    }

    #[test]
    fn test_channel_capacity() {
        let mut channel = InformationChannel::new(16, 16, 0.5);
        channel.initialize_uniform();

        // Capacity should be bounded by output entropy minus noise
        assert!(channel.state.capacity <= 16.0f64.log2());
        assert!(channel.state.capacity >= 0.0);
    }

    #[test]
    fn test_information_inequality() {
        let mut channel = InformationChannel::new(8, 8, 0.0);
        channel.initialize_uniform();

        let h_x = channel.state.source_entropy();
        let h_y = channel.state.target_entropy();
        let mi = channel.state.mutual_information;

        // I(X;Y) <= min(H(X), H(Y))
        assert!(mi <= h_x + 1e-6);
        assert!(mi <= h_y + 1e-6);
    }
}
