//! Adaptive State Indexing for FluxNet RL
//!
//! Implements percentile-aware state quantization to improve Q-table distribution.
//! Instead of linear binning (0-255 → hash), tracks empirical distributions of
//! each state feature and maps bins to percentiles before indexing.
//!
//! # Motivation
//!
//! Linear hashing causes uneven Q-table coverage when state features have skewed
//! distributions. For example:
//! - Chromatic numbers cluster near target (e.g., 83-95 for DSJC1000.5)
//! - Conflicts are log-distributed (1, 10, 100, 1000...)
//! - GPU utilization often bimodal (idle or saturated)
//!
//! Percentile-based binning spreads states more evenly across the 16K table.
//!
//! # Architecture
//!
//! ```text
//! Raw State → AdaptiveStateIndexer → Percentile Bins → Hash → Index
//!   (u8)        (sliding histograms)      (0-15)       (u64)  (0-16383)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Maximum samples per feature histogram (2000 samples ~= 40 runs × 50 temps)
const HISTOGRAM_WINDOW: usize = 2000;

/// Minimum samples before adaptive indexing activates
const MIN_SAMPLES_FOR_ADAPTIVE: usize = 100;

/// Adaptive state quantizer with per-feature histograms
///
/// Tracks empirical distributions of state features using sliding windows,
/// computes percentiles on demand, and maps raw bins to normalized buckets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStateIndexer {
    /// Sliding histogram for chromatic_bin (window size 2000)
    chromatic_hist: VecDeque<u8>,

    /// Sliding histogram for conflicts_bin
    conflicts_hist: VecDeque<u8>,

    /// Sliding histogram for iteration_bin
    iteration_hist: VecDeque<u8>,

    /// Sliding histogram for gpu_util_bin
    gpu_util_hist: VecDeque<u8>,

    /// Phase-specific metric histograms (6 metrics × 4 bits each)
    phase_metric_hists: [VecDeque<u8>; 6],

    /// Cached percentile breakpoints (recomputed every N samples)
    chromatic_breakpoints: Option<[u8; 16]>,
    conflicts_breakpoints: Option<[u8; 16]>,
    iteration_breakpoints: Option<[u8; 16]>,
    gpu_util_breakpoints: Option<[u8; 16]>,
    phase_breakpoints: [Option<[u8; 16]>; 6],

    /// Samples since last breakpoint recomputation
    samples_since_recompute: usize,

    /// Recompute breakpoints every N samples (lazy update)
    recompute_interval: usize,

    /// Total samples observed
    total_samples: usize,
}

impl Default for AdaptiveStateIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveStateIndexer {
    /// Create new adaptive indexer with empty histograms
    pub fn new() -> Self {
        Self {
            chromatic_hist: VecDeque::with_capacity(HISTOGRAM_WINDOW),
            conflicts_hist: VecDeque::with_capacity(HISTOGRAM_WINDOW),
            iteration_hist: VecDeque::with_capacity(HISTOGRAM_WINDOW),
            gpu_util_hist: VecDeque::with_capacity(HISTOGRAM_WINDOW),
            phase_metric_hists: [
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
                VecDeque::with_capacity(HISTOGRAM_WINDOW),
            ],
            chromatic_breakpoints: None,
            conflicts_breakpoints: None,
            iteration_breakpoints: None,
            gpu_util_breakpoints: None,
            phase_breakpoints: [None, None, None, None, None, None],
            samples_since_recompute: 0,
            recompute_interval: 50, // Recompute every 50 samples
            total_samples: 0,
        }
    }

    /// Update histograms with new state observation
    ///
    /// Call this from MultiPhaseRLController::update() after each experience
    pub fn observe(
        &mut self,
        chromatic_bin: u8,
        conflicts_bin: u8,
        iteration_bin: u8,
        gpu_util_bin: u8,
        phase_metrics: [u8; 6],
    ) {
        // Update global histograms
        Self::push_histogram(&mut self.chromatic_hist, chromatic_bin);
        Self::push_histogram(&mut self.conflicts_hist, conflicts_bin);
        Self::push_histogram(&mut self.iteration_hist, iteration_bin);
        Self::push_histogram(&mut self.gpu_util_hist, gpu_util_bin);

        // Update phase-specific histograms
        for (i, &metric) in phase_metrics.iter().enumerate() {
            Self::push_histogram(&mut self.phase_metric_hists[i], metric);
        }

        self.total_samples += 1;
        self.samples_since_recompute += 1;

        // Lazily recompute breakpoints every N samples
        if self.samples_since_recompute >= self.recompute_interval {
            self.recompute_breakpoints();
            self.samples_since_recompute = 0;
        }
    }

    /// Push value to histogram with eviction policy (FIFO window)
    fn push_histogram(hist: &mut VecDeque<u8>, value: u8) {
        if hist.len() >= HISTOGRAM_WINDOW {
            hist.pop_front();
        }
        hist.push_back(value);
    }

    /// Compute percentile of a value within a histogram
    ///
    /// Returns value in [0.0, 1.0] representing the percentile rank
    pub fn compute_percentile(hist: &VecDeque<u8>, value: u8) -> f32 {
        if hist.is_empty() {
            return 0.5; // Default to median if no data
        }

        let count_below = hist.iter().filter(|&&x| x < value).count();
        let count_equal = hist.iter().filter(|&&x| x == value).count();

        // Interpolated percentile rank
        let rank = count_below as f32 + (count_equal as f32 / 2.0);
        rank / hist.len() as f32
    }

    /// Recompute percentile breakpoints for 16 buckets (0-15)
    ///
    /// For each histogram, find the 16 quantiles (6.25%, 12.5%, ..., 93.75%)
    fn recompute_breakpoints(&mut self) {
        if self.total_samples < MIN_SAMPLES_FOR_ADAPTIVE {
            return; // Not enough data yet
        }

        self.chromatic_breakpoints = Some(Self::compute_16_breakpoints(&self.chromatic_hist));
        self.conflicts_breakpoints = Some(Self::compute_16_breakpoints(&self.conflicts_hist));
        self.iteration_breakpoints = Some(Self::compute_16_breakpoints(&self.iteration_hist));
        self.gpu_util_breakpoints = Some(Self::compute_16_breakpoints(&self.gpu_util_hist));

        for i in 0..6 {
            self.phase_breakpoints[i] =
                Some(Self::compute_16_breakpoints(&self.phase_metric_hists[i]));
        }
    }

    /// Compute 16 breakpoints (quantiles) from histogram
    ///
    /// Returns array where breakpoints[i] is the value at quantile (i+1)/16
    fn compute_16_breakpoints(hist: &VecDeque<u8>) -> [u8; 16] {
        if hist.is_empty() {
            return [0; 16]; // Default to zeros
        }

        let mut sorted: Vec<u8> = hist.iter().copied().collect();
        sorted.sort_unstable();

        let n = sorted.len();
        let mut breakpoints = [0u8; 16];

        for i in 0..16 {
            // Quantile at (i+1)/16
            let quantile = (i + 1) as f32 / 16.0;
            let idx = ((n as f32 * quantile) as usize).saturating_sub(1).min(n - 1);
            breakpoints[i] = sorted[idx];
        }

        breakpoints
    }

    /// Map raw bin value to 4-bit bucket (0-15) using percentile breakpoints
    ///
    /// Returns bucket index based on learned distribution
    pub fn quantize_to_4bit(&self, hist: &VecDeque<u8>, breakpoints: Option<[u8; 16]>, value: u8) -> u8 {
        match breakpoints {
            Some(bp) => {
                // Find which bucket this value falls into
                for (i, &threshold) in bp.iter().enumerate() {
                    if value <= threshold {
                        return i as u8;
                    }
                }
                15 // Max bucket
            }
            None => {
                // Fallback to linear quantization (value / 16)
                (value / 16).min(15)
            }
        }
    }

    /// Quantize chromatic bin to 4-bit bucket
    pub fn quantize_chromatic(&self, value: u8) -> u8 {
        self.quantize_to_4bit(&self.chromatic_hist, self.chromatic_breakpoints, value)
    }

    /// Quantize conflicts bin to 4-bit bucket
    pub fn quantize_conflicts(&self, value: u8) -> u8 {
        self.quantize_to_4bit(&self.conflicts_hist, self.conflicts_breakpoints, value)
    }

    /// Quantize iteration bin to 4-bit bucket
    pub fn quantize_iteration(&self, value: u8) -> u8 {
        self.quantize_to_4bit(&self.iteration_hist, self.iteration_breakpoints, value)
    }

    /// Quantize GPU util bin to 4-bit bucket
    pub fn quantize_gpu_util(&self, value: u8) -> u8 {
        self.quantize_to_4bit(&self.gpu_util_hist, self.gpu_util_breakpoints, value)
    }

    /// Quantize phase metric bin to 4-bit bucket
    pub fn quantize_phase_metric(&self, metric_idx: usize, value: u8) -> u8 {
        if metric_idx < 6 {
            self.quantize_to_4bit(
                &self.phase_metric_hists[metric_idx],
                self.phase_breakpoints[metric_idx],
                value,
            )
        } else {
            (value / 16).min(15) // Fallback
        }
    }

    /// Check if adaptive indexing is ready (enough samples collected)
    pub fn is_ready(&self) -> bool {
        self.total_samples >= MIN_SAMPLES_FOR_ADAPTIVE
            && self.chromatic_breakpoints.is_some()
    }

    /// Get total samples observed
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Export learned breakpoints for debugging
    pub fn learned_bins(&self) -> AdaptiveBinsSnapshot {
        AdaptiveBinsSnapshot {
            chromatic: self.chromatic_breakpoints,
            conflicts: self.conflicts_breakpoints,
            iteration: self.iteration_breakpoints,
            gpu_util: self.gpu_util_breakpoints,
            phase_metrics: self.phase_breakpoints,
            total_samples: self.total_samples,
        }
    }

    /// Get histogram stats for observability
    pub fn stats(&self) -> AdaptiveIndexStats {
        AdaptiveIndexStats {
            total_samples: self.total_samples,
            chromatic_hist_size: self.chromatic_hist.len(),
            conflicts_hist_size: self.conflicts_hist.len(),
            adaptive_ready: self.is_ready(),
            samples_until_recompute: self.recompute_interval.saturating_sub(self.samples_since_recompute),
        }
    }

    /// Save adaptive indexer to binary file
    ///
    /// Serializes the entire state including histograms and breakpoints.
    /// MUST be saved alongside Q-table to ensure index consistency.
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load adaptive indexer from binary file
    ///
    /// Restores histograms and learned percentile breakpoints.
    /// MUST be loaded with matching Q-table to ensure correct state indices.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let indexer: AdaptiveStateIndexer = bincode::deserialize(&bytes)?;
        Ok(indexer)
    }
}

/// Snapshot of learned quantile breakpoints (for debugging/telemetry)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBinsSnapshot {
    pub chromatic: Option<[u8; 16]>,
    pub conflicts: Option<[u8; 16]>,
    pub iteration: Option<[u8; 16]>,
    pub gpu_util: Option<[u8; 16]>,
    pub phase_metrics: [Option<[u8; 16]>; 6],
    pub total_samples: usize,
}

/// Runtime statistics for adaptive indexer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndexStats {
    pub total_samples: usize,
    pub chromatic_hist_size: usize,
    pub conflicts_hist_size: usize,
    pub adaptive_ready: bool,
    pub samples_until_recompute: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_window_eviction() {
        let mut indexer = AdaptiveStateIndexer::new();

        // Fill beyond window size
        for i in 0..(HISTOGRAM_WINDOW + 100) {
            indexer.observe((i % 256) as u8, 0, 0, 0, [0; 6]);
        }

        assert_eq!(indexer.chromatic_hist.len(), HISTOGRAM_WINDOW);
        assert_eq!(indexer.total_samples, HISTOGRAM_WINDOW + 100);
    }

    #[test]
    fn test_percentile_computation() {
        let mut hist = VecDeque::new();
        hist.extend(vec![10, 20, 30, 40, 50]);

        assert_eq!(AdaptiveStateIndexer::compute_percentile(&hist, 10), 0.1);
        assert_eq!(AdaptiveStateIndexer::compute_percentile(&hist, 30), 0.5);
        assert_eq!(AdaptiveStateIndexer::compute_percentile(&hist, 50), 0.9);
    }

    #[test]
    fn test_adaptive_ready_threshold() {
        let mut indexer = AdaptiveStateIndexer::new();

        assert!(!indexer.is_ready());

        // Observe MIN_SAMPLES_FOR_ADAPTIVE samples
        for _ in 0..MIN_SAMPLES_FOR_ADAPTIVE {
            indexer.observe(100, 50, 128, 200, [5, 10, 3, 8, 12, 7]);
        }

        // Should trigger recomputation
        assert!(indexer.is_ready());
        assert!(indexer.chromatic_breakpoints.is_some());
    }

    #[test]
    fn test_breakpoint_distribution() {
        let mut indexer = AdaptiveStateIndexer::new();

        // Create skewed distribution (clustered around 80-100)
        for _ in 0..200 {
            let value = 80 + (rand::random::<u8>() % 20); // 80-99
            indexer.observe(value, 0, 0, 0, [0; 6]);
        }

        indexer.recompute_breakpoints();
        let breakpoints = indexer.chromatic_breakpoints.unwrap();

        // All breakpoints should be in range [80, 99]
        assert!(breakpoints.iter().all(|&b| b >= 80 && b < 100));
    }

    #[test]
    fn test_quantize_fallback() {
        let indexer = AdaptiveStateIndexer::new();
        let empty_hist = VecDeque::new();

        // Without breakpoints, should fall back to linear (value / 16)
        assert_eq!(indexer.quantize_to_4bit(&empty_hist, None, 0), 0);
        assert_eq!(indexer.quantize_to_4bit(&empty_hist, None, 16), 1);
        assert_eq!(indexer.quantize_to_4bit(&empty_hist, None, 240), 15);
    }
}
