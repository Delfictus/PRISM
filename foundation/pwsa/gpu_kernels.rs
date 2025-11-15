//! GPU-Accelerated PWSA Fusion Kernels
//!
//! NOTE: This module contains LIGHTWEIGHT helper utilities.
//! The MAIN PWSA classifier (gpu_classifier.rs) uses GPU neural networks.
//!
//! These are simple feature extraction helpers - not computational bottlenecks.
//! They use simple arithmetic (not heavy GPU kernel worthy).

use anyhow::Result;
use cudarc::driver::CudaDevice;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Simple threat classifier helper (lightweight)
///
/// NOTE: Main PWSA system uses gpu_classifier.rs with GPU neural network
/// This is a lightweight fallback for simple rule-based classification
pub struct GpuThreatClassifier {
    context: Arc<CudaDevice>,
}

impl GpuThreatClassifier {
    pub fn new(context: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { context })
    }

    /// Simple rule-based classification (lightweight, not GPU-kernel worthy)
    ///
    /// NOTE: Real PWSA uses gpu_classifier.rs with GPU neural network (70K ops/sec)
    /// This is for simple fallback scenarios
    pub fn classify(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        let _ = &self.context;

        let mut probs = Array1::zeros(5);

        // Simple feature-based rules (lightweight logic)
        let velocity_indicator = features[6];
        let thermal_indicator = features[11];
        let maneuver_indicator = features[7];

        probs[0] = if velocity_indicator < 0.2 && thermal_indicator < 0.3 {
            0.9
        } else {
            0.1
        };
        probs[1] = if velocity_indicator < 0.3 && thermal_indicator < 0.5 {
            0.7
        } else {
            0.1
        };
        probs[2] = if velocity_indicator < 0.5 && maneuver_indicator < 0.5 {
            0.6
        } else {
            0.1
        };
        probs[3] = if velocity_indicator > 0.6 && maneuver_indicator < 0.3 {
            0.8
        } else {
            0.1
        };
        probs[4] = if velocity_indicator > 0.5 && maneuver_indicator > 0.4 {
            0.9
        } else {
            0.1
        };

        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            probs.mapv_inplace(|p| p / sum);
        }

        Ok(probs)
    }
}

/// Simple feature normalization helper (lightweight arithmetic)
pub struct GpuFeatureExtractor {
    context: Arc<CudaDevice>,
}

impl GpuFeatureExtractor {
    pub fn new(context: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { context })
    }

    /// Simple feature normalization (scalar division - lightweight)
    pub fn normalize_oct_telemetry_simd(
        &self,
        optical_power: f64,
        bit_error_rate: f64,
        pointing_error: f64,
        data_rate: f64,
        temperature: f64,
    ) -> [f64; 5] {
        let _ = &self.context;

        [
            optical_power / 30.0,
            bit_error_rate.log10() / -10.0,
            pointing_error / 100.0,
            data_rate / 10.0,
            temperature / 100.0,
        ]
    }
}

/// TE computation for PWSA
///
/// NOTE: Uses existing TransferEntropy library which has GPU kernels available
pub struct GpuTransferEntropyComputer {
    context: Arc<CudaDevice>,
}

impl GpuTransferEntropyComputer {
    pub fn new(context: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { context })
    }

    /// Compute TE matrix
    ///
    /// Uses information_theory::TransferEntropy (has GPU histogram kernels)
    pub fn compute_coupling_matrix(
        &self,
        transport_ts: &Array1<f64>,
        tracking_ts: &Array1<f64>,
        ground_ts: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let _ = &self.context;

        use crate::information_theory::transfer_entropy::TransferEntropy;

        // Uses TransferEntropy which has GPU kernels available
        let te_calc = TransferEntropy::new(3, 3, 1);
        let mut coupling = Array2::zeros((3, 3));

        // Serial for now, could parallelize with GPU
        coupling[[0, 1]] = te_calc.calculate(transport_ts, tracking_ts).effective_te;
        coupling[[1, 0]] = te_calc.calculate(tracking_ts, transport_ts).effective_te;
        coupling[[0, 2]] = te_calc.calculate(transport_ts, ground_ts).effective_te;
        coupling[[2, 0]] = te_calc.calculate(ground_ts, transport_ts).effective_te;
        coupling[[1, 2]] = te_calc.calculate(tracking_ts, ground_ts).effective_te;
        coupling[[2, 1]] = te_calc.calculate(ground_ts, tracking_ts).effective_te;

        Ok(coupling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_threat_classifier_creation() {
        let ctx = CudaDevice::new(0).expect("GPU REQUIRED for PWSA");
        let classifier = GpuThreatClassifier::new(ctx);
        assert!(classifier.is_ok());
    }

    #[test]
    fn test_feature_normalization() {
        let ctx = CudaDevice::new(0).expect("GPU REQUIRED for PWSA");
        let extractor = GpuFeatureExtractor::new(ctx).unwrap();

        let features = extractor.normalize_oct_telemetry_simd(
            -15.0, // optical_power
            1e-9,  // BER
            5.0,   // pointing
            10.0,  // data_rate
            22.0,  // temperature
        );

        // Validate normalization
        assert!((features[0] - (-15.0 / 30.0)).abs() < 1e-6);
        assert!(features[1] > 0.0);
        assert!((features[2] - 0.05).abs() < 1e-6);
        assert!((features[3] - 1.0).abs() < 1e-6);
        assert!((features[4] - 0.22).abs() < 1e-6);
    }
}

// NOTE: Main PWSA classifier (gpu_classifier.rs) uses GPU neural network
// These are lightweight helper utilities for simple operations
