// Information Theory Module
// Constitution: Phase 1 Task 1.2

pub mod advanced_transfer_entropy;
pub mod transfer_entropy;

pub use transfer_entropy::{
    detect_causal_direction, CausalDirection, TransferEntropy, TransferEntropyResult,
};

pub use advanced_transfer_entropy::{
    AdvancedTransferEntropy, ConditionalTransferEntropy, KozachenkoLeonenkoEstimator,
    LocalTransferEntropy, PartialInformationDecomposition, RenyiTransferEntropy,
    SurrogateDataGenerator, SurrogateMethod, SymbolicTransferEntropy,
};

// GPU-accelerated transfer entropy
pub mod gpu_transfer_entropy;
pub use gpu_transfer_entropy::{GpuTransferEntropy, TransferEntropyGpuExt};

/// Information-theoretic measures for the Active Inference Platform
pub trait InformationMeasure {
    /// Calculate entropy H(X)
    fn entropy(&self) -> f64;

    /// Calculate mutual information I(X;Y)
    fn mutual_information(&self, other: &Self) -> f64;

    /// Calculate conditional entropy H(X|Y)
    fn conditional_entropy(&self, condition: &Self) -> f64;
}

/// Verify information-theoretic inequalities
pub fn verify_information_bounds(entropy: f64, mutual_info: f64) -> bool {
    // H(X) >= 0 (non-negativity of entropy)
    if entropy < 0.0 {
        return false;
    }

    // I(X;Y) >= 0 (non-negativity of mutual information)
    if mutual_info < 0.0 {
        return false;
    }

    // I(X;Y) <= min(H(X), H(Y)) (mutual information bound)
    // This would need both entropies to verify fully

    true
}
