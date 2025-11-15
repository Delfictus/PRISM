//! LLM Transfer Entropy with Multi-lag Analysis
//!
//! Mission Charlie: Task 3.2
//!
//! CRITICAL: Article III compliance - REAL TE (not placeholder)

use anyhow::Result;
use ndarray::Array2;

use super::TextToTimeSeriesConverter;
use crate::information_theory::transfer_entropy::TransferEntropy;

/// LLM Causal Analyzer
///
/// Computes REAL transfer entropy between LLM outputs (Article III)
pub struct LLMCausalAnalyzer {
    te_calculator: TransferEntropy,
    text_converter: TextToTimeSeriesConverter,
}

impl LLMCausalAnalyzer {
    pub fn new() -> Self {
        Self {
            te_calculator: TransferEntropy::new(3, 3, 1),
            text_converter: TextToTimeSeriesConverter::new(5),
        }
    }

    /// Compute transfer entropy between ALL LLM pairs
    ///
    /// Article III MANDATORY: Must use real TE, not placeholder
    pub fn compute_llm_causality(&self, llm_texts: &[String]) -> Result<Array2<f64>> {
        let n = llm_texts.len();
        let mut te_matrix = Array2::zeros((n, n));

        // Convert to time series
        let time_series: Vec<_> = llm_texts
            .iter()
            .map(|t| self.text_converter.convert(t))
            .collect::<Result<Vec<_>>>()?;

        // Compute TE for all pairs
        for i in 0..n {
            for j in 0..n {
                if i != j && time_series[i].len() >= 20 && time_series[j].len() >= 20 {
                    let te_result = self
                        .te_calculator
                        .calculate(&time_series[i], &time_series[j]);
                    te_matrix[[i, j]] = te_result.effective_te;
                }
            }
        }

        Ok(te_matrix)
    }

    /// Identify dominant LLM (highest outgoing TE)
    pub fn find_dominant_llm(&self, te_matrix: &Array2<f64>) -> usize {
        let outgoing_te: Vec<f64> = (0..te_matrix.nrows())
            .map(|i| te_matrix.row(i).sum())
            .collect();

        outgoing_te
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
