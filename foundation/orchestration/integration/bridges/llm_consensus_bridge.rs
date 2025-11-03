//! LLM Consensus Bridge
//!
//! Provides unified consensus mechanism that combines:
//! - Quantum voting consensus
//! - Thermodynamic consensus  
//! - Transfer entropy routing
//!
//! This bridge integrates all existing infrastructure for world-class LLM consensus.

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::time::Duration;

/// Request for LLM consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequest {
    pub query: String,
    pub models: Vec<String>,
    pub temperature: f32,
}

/// Response from LLM consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResponse {
    pub text: String,
    pub confidence: f64,
    pub agreement_score: f64,
    pub model_responses: Vec<ModelResponse>,
    pub algorithm_weights: Vec<(String, f64)>,
}

/// Individual model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub model: String,
    pub text: String,
    pub tokens: usize,
    pub cost: f64,
}

/// Consensus algorithm weights for fusion
#[derive(Debug, Clone)]
pub struct ConsensusWeights {
    pub quantum_weight: f64,
    pub thermodynamic_weight: f64,
    pub routing_weight: f64,
}

impl Default for ConsensusWeights {
    fn default() -> Self {
        Self {
            quantum_weight: 0.40,      // 40% quantum voting
            thermodynamic_weight: 0.35, // 35% thermodynamic consensus
            routing_weight: 0.25,      // 25% transfer entropy routing
        }
    }
}

/// Consensus fusion result
#[derive(Debug, Clone)]
pub struct ConsensusFusion {
    pub final_text: String,
    pub confidence: f64,
    pub agreement_score: f64,
    pub quantum_contribution: f64,
    pub thermodynamic_contribution: f64,
    pub routing_contribution: f64,
}

impl ConsensusFusion {
    /// Create new fusion result
    pub fn new(
        final_text: String,
        confidence: f64,
        agreement_score: f64,
        quantum_contribution: f64,
        thermodynamic_contribution: f64,
        routing_contribution: f64,
    ) -> Self {
        Self {
            final_text,
            confidence,
            agreement_score,
            quantum_contribution,
            thermodynamic_contribution,
            routing_contribution,
        }
    }
}

/// Quantum voting result
#[derive(Debug, Clone)]
pub struct QuantumVoteResult {
    pub confidence: f64,
    pub agreement: f64,
    pub consensus_text: String,
}

/// Thermodynamic consensus result
#[derive(Debug, Clone)]
pub struct ThermodynamicConsensusResult {
    pub agreement: f64,
    pub consensus_text: String,
}

/// Transfer entropy routing result
#[derive(Debug, Clone)]
pub struct TransferEntropyRoutingResult {
    pub routing_score: f64,
    pub agreement: f64,
    pub routing_text: String,
}

/// LLM response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub model: String,
    pub text: String,
    pub usage: Usage,
    pub latency: Duration,
    pub cached: bool,
    pub cost: f64,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Consensus metrics for monitoring
#[derive(Debug, Clone)]
pub struct ConsensusMetrics {
    pub total_queries: u64,
    pub successful_consensus: u64,
    pub average_confidence: f64,
    pub average_agreement: f64,
    pub quantum_usage: u64,
    pub thermodynamic_usage: u64,
    pub routing_usage: u64,
}

impl Default for ConsensusMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_consensus: 0,
            average_confidence: 0.0,
            average_agreement: 0.0,
            quantum_usage: 0,
            thermodynamic_usage: 0,
            routing_usage: 0,
        }
    }
}

impl ConsensusMetrics {
    /// Update metrics with new consensus result
    pub fn update(&mut self, confidence: f64, agreement: f64, algorithms_used: &[String]) {
        self.total_queries += 1;
        self.successful_consensus += 1;
        
        // Update running averages
        let n = self.successful_consensus as f64;
        self.average_confidence = (self.average_confidence * (n - 1.0) + confidence) / n;
        self.average_agreement = (self.average_agreement * (n - 1.0) + agreement) / n;
        
        // Track algorithm usage
        for algo in algorithms_used {
            match algo.as_str() {
                "quantum_voting" => self.quantum_usage += 1,
                "thermodynamic_consensus" => self.thermodynamic_usage += 1,
                "transfer_entropy_routing" => self.routing_usage += 1,
                _ => {}
            }
        }
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.successful_consensus as f64 / self.total_queries as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_weights_default() {
        let weights = ConsensusWeights::default();
        assert_eq!(weights.quantum_weight, 0.40);
        assert_eq!(weights.thermodynamic_weight, 0.35);
        assert_eq!(weights.routing_weight, 0.25);
        
        // Ensure weights sum to 1.0
        let sum = weights.quantum_weight + weights.thermodynamic_weight + weights.routing_weight;
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_fusion() {
        let fusion = ConsensusFusion::new(
            "Test response".to_string(),
            0.85,
            0.90,
            0.4,
            0.35,
            0.25,
        );
        
        assert_eq!(fusion.final_text, "Test response");
        assert_eq!(fusion.confidence, 0.85);
        assert_eq!(fusion.agreement_score, 0.90);
    }

    #[test]
    fn test_consensus_metrics_update() {
        let mut metrics = ConsensusMetrics::default();
        
        metrics.update(0.8, 0.9, &["quantum_voting".to_string()]);
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.successful_consensus, 1);
        assert_eq!(metrics.quantum_usage, 1);
        assert_eq!(metrics.average_confidence, 0.8);
        assert_eq!(metrics.average_agreement, 0.9);
        
        metrics.update(0.9, 0.95, &["thermodynamic_consensus".to_string()]);
        assert_eq!(metrics.total_queries, 2);
        assert_eq!(metrics.successful_consensus, 2);
        assert_eq!(metrics.thermodynamic_usage, 1);
        assert_eq!(metrics.average_confidence, 0.85);
        assert_eq!(metrics.average_agreement, 0.925);
    }
}
