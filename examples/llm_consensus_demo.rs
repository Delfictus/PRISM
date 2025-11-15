//! LLM Consensus Demo
//!
//! Demonstrates the world-class LLM consensus system that combines:
//! - Quantum voting consensus (40% weight)
//! - Thermodynamic consensus (35% weight)
//! - Transfer entropy routing (25% weight)

use anyhow::Result;
use prism_ai::foundation::orchestration::integration::bridges::{
    ConsensusRequest, ConsensusResponse,
};
use prism_ai::foundation::orchestration::integration::prism_ai_integration::PrismAIOrchestrator;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== PRISM-AI LLM Consensus Demo ===\n");

    // Initialize the orchestrator
    println!("Initializing PRISM-AI Orchestrator...");
    let orchestrator = PrismAIOrchestrator::new(Default::default()).await?;
    println!("✅ Orchestrator initialized successfully\n");

    // Example queries for consensus
    let queries = vec![
        "What is the future of artificial intelligence?",
        "Explain quantum computing in simple terms",
        "How can we solve climate change?",
        "What are the ethical implications of AI?",
    ];

    let models = vec!["gpt-4", "claude-3", "gemini-pro", "grok-2"];

    for (i, query) in queries.iter().enumerate() {
        println!("--- Query {}: {} ---", i + 1, query);

        // Get consensus from all models
        let consensus_response = orchestrator.llm_consensus(query, &models).await?;

        println!("📊 Consensus Results:");
        println!("   Confidence: {:.3}", consensus_response.confidence);
        println!("   Agreement: {:.3}", consensus_response.agreement_score);
        println!(
            "   Response Length: {} characters",
            consensus_response.text.len()
        );
        println!(
            "   Models Used: {}",
            consensus_response.model_responses.len()
        );

        println!("🔬 Algorithm Weights:");
        for (algorithm, weight) in &consensus_response.algorithm_weights {
            println!("   {}: {:.1}%", algorithm, weight * 100.0);
        }

        println!("🤖 Individual Model Responses:");
        for model_response in &consensus_response.model_responses {
            println!(
                "   {}: {} tokens, ${:.4} cost",
                model_response.model, model_response.tokens, model_response.cost
            );
        }

        println!("📝 Consensus Response:");
        println!("   {}", consensus_response.text);
        println!();
    }

    // Show system health
    let health = orchestrator.get_health_status();
    println!("🏥 System Health:");
    println!("   Overall Health: {:?}", health.overall_health);
    println!("   Total Queries: {}", health.metrics.total_queries);
    println!(
        "   GPU Accelerated Ops: {}",
        health.metrics.gpu_accelerated_ops
    );
    println!(
        "   System Health Score: {:.3}",
        health.metrics.system_health
    );

    println!("\n=== Demo Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_basic() {
        let orchestrator = PrismAIOrchestrator::new(Default::default()).await.unwrap();
        let models = vec!["gpt-4", "claude-3"];

        let response = orchestrator
            .llm_consensus("What is 2+2?", &models)
            .await
            .unwrap();

        assert!(!response.text.is_empty());
        assert!(response.confidence > 0.0);
        assert!(response.agreement_score > 0.0);
        assert_eq!(response.model_responses.len(), 2);
        assert_eq!(response.algorithm_weights.len(), 3);
    }

    #[tokio::test]
    async fn test_consensus_error_handling() {
        let orchestrator = PrismAIOrchestrator::new(Default::default()).await.unwrap();
        let invalid_models = vec!["nonexistent-model"];

        let result = orchestrator
            .llm_consensus("Test query", &invalid_models)
            .await;

        // Should handle errors gracefully
        assert!(result.is_err());
    }
}
