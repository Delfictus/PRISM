//! LLM Client Infrastructure
//!
//! Production-grade API clients for multiple LLM providers:
//! - OpenAI GPT-4
//! - Anthropic Claude
//! - Google Gemini
//! - xAI Grok-4

pub mod claude_client;
pub mod ensemble;
pub mod gemini_client;
pub mod grok_client;
pub mod openai_client;

// Re-export primary types
pub use claude_client::ClaudeClient;
pub use ensemble::{
    BanditLLMEnsemble, BanditResponse, BayesianConsensusResponse, BayesianLLMEnsemble,
    LLMOrchestrator,
};
pub use gemini_client::GeminiClient;
pub use grok_client::GrokClient;
pub use openai_client::{LLMResponse, OpenAIClient, Usage};

/// Unified LLM client trait
#[async_trait::async_trait]
pub trait LLMClient: Send + Sync {
    /// Generate response from prompt
    async fn generate(&self, prompt: &str, temperature: f32) -> anyhow::Result<LLMResponse>;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Get total cost (USD)
    fn get_total_cost(&self) -> f64;

    /// Get total tokens processed
    fn get_total_tokens(&self) -> usize;
}
