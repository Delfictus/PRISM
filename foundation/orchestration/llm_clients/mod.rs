//! LLM Client Infrastructure
//!
//! Production-grade API clients for multiple LLM providers:
//! - OpenAI GPT-4
//! - Anthropic Claude
//! - Google Gemini
//! - xAI Grok-4

pub mod openai_client;
pub mod claude_client;
pub mod gemini_client;
pub mod grok_client;
pub mod ensemble;

// Re-export primary types
pub use openai_client::{OpenAIClient, LLMResponse, Usage};
pub use claude_client::ClaudeClient;
pub use gemini_client::GeminiClient;
pub use grok_client::GrokClient;
pub use ensemble::{
    LLMOrchestrator,
    BanditLLMEnsemble,
    BayesianLLMEnsemble,
    BanditResponse,
    BayesianConsensusResponse,
};

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
