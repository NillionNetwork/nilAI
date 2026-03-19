use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub web_searches: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMCost {
    pub prompt_tokens_price: f64,
    pub completion_tokens_price: f64,
    pub web_search_cost: f64,
}

impl LLMCost {
    /// Prices are per-million tokens in config, but stored normalized here.
    pub fn from_config(
        prompt_price_per_million: f64,
        completion_price_per_million: f64,
        web_search_cost: f64,
    ) -> Self {
        Self {
            prompt_tokens_price: prompt_price_per_million / 1_000_000.0,
            completion_tokens_price: completion_price_per_million / 1_000_000.0,
            web_search_cost,
        }
    }

    pub fn total_cost(&self, prompt_tokens: u64, completion_tokens: u64, web_searches: u64) -> f64 {
        self.prompt_tokens_price * prompt_tokens as f64
            + self.completion_tokens_price * completion_tokens as f64
            + self.web_search_cost * web_searches as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMPriceConfig {
    #[serde(default = "default_prompt_price")]
    pub prompt_tokens_price: f64,
    #[serde(default = "default_completion_price")]
    pub completion_tokens_price: f64,
    #[serde(default = "default_web_search_cost")]
    pub web_search_cost: f64,
}

fn default_prompt_price() -> f64 {
    2.0
}
fn default_completion_price() -> f64 {
    2.0
}
fn default_web_search_cost() -> f64 {
    0.05
}

impl Default for LLMPriceConfig {
    fn default() -> Self {
        Self {
            prompt_tokens_price: default_prompt_price(),
            completion_tokens_price: default_completion_price(),
            web_search_cost: default_web_search_cost(),
        }
    }
}

/// Query log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLog {
    pub user_id: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub web_searches: u64,
    pub lock_id: String,
}
