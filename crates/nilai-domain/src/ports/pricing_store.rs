use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::ModelName;
use crate::usage::LLMPriceConfig;

#[async_trait]
pub trait PricingStore: Send + Sync {
    async fn get_price(&self, model_name: &ModelName) -> NilaiResult<LLMPriceConfig>;
    async fn get_all_prices(&self) -> NilaiResult<HashMap<String, LLMPriceConfig>>;
    async fn set_price(&self, model_name: &ModelName, config: &LLMPriceConfig) -> NilaiResult<()>;
    async fn delete_price(&self, model_name: &ModelName) -> NilaiResult<bool>;
}
