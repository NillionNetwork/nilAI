use std::collections::HashMap;

use async_trait::async_trait;
use redis::AsyncCommands;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::ModelName;
use nilai_domain::ports::PricingStore;
use nilai_domain::usage::LLMPriceConfig;

use crate::redis::pool::RedisPool;

const PRICING_PREFIX: &str = "nilai:pricing";
const PRICING_ALL_KEY: &str = "nilai:pricing:_all";

pub struct RedisPricingRepo {
    pool: RedisPool,
}

impl RedisPricingRepo {
    pub fn new(pool: RedisPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl PricingStore for RedisPricingRepo {
    async fn get_price(&self, model_name: &ModelName) -> NilaiResult<LLMPriceConfig> {
        let mut conn = self.pool.connection();
        let key = format!("{}:{}", PRICING_PREFIX, model_name.as_str());
        let value: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        match value {
            Some(json) => serde_json::from_str(&json)
                .map_err(|e| NilaiError::Internal(format!("Deserialization error: {}", e))),
            None => Ok(LLMPriceConfig::default()),
        }
    }

    async fn get_all_prices(&self) -> NilaiResult<HashMap<String, LLMPriceConfig>> {
        let mut conn = self.pool.connection();
        let all: HashMap<String, String> = conn
            .hgetall(PRICING_ALL_KEY)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        let mut result = HashMap::new();
        for (name, json) in all {
            if let Ok(config) = serde_json::from_str(&json) {
                result.insert(name, config);
            }
        }
        Ok(result)
    }

    async fn set_price(&self, model_name: &ModelName, config: &LLMPriceConfig) -> NilaiResult<()> {
        let mut conn = self.pool.connection();
        let json = serde_json::to_string(config)
            .map_err(|e| NilaiError::Internal(format!("Serialization error: {}", e)))?;

        let key = format!("{}:{}", PRICING_PREFIX, model_name.as_str());
        let _: () = conn
            .set(&key, &json)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        let _: () = conn
            .hset(PRICING_ALL_KEY, model_name.as_str(), &json)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        Ok(())
    }

    async fn delete_price(&self, model_name: &ModelName) -> NilaiResult<bool> {
        let mut conn = self.pool.connection();
        let key = format!("{}:{}", PRICING_PREFIX, model_name.as_str());

        let deleted: u64 = conn
            .del(&key)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        let _: () = conn
            .hdel(PRICING_ALL_KEY, model_name.as_str())
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        Ok(deleted > 0)
    }
}
