use async_trait::async_trait;

use crate::error::NilaiResult;

#[async_trait]
pub trait RateLimiter: Send + Sync {
    async fn check_bucket(&self, key: &str, limit: u64, window_ms: u64) -> NilaiResult<()>;
    async fn increment_concurrent(&self, key: &str) -> NilaiResult<u64>;
    async fn decrement_concurrent(&self, key: &str) -> NilaiResult<()>;
}
