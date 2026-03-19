use async_trait::async_trait;
use redis::AsyncCommands;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::RateLimiter;

use crate::redis::pool::RedisPool;

pub struct RedisRateLimiter {
    pool: RedisPool,
}

impl RedisRateLimiter {
    pub fn new(pool: RedisPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl RateLimiter for RedisRateLimiter {
    async fn check_bucket(&self, key: &str, limit: u64, window_ms: u64) -> NilaiResult<()> {
        let mut conn = self.pool.connection();
        let expire: i64 = redis::cmd("EVALSHA")
            .arg(self.pool.rate_limit_sha())
            .arg(1)
            .arg(key)
            .arg(limit)
            .arg(window_ms)
            .query_async(&mut conn)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis rate limit error: {}", e)))?;

        if expire > 0 {
            Err(NilaiError::RateLimited {
                retry_after_ms: expire as u64,
            })
        } else {
            Ok(())
        }
    }

    async fn increment_concurrent(&self, key: &str) -> NilaiResult<u64> {
        let mut conn = self.pool.connection();
        let count: u64 = conn
            .incr(format!("concurrent:{}", key), 1u64)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;
        Ok(count)
    }

    async fn decrement_concurrent(&self, key: &str) -> NilaiResult<()> {
        let mut conn = self.pool.connection();
        let _: () = conn
            .decr(format!("concurrent:{}", key), 1u64)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;
        Ok(())
    }
}
