use std::sync::Arc;

use nilai_domain::auth::AuthenticationInfo;
use nilai_domain::error::NilaiError;
use nilai_domain::ports::RateLimiter;

const MINUTE_MS: u64 = 60_000;
const HOUR_MS: u64 = 3_600_000;
const DAY_MS: u64 = 86_400_000;

/// Check all rate limit buckets for a user.
/// Called from route handlers after auth is resolved.
pub async fn check_rate_limits(
    rate_limiter: &dyn RateLimiter,
    auth_info: &AuthenticationInfo,
    web_search_enabled: bool,
    web_search_rps: Option<u64>,
) -> Result<(), NilaiError> {
    let user_id = auth_info.user.user_id.as_str();
    let limits = &auth_info.user.rate_limits;

    // Per-minute bucket
    if let Some(limit) = limits.user_rate_limit_minute {
        rate_limiter
            .check_bucket(&format!("minute:{}", user_id), limit, MINUTE_MS)
            .await?;
    }

    // Per-hour bucket
    if let Some(limit) = limits.user_rate_limit_hour {
        rate_limiter
            .check_bucket(&format!("hour:{}", user_id), limit, HOUR_MS)
            .await?;
    }

    // Per-day bucket
    if let Some(limit) = limits.user_rate_limit_day {
        rate_limiter
            .check_bucket(&format!("day:{}", user_id), limit, DAY_MS)
            .await?;
    }

    // For-good bucket (no expiry)
    if let Some(limit) = limits.user_rate_limit {
        rate_limiter
            .check_bucket(&format!("user:{}", user_id), limit, 0)
            .await?;
    }

    // Token-based rate limits (NUC mode)
    if let Some(ref token_limits) = auth_info.token_rate_limit {
        for limit in &token_limits.limits {
            if let Some(usage_limit) = limit.usage_limit {
                rate_limiter
                    .check_bucket(
                        &format!("token:{}", limit.signature),
                        usage_limit,
                        limit.ms_remaining(),
                    )
                    .await?;
            }
        }
    }

    // Web search rate limits
    if web_search_enabled {
        if let Some(limit) = limits.web_search_rate_limit_minute {
            rate_limiter
                .check_bucket(&format!("web_search_minute:{}", user_id), limit, MINUTE_MS)
                .await?;
        }
        if let Some(limit) = limits.web_search_rate_limit_hour {
            rate_limiter
                .check_bucket(&format!("web_search_hour:{}", user_id), limit, HOUR_MS)
                .await?;
        }
        if let Some(limit) = limits.web_search_rate_limit_day {
            rate_limiter
                .check_bucket(&format!("web_search_day:{}", user_id), limit, DAY_MS)
                .await?;
        }
        if let Some(rps) = web_search_rps {
            rate_limiter
                .check_bucket("web_search_rps", rps, 1000)
                .await?;
        }
        if let Some(limit) = limits.web_search_rate_limit {
            rate_limiter
                .check_bucket(&format!("web_search:{}", user_id), limit, 0)
                .await?;
        }
    }

    Ok(())
}

/// Guard for concurrent request tracking.
/// On drop, decrements the concurrent counter.
pub struct ConcurrentGuard {
    rate_limiter: Arc<dyn RateLimiter>,
    key: String,
}

impl ConcurrentGuard {
    /// Check and increment concurrent counter. Returns guard on success.
    pub async fn acquire(
        rate_limiter: Arc<dyn RateLimiter>,
        key: String,
        max_concurrent: u64,
    ) -> Result<Self, NilaiError> {
        let current = rate_limiter.increment_concurrent(&key).await?;
        if current > max_concurrent {
            // Exceeded, decrement and return error
            let _ = rate_limiter.decrement_concurrent(&key).await;
            return Err(NilaiError::RateLimited { retry_after_ms: 0 });
        }
        Ok(Self { rate_limiter, key })
    }
}

impl Drop for ConcurrentGuard {
    fn drop(&mut self) {
        let limiter = self.rate_limiter.clone();
        let key = self.key.clone();
        tokio::spawn(async move {
            let _ = limiter.decrement_concurrent(&key).await;
        });
    }
}
