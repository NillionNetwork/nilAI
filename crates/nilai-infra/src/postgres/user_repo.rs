use async_trait::async_trait;
use sqlx::PgPool;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::UserId;
use nilai_domain::ports::UserStore;
use nilai_domain::user::{RateLimits, UserData};

pub struct PgUserRepo {
    pool: PgPool,
}

impl PgUserRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl UserStore for PgUserRepo {
    async fn get_user(&self, user_id: &UserId) -> NilaiResult<Option<UserData>> {
        let row = sqlx::query_as::<_, (String, Option<serde_json::Value>)>(
            "SELECT user_id, rate_limits FROM users WHERE user_id = $1",
        )
        .bind(user_id.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| NilaiError::Internal(format!("DB error: {}", e)))?;

        Ok(row.map(|(uid, limits)| {
            let rate_limits = limits
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            UserData {
                user_id: UserId::new(uid),
                rate_limits,
            }
        }))
    }

    async fn insert_user(&self, user_id: &UserId) -> NilaiResult<UserData> {
        sqlx::query(
            "INSERT INTO users (user_id, rate_limits) VALUES ($1, $2) ON CONFLICT (user_id) DO NOTHING",
        )
        .bind(user_id.as_str())
        .bind(serde_json::json!({}))
        .execute(&self.pool)
        .await
        .map_err(|e| NilaiError::Internal(format!("DB error: {}", e)))?;

        Ok(UserData {
            user_id: user_id.clone(),
            rate_limits: RateLimits::default(),
        })
    }

    async fn update_rate_limits(&self, user_id: &UserId, limits: &RateLimits) -> NilaiResult<()> {
        let limits_json = serde_json::to_value(limits)
            .map_err(|e| NilaiError::Internal(format!("Serialization error: {}", e)))?;

        sqlx::query("UPDATE users SET rate_limits = $1 WHERE user_id = $2")
            .bind(limits_json)
            .bind(user_id.as_str())
            .execute(&self.pool)
            .await
            .map_err(|e| NilaiError::Internal(format!("DB error: {}", e)))?;

        Ok(())
    }
}
