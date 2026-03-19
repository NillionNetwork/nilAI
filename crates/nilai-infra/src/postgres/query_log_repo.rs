use async_trait::async_trait;
use sqlx::PgPool;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::UserId;
use nilai_domain::ports::QueryLogStore;
use nilai_domain::usage::{QueryLog, Usage};

pub struct PgQueryLogRepo {
    pool: PgPool,
}

impl PgQueryLogRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl QueryLogStore for PgQueryLogRepo {
    async fn log_query(&self, log: &QueryLog) -> NilaiResult<()> {
        sqlx::query(
            "INSERT INTO query_logs (user_id, model, prompt_tokens, completion_tokens, total_tokens, web_searches, lock_id) VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(&log.user_id)
        .bind(&log.model)
        .bind(log.prompt_tokens as i64)
        .bind(log.completion_tokens as i64)
        .bind(log.total_tokens as i64)
        .bind(log.web_searches as i64)
        .bind(&log.lock_id)
        .execute(&self.pool)
        .await
        .map_err(|e| NilaiError::Internal(format!("DB error: {}", e)))?;

        Ok(())
    }

    async fn get_user_usage(&self, user_id: &UserId) -> NilaiResult<Usage> {
        let row = sqlx::query_as::<_, (Option<i64>, Option<i64>, Option<i64>)>(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(total_tokens), 0) FROM query_logs WHERE user_id = $1",
        )
        .bind(user_id.as_str())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| NilaiError::Internal(format!("DB error: {}", e)))?;

        Ok(Usage {
            prompt_tokens: row.0.unwrap_or(0) as u64,
            completion_tokens: row.1.unwrap_or(0) as u64,
            total_tokens: row.2.unwrap_or(0) as u64,
        })
    }
}
