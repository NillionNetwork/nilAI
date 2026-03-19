use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::config::DatabaseConfig;
use nilai_domain::error::{NilaiError, NilaiResult};

pub async fn create_pool(config: &DatabaseConfig) -> NilaiResult<PgPool> {
    PgPoolOptions::new()
        .max_connections(10)
        .connect(&config.connection_string())
        .await
        .map_err(|e| NilaiError::Internal(format!("Failed to connect to PostgreSQL: {}", e)))
}
