use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use std::time::Duration;
use tokio::sync::watch;
use tracing;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::model::ModelEndpoint;

/// Run a keep-alive loop that refreshes the TTL for a registered model.
/// Stops when the shutdown receiver signals.
pub async fn keep_alive(
    conn: ConnectionManager,
    key: String,
    model_endpoint: &ModelEndpoint,
    lease_ttl: u64,
    mut shutdown: watch::Receiver<bool>,
) -> NilaiResult<()> {
    let model_json = serde_json::to_string(model_endpoint)
        .map_err(|e| NilaiError::Internal(format!("Serialization error: {}", e)))?;

    let refresh_interval = Duration::from_secs(lease_ttl / 2);
    let mut conn = conn;

    loop {
        tokio::select! {
            _ = tokio::time::sleep(refresh_interval) => {
                match refresh_ttl(&mut conn, &key, &model_json, lease_ttl).await {
                    Ok(()) => {
                        tracing::debug!("Refreshed TTL for {}", key);
                    }
                    Err(e) => {
                        tracing::error!("Failed to refresh TTL for {}: {}", key, e);
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("Keep-alive shutting down for {}", key);
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn refresh_ttl(
    conn: &mut ConnectionManager,
    key: &str,
    model_json: &str,
    lease_ttl: u64,
) -> NilaiResult<()> {
    let () = conn
        .set_ex(key, model_json, lease_ttl)
        .await
        .map_err(|e| NilaiError::Internal(format!("Redis SETEX error: {}", e)))?;
    Ok(())
}
