use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use std::collections::HashMap;
use tracing;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::{ModelId, ModelName};
use nilai_domain::model::ModelEndpoint;
use nilai_domain::ports::ModelRegistry;

const MODEL_PREFIX: &str = "/models";

pub struct RedisModelRegistry {
    conn: ConnectionManager,
    lease_ttl: u64,
}

impl RedisModelRegistry {
    pub async fn new(redis_url: &str, lease_ttl: u64) -> NilaiResult<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| NilaiError::Internal(format!("Redis client error: {}", e)))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis connection error: {}", e)))?;

        Ok(Self { conn, lease_ttl })
    }

    pub fn from_connection(conn: ConnectionManager, lease_ttl: u64) -> Self {
        Self { conn, lease_ttl }
    }

    pub fn connection(&self) -> ConnectionManager {
        self.conn.clone()
    }

    pub fn lease_ttl(&self) -> u64 {
        self.lease_ttl
    }
}

#[async_trait]
impl ModelRegistry for RedisModelRegistry {
    async fn get_model(&self, model_id: &ModelId) -> NilaiResult<Option<ModelEndpoint>> {
        let mut conn = self.conn.clone();
        let key = format!("{}/{}", MODEL_PREFIX, model_id.as_str());

        let value: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?;

        // Try without prefix if not found
        let value = match value {
            Some(v) => Some(v),
            None => conn
                .get(model_id.as_str())
                .await
                .map_err(|e| NilaiError::Internal(format!("Redis error: {}", e)))?,
        };

        match value {
            Some(json) => {
                let endpoint: ModelEndpoint = serde_json::from_str(&json)
                    .map_err(|e| NilaiError::Internal(format!("Deserialization error: {}", e)))?;
                Ok(Some(endpoint))
            }
            None => Ok(None),
        }
    }

    async fn discover_models(
        &self,
        name: Option<&ModelName>,
        feature: Option<&str>,
    ) -> NilaiResult<HashMap<String, ModelEndpoint>> {
        let mut conn = self.conn.clone();
        let pattern = format!("{}/*", MODEL_PREFIX);
        let mut discovered = HashMap::new();
        let mut cursor: u64 = 0;

        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await
                .map_err(|e| NilaiError::Internal(format!("Redis SCAN error: {}", e)))?;

            for key in keys {
                let value: Option<String> = conn
                    .get(&key)
                    .await
                    .map_err(|e| NilaiError::Internal(format!("Redis GET error: {}", e)))?;

                if let Some(json) = value {
                    match serde_json::from_str::<ModelEndpoint>(&json) {
                        Ok(endpoint) => {
                            // Apply name filter (case-insensitive substring match)
                            if let Some(name_filter) = name {
                                if !endpoint
                                    .metadata
                                    .name
                                    .to_lowercase()
                                    .contains(&name_filter.as_str().to_lowercase())
                                {
                                    continue;
                                }
                            }

                            // Apply feature filter
                            if let Some(feature_filter) = feature {
                                if !endpoint
                                    .metadata
                                    .supported_features
                                    .contains(&feature_filter.to_string())
                                {
                                    continue;
                                }
                            }

                            discovered.insert(endpoint.metadata.id.clone(), endpoint);
                        }
                        Err(e) => {
                            tracing::error!("Error parsing model from key {}: {}", key, e);
                        }
                    }
                }
            }

            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }

        Ok(discovered)
    }

    async fn register_model(&self, endpoint: &ModelEndpoint) -> NilaiResult<String> {
        let mut conn = self.conn.clone();
        let key = format!("{}/{}", MODEL_PREFIX, endpoint.metadata.id);
        let json = serde_json::to_string(endpoint)
            .map_err(|e| NilaiError::Internal(format!("Serialization error: {}", e)))?;

        let () = conn
            .set_ex(&key, &json, self.lease_ttl)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis SETEX error: {}", e)))?;

        Ok(key)
    }

    async fn unregister_model(&self, model_id: &ModelId) -> NilaiResult<()> {
        let mut conn = self.conn.clone();
        let key = format!("{}/{}", MODEL_PREFIX, model_id.as_str());
        let _: () = conn
            .del(&key)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis DEL error: {}", e)))?;
        Ok(())
    }
}
