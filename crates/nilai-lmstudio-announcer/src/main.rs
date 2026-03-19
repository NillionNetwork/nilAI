//! LMStudio Model Announcer
//!
//! Connects to an LMStudio (or any OpenAI-compatible) API, discovers available
//! models, and registers them in Redis so the NilAI API can route requests.
//!
//! This is the Rust equivalent of the Python `lmstudio_announcer.py`.
//!
//! # Configuration (environment variables)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `LMSTUDIO_API_BASE` | `http://localhost:1234` | LMStudio API base URL |
//! | `LMSTUDIO_MODELS_ENDPOINT` | `/v1/models` | Endpoint to list models |
//! | `LMSTUDIO_REGISTRATION_URL` | same as API_BASE | URL registered in Redis (what NilAI calls) |
//! | `LMSTUDIO_MODEL_IDS` | *(auto-discover)* | Comma-separated model IDs to announce |
//! | `DISCOVERY_URL` | `redis://localhost:6379` | Redis URL for service discovery |
//! | `LMSTUDIO_LEASE_TTL` | `60` | Redis key TTL in seconds |
//! | `LMSTUDIO_TOOL_SUPPORT_DEFAULT` | `true` | Default tool support for models |
//! | `LMSTUDIO_MULTIMODAL_DEFAULT` | `true` | Default multimodal support for models |

use std::env;
use std::time::Duration;

use nilai_discovery::RedisModelRegistry;
use nilai_domain::model::{ModelEndpoint, ModelMetadata};
use nilai_domain::ports::ModelRegistry;
use tokio::sync::watch;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_bool(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "true" | "1" | "yes")
}

#[derive(serde::Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(serde::Deserialize)]
struct ModelEntry {
    id: String,
}

/// Fetch model IDs from an OpenAI-compatible /v1/models endpoint with retries.
async fn fetch_model_ids(
    api_base: &str,
    endpoint: &str,
    timeout_secs: f64,
    max_retries: u32,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs))
        .build()?;
    let url = format!("{}{}", api_base.trim_end_matches('/'), endpoint);

    for attempt in 1..=max_retries {
        tracing::info!(
            "Fetching models from {} (attempt {}/{})",
            url,
            attempt,
            max_retries
        );

        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let models: ModelsResponse = resp.json().await?;
                let ids: Vec<String> = models.data.into_iter().map(|m| m.id).collect();
                if ids.is_empty() {
                    tracing::warn!("Received empty model list, retrying");
                } else {
                    tracing::info!("Discovered models: {}", ids.join(", "));
                    return Ok(ids);
                }
            }
            Ok(resp) => {
                tracing::warn!("API returned status {}", resp.status());
            }
            Err(e) => {
                tracing::warn!("Failed to connect: {}", e);
            }
        }

        if attempt < max_retries {
            let delay = Duration::from_secs(2u64.pow(attempt.min(5)));
            tracing::info!("Retrying in {:?}", delay);
            tokio::time::sleep(delay).await;
        }
    }

    Err("Failed to discover models after max retries".into())
}

/// Register a single model and keep it alive until shutdown.
async fn announce_model(
    metadata: ModelMetadata,
    registration_url: String,
    discovery_url: String,
    lease_ttl: u64,
    mut shutdown: watch::Receiver<bool>,
) {
    let registry = match RedisModelRegistry::new(&discovery_url, lease_ttl).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                "Failed to connect to Redis for model {}: {}",
                metadata.name,
                e
            );
            return;
        }
    };

    let endpoint = ModelEndpoint {
        url: registration_url,
        metadata: metadata.clone(),
    };

    let key = match registry.register_model(&endpoint).await {
        Ok(k) => k,
        Err(e) => {
            tracing::error!("Failed to register model {}: {}", metadata.name, e);
            return;
        }
    };

    tracing::info!("Registered model {} at key {}", metadata.name, key);

    // Keep-alive loop
    let conn = registry.connection();
    let refresh_interval = Duration::from_secs(lease_ttl / 2);
    let model_json = serde_json::to_string(&endpoint).unwrap_or_default();
    let mut conn = conn;

    loop {
        tokio::select! {
            _ = tokio::time::sleep(refresh_interval) => {
                use redis::AsyncCommands;
                match conn.set_ex::<_, _, ()>(&key, &model_json, lease_ttl).await {
                    Ok(()) => tracing::debug!("Refreshed TTL for {}", key),
                    Err(e) => tracing::error!("Failed to refresh TTL for {}: {}", key, e),
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("Shutting down announcer for model {}", metadata.name);
                    // Unregister
                    let model_id = nilai_domain::ids::ModelId::new(&metadata.id);
                    if let Err(e) = registry.unregister_model(&model_id).await {
                        tracing::error!("Failed to unregister model {}: {}", metadata.name, e);
                    } else {
                        tracing::info!("Unregistered model {}", metadata.name);
                    }
                    break;
                }
            }
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nilai_lmstudio_announcer=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting NilAI LMStudio Announcer");

    // Configuration from environment
    let api_base = env::var("LMSTUDIO_API_BASE").unwrap_or_else(|_| "http://localhost:1234".into());
    let models_endpoint =
        env::var("LMSTUDIO_MODELS_ENDPOINT").unwrap_or_else(|_| "/v1/models".into());
    let registration_url = env::var("LMSTUDIO_REGISTRATION_URL")
        .unwrap_or_else(|_| api_base.clone())
        .trim_end_matches('/')
        .to_string();
    let discovery_url =
        env::var("DISCOVERY_URL").unwrap_or_else(|_| "redis://localhost:6379".into());
    let lease_ttl: u64 = env::var("LMSTUDIO_LEASE_TTL")
        .unwrap_or_else(|_| "60".into())
        .parse()
        .unwrap_or(60);
    let fetch_timeout: f64 = env::var("LMSTUDIO_FETCH_TIMEOUT")
        .unwrap_or_else(|_| "15".into())
        .parse()
        .unwrap_or(15.0);
    let max_retries: u32 = env::var("LMSTUDIO_MAX_RETRIES")
        .unwrap_or_else(|_| "30".into())
        .parse()
        .unwrap_or(30);

    // Feature defaults
    let tool_default =
        parse_bool(&env::var("LMSTUDIO_TOOL_SUPPORT_DEFAULT").unwrap_or_else(|_| "true".into()));
    let multimodal_default =
        parse_bool(&env::var("LMSTUDIO_MULTIMODAL_DEFAULT").unwrap_or_else(|_| "true".into()));
    let supported_features = {
        let raw =
            env::var("LMSTUDIO_SUPPORTED_FEATURES").unwrap_or_else(|_| "chat_completion".into());
        let parsed = parse_csv(&raw);
        if parsed.is_empty() {
            vec!["chat_completion".to_string()]
        } else {
            parsed
        }
    };
    let version = env::var("LMSTUDIO_MODEL_VERSION").unwrap_or_else(|_| "local".into());
    let author = env::var("LMSTUDIO_MODEL_AUTHOR").unwrap_or_else(|_| "LMStudio".into());
    let license = env::var("LMSTUDIO_MODEL_LICENSE").unwrap_or_else(|_| "local-use-only".into());

    // Discover or parse model IDs
    let model_ids = {
        let env_ids = env::var("LMSTUDIO_MODEL_IDS").unwrap_or_default();
        let parsed = parse_csv(&env_ids);
        if parsed.is_empty() {
            match fetch_model_ids(&api_base, &models_endpoint, fetch_timeout, max_retries).await {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!("Failed to discover models: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            parsed
        }
    };

    if model_ids.is_empty() {
        tracing::error!("No models discovered; nothing to announce");
        std::process::exit(1);
    }

    tracing::info!(
        "Announcing {} model(s) via {} with Redis at {}",
        model_ids.len(),
        registration_url,
        discovery_url
    );

    // Setup shutdown signal
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Spawn an announcer task per model
    let mut handles = Vec::new();
    for model_id in &model_ids {
        let metadata = ModelMetadata {
            id: model_id.clone(),
            name: model_id.clone(),
            version: version.clone(),
            description: format!("LMStudio served model {}", model_id),
            author: author.clone(),
            license: license.clone(),
            source: format!("lmstudio://{}", model_id),
            supported_features: supported_features.clone(),
            tool_support: tool_default,
            multimodal_support: multimodal_default,
        };

        let handle = tokio::spawn(announce_model(
            metadata,
            registration_url.clone(),
            discovery_url.clone(),
            lease_ttl,
            shutdown_rx.clone(),
        ));
        handles.push(handle);
    }

    // Wait for shutdown signal
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, stopping all announcers...");
    let _ = shutdown_tx.send(true);

    for handle in handles {
        let _ = handle.await;
    }

    tracing::info!("LMStudio Announcer stopped");
}
