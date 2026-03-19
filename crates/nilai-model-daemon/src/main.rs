use serde::Deserialize;
use std::env;
use tokio::sync::watch;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use nilai_discovery::RedisModelRegistry;
use nilai_domain::model::{ModelEndpoint, ModelMetadata};
use nilai_domain::ports::ModelRegistry;

#[derive(Deserialize)]
struct VllmModelsResponse {
    data: Vec<VllmModel>,
}

#[derive(Deserialize)]
struct VllmModel {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}

/// Fetch model metadata from the vLLM /v1/models endpoint with retries.
async fn get_metadata(model_url: &str) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/models", model_url);

    let mut attempts = 0;
    let max_attempts = 10;

    loop {
        attempts += 1;
        tracing::info!(
            "Fetching model metadata from {} (attempt {}/{})",
            url,
            attempts,
            max_attempts
        );

        match client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    let models: VllmModelsResponse = resp.json().await?;
                    if let Some(model) = models.data.first() {
                        return Ok(ModelMetadata {
                            id: uuid::Uuid::new_v4().to_string(),
                            name: model.id.clone(),
                            version: "1.0.0".to_string(),
                            description: format!("Model served by vLLM: {}", model.id),
                            author: model
                                .owned_by
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string()),
                            license: "unknown".to_string(),
                            source: model_url.to_string(),
                            supported_features: vec!["chat".to_string()],
                            tool_support: false,
                            multimodal_support: false,
                        });
                    }
                    return Err("No models found in vLLM response".into());
                }

                tracing::warn!("vLLM returned status {}", resp.status());
            }
            Err(e) => {
                tracing::warn!("Failed to connect to vLLM: {}", e);
            }
        }

        if attempts >= max_attempts {
            return Err("Max retry attempts reached for model metadata".into());
        }

        let delay = std::time::Duration::from_secs(2u64.pow((attempts as u32).min(5)));
        tracing::info!("Retrying in {:?}", delay);
        tokio::time::sleep(delay).await;
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nilai_model_daemon=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting NilAI Model Daemon");

    // Configuration from env vars
    let model_url = env::var("MODEL_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
    let discovery_url =
        env::var("DISCOVERY_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let lease_ttl: u64 = env::var("LEASE_TTL")
        .unwrap_or_else(|_| "60".to_string())
        .parse()
        .unwrap_or(60);

    // Fetch model metadata
    let metadata = match get_metadata(&model_url).await {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Failed to get model metadata: {}", e);
            std::process::exit(1);
        }
    };

    tracing::info!("Discovered model: {} ({})", metadata.name, metadata.id);

    // Create model endpoint
    let endpoint = ModelEndpoint {
        url: model_url.clone(),
        metadata,
    };

    // Initialize discovery service
    let registry = match RedisModelRegistry::new(&discovery_url, lease_ttl).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to connect to Redis: {}", e);
            std::process::exit(1);
        }
    };

    // Register model
    let key = match registry.register_model(&endpoint).await {
        Ok(k) => k,
        Err(e) => {
            tracing::error!("Failed to register model: {}", e);
            std::process::exit(1);
        }
    };

    tracing::info!("Model registered at key: {}", key);

    // Setup shutdown signal
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Spawn keep-alive task
    let conn = registry.connection();
    let endpoint_clone = endpoint.clone();
    let keep_alive_handle = tokio::spawn(async move {
        if let Err(e) = nilai_discovery::keepalive::keep_alive(
            conn,
            key,
            &endpoint_clone,
            lease_ttl,
            shutdown_rx,
        )
        .await
        {
            tracing::error!("Keep-alive error: {}", e);
        }
    });

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

    tracing::info!("Shutdown signal received, cleaning up...");

    // Signal shutdown
    let _ = shutdown_tx.send(true);

    // Wait for keep-alive to finish
    let _ = keep_alive_handle.await;

    tracing::info!("Model daemon stopped");
}
