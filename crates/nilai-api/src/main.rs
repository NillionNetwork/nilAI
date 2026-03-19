// Allow dead code during incremental migration -- scaffolded modules
// are not yet wired into all routes.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;
use tokio::signal;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod features;
mod middleware;
mod routes;
mod signing;
mod state;

use nilai_infra::config::NilaiConfig;
use nilai_infra::crypto::KeyPair;
use state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nilai_api=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting NilAI Rust API server");

    // Load configuration
    let config_path = std::path::Path::new("nilai-api/src/nilai_api/config/config.yaml");
    let config = NilaiConfig::load(if config_path.exists() {
        Some(config_path)
    } else {
        None
    })
    .unwrap_or_else(|e| {
        tracing::warn!("Failed to load config file, using defaults: {}", e);
        NilaiConfig::load(None).expect("Failed to load default config")
    });

    let keypair =
        KeyPair::generate_or_load("private_key.key").expect("Failed to generate or load key pair");
    tracing::info!("Public key: {}", keypair.public_key_b64().as_str());

    // Wire up infrastructure adapters based on config
    let model_registry = wire_model_registry(&config).await;
    let user_store = wire_user_store(&config).await;
    let query_log_store = wire_query_log_store(&config).await;
    let rate_limiter = wire_rate_limiter(&config).await;
    let pricing_store = wire_pricing_store(&config).await;
    let inference_client = wire_inference_client();
    let search_provider = wire_search_provider(&config);
    let attester = wire_attester();
    let credit_service = wire_credit_service(&config);

    let state = AppState {
        keypair: Arc::new(keypair),
        config: Arc::new(config),
        start_time: Instant::now(),
        model_registry,
        user_store,
        query_log_store,
        rate_limiter,
        pricing_store,
        inference_client,
        search_provider,
        attester,
        credit_service,
        code_executor: None, // TODO: e2b integration
        secret_vault: None,  // TODO: nilDB integration
    };

    let app = app::create_app(state);

    let addr = "0.0.0.0:8081";
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn wire_model_registry(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::ModelRegistry>> {
    let url = config
        .discovery
        .as_ref()
        .map(|d| d.url.as_str())
        .unwrap_or("redis://localhost:6379");
    match nilai_discovery::RedisModelRegistry::new(url, 60).await {
        Ok(r) => {
            tracing::info!("Model registry connected to {}", url);
            Some(Arc::new(r))
        }
        Err(e) => {
            tracing::warn!("Model registry unavailable: {}", e);
            None
        }
    }
}

async fn wire_user_store(config: &NilaiConfig) -> Option<Arc<dyn nilai_domain::ports::UserStore>> {
    if let Some(ref db_config) = config.database {
        match nilai_infra::postgres::pool::create_pool(db_config).await {
            Ok(pool) => {
                tracing::info!("PostgreSQL user store connected");
                Some(Arc::new(nilai_infra::postgres::user_repo::PgUserRepo::new(
                    pool,
                )))
            }
            Err(e) => {
                tracing::warn!("PostgreSQL unavailable: {}", e);
                None
            }
        }
    } else {
        tracing::info!("No database config, user store unavailable");
        None
    }
}

async fn wire_query_log_store(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::QueryLogStore>> {
    if let Some(ref db_config) = config.database {
        match nilai_infra::postgres::pool::create_pool(db_config).await {
            Ok(pool) => {
                tracing::info!("PostgreSQL query log store connected");
                Some(Arc::new(
                    nilai_infra::postgres::query_log_repo::PgQueryLogRepo::new(pool),
                ))
            }
            Err(e) => {
                tracing::warn!("PostgreSQL query log unavailable: {}", e);
                None
            }
        }
    } else {
        None
    }
}

async fn wire_rate_limiter(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::RateLimiter>> {
    let url = config.redis.as_ref().map(|r| r.url.as_str());
    if let Some(redis_url) = url {
        match nilai_infra::redis::pool::RedisPool::new(redis_url).await {
            Ok(pool) => {
                tracing::info!("Redis rate limiter connected");
                Some(Arc::new(
                    nilai_infra::redis::rate_limiter::RedisRateLimiter::new(pool),
                ))
            }
            Err(e) => {
                tracing::warn!("Redis rate limiter unavailable: {}", e);
                None
            }
        }
    } else {
        None
    }
}

async fn wire_pricing_store(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::PricingStore>> {
    let url = config.redis.as_ref().map(|r| r.url.as_str());
    if let Some(redis_url) = url {
        match nilai_infra::redis::pool::RedisPool::new(redis_url).await {
            Ok(pool) => {
                tracing::info!("Redis pricing store connected");
                Some(Arc::new(
                    nilai_infra::redis::pricing_repo::RedisPricingRepo::new(pool),
                ))
            }
            Err(e) => {
                tracing::warn!("Redis pricing store unavailable: {}", e);
                None
            }
        }
    } else {
        None
    }
}

fn wire_inference_client() -> Option<Arc<dyn nilai_domain::ports::InferenceClient>> {
    Some(Arc::new(nilai_infra::http::vllm_client::VllmClient::new()))
}

fn wire_search_provider(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::SearchProvider>> {
    if let Some(ref api_key) = config.web_search.api_key {
        Some(Arc::new(
            nilai_infra::http::brave_search::BraveSearchClient::new(
                api_key.clone(),
                config.web_search.api_path.clone(),
            ),
        ))
    } else {
        tracing::info!("No web search API key, search provider unavailable");
        None
    }
}

fn wire_attester() -> Option<Arc<dyn nilai_domain::ports::Attester>> {
    // nilcc-attester runs as a sidecar, default to localhost:8082
    Some(Arc::new(
        nilai_infra::http::attester_client::NilccAttesterClient::new(
            "http://localhost:8082".to_string(),
        ),
    ))
}

fn wire_credit_service(
    config: &NilaiConfig,
) -> Option<Arc<dyn nilai_domain::ports::CreditService>> {
    if let Some(url) = config.auth.credit_service_url() {
        Some(Arc::new(
            nilai_infra::http::credit_client::NilauthCreditClient::new(
                url.to_string(),
                config.auth.credit_api_token.clone(),
            ),
        ))
    } else {
        tracing::warn!("No credit service URL configured");
        None
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
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

    tracing::info!("Shutdown signal received");
}
