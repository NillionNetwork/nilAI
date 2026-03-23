use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};
use nilai_domain::auth::AuthenticationInfo;
use nilai_domain::chat::{ChatRequest, SignedChatCompletion};
use nilai_domain::error::NilaiError;
use nilai_domain::ids::ModelName;

use super::stream;
use crate::middleware::rate_limit;
use crate::state::AppState;

type ApiError = (StatusCode, Json<serde_json::Value>);

pub async fn chat_completion(
    State(state): State<AppState>,
    Extension(auth_info): Extension<AuthenticationInfo>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    // Validate request
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": "Messages list cannot be empty"})),
        ));
    }

    let model_name = &req.model;

    // Look up model endpoint
    let registry = state.model_registry.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Model registry unavailable"})),
        )
    })?;

    let models = registry
        .discover_models(Some(&ModelName::new(model_name)), None)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("Model discovery error: {}", e)})),
            )
        })?;

    let endpoint = models.values().next().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": format!("Model '{}' not found", model_name)})),
        )
    })?;

    // Validate tool support
    if req.tools.is_some() && !endpoint.metadata.tool_support {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({"detail": format!("Model '{}' does not support tools", model_name)}),
            ),
        ));
    }

    // Check rate limits if available
    if let Some(ref limiter) = state.rate_limiter {
        let web_search_enabled = req.web_search.unwrap_or(false);
        let web_search_rps = Some(state.config.web_search.rps);
        rate_limit::check_rate_limits(
            limiter.as_ref(),
            &auth_info,
            web_search_enabled,
            web_search_rps,
        )
        .await
        .map_err(|e| match e {
            NilaiError::RateLimited { retry_after_ms } => {
                let mut headers = axum::http::HeaderMap::new();
                headers.insert("Retry-After", retry_after_ms.to_string().parse().unwrap());
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({"detail": "Too Many Requests"})),
                )
            }
            other => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", other)})),
            ),
        })?;

        // Concurrent rate limiting
        let concurrent_key = format!("chat:{}", model_name);
        let max_concurrent = state
            .config
            .rate_limiting
            .model_concurrent_rate_limit
            .get(model_name)
            .or_else(|| {
                state
                    .config
                    .rate_limiting
                    .model_concurrent_rate_limit
                    .get("default")
            })
            .copied()
            .unwrap_or(50);

        let _guard =
            rate_limit::ConcurrentGuard::acquire(limiter.clone(), concurrent_key, max_concurrent)
                .await
                .map_err(|_| {
                    (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(serde_json::json!({"detail": "Too Many Requests"})),
                    )
                })?;
    }

    let inference_client = state.inference_client.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Inference client unavailable"})),
        )
    })?;

    let base_url = url::Url::parse(&format!("{}/v1/", endpoint.url)).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"detail": format!("Invalid model URL: {}", e)})),
        )
    })?;

    // Handle streaming vs non-streaming
    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        // Streaming response
        let sse_stream = stream::create_sse_stream(inference_client.clone(), base_url, req)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("{}", e)})),
                )
            })?;

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(Body::from_stream(sse_stream))
            .unwrap())
    } else {
        // Non-streaming response
        let completion = inference_client
            .chat_completion(&base_url, &req)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("{}", e)})),
                )
            })?;

        // Sign the response
        let response_json = serde_json::to_string(&completion).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("Serialization error: {}", e)})),
            )
        })?;

        let signature = state
            .keypair
            .sign_message_b64(&response_json)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("Signing error: {}", e)})),
                )
            })?;

        let signed = SignedChatCompletion {
            completion,
            signature,
            sources: None, // TODO: add web search sources when web search is integrated
        };

        // Log usage in background
        if let (Some(ref log_store), Some(ref usage)) =
            (&state.query_log_store, &signed.completion.usage)
        {
            let log = nilai_domain::usage::QueryLog {
                user_id: auth_info.user.user_id.as_str().to_string(),
                model: model_name.clone(),
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                web_searches: 0,
                lock_id: String::new(),
            };
            let store = log_store.clone();
            tokio::spawn(async move {
                if let Err(e) = store.log_query(&log).await {
                    tracing::error!("Failed to log query: {}", e);
                }
            });
        }

        Ok(Json(signed).into_response())
    }
}
