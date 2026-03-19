use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};
use nilai_domain::auth::AuthenticationInfo;
use nilai_domain::error::NilaiError;
use nilai_domain::ids::ModelName;
use nilai_domain::response::ResponseRequest;

use crate::middleware::rate_limit;
use crate::state::AppState;

type ApiError = (StatusCode, Json<serde_json::Value>);

pub async fn create_response(
    State(state): State<AppState>,
    Extension(auth_info): Extension<AuthenticationInfo>,
    Json(req): Json<ResponseRequest>,
) -> Result<Response, ApiError> {
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

    // Check rate limits
    if let Some(ref limiter) = state.rate_limiter {
        let web_search_enabled = req.web_search.unwrap_or(false);
        rate_limit::check_rate_limits(
            limiter.as_ref(),
            &auth_info,
            web_search_enabled,
            Some(state.config.web_search.rps),
        ).await.map_err(|e| match e {
            NilaiError::RateLimited { retry_after_ms } => {
                (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({"detail": "Too Many Requests", "retry_after_ms": retry_after_ms})))
            }
            other => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"detail": format!("{}", other)})))
        })?;
    }

    // Forward the request to vLLM's responses endpoint via reqwest.
    // The InferenceClient port currently only supports chat completions,
    // so we use reqwest directly for the responses API.
    let client = reqwest::Client::new();
    let base_url = format!("{}/v1/responses", endpoint.url);

    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        // Forward streaming response
        let resp = client
            .post(&base_url)
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("Inference request failed: {}", e)})),
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err((
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(serde_json::json!({"detail": body})),
            ));
        }

        let byte_stream = resp.bytes_stream();
        use futures::StreamExt;
        let mapped = byte_stream.map(|result| result.map_err(std::io::Error::other));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(Body::from_stream(mapped))
            .unwrap())
    } else {
        // Non-streaming: forward, sign, return
        let resp = client
            .post(&base_url)
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("Inference request failed: {}", e)})),
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err((
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(serde_json::json!({"detail": body})),
            ));
        }

        let response_body: serde_json::Value = resp.json().await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("Parse error: {}", e)})),
            )
        })?;

        // Sign the response
        let response_json = serde_json::to_string(&response_body).unwrap_or_default();
        let signature = state
            .keypair
            .sign_message_b64(&response_json)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"detail": format!("Signing error: {}", e)})),
                )
            })?;

        let signed = nilai_domain::response::SignedResponse {
            response: response_body,
            signature,
            sources: None,
        };

        Ok(Json(signed).into_response())
    }
}
