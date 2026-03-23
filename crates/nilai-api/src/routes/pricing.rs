use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    routing::get,
    Json, Router,
};
use nilai_domain::ids::ModelName;
use nilai_domain::usage::LLMPriceConfig;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/pricing", get(get_all_prices))
        .route(
            "/v1/pricing/{*model_name}",
            get(get_price).put(set_price).delete(delete_price),
        )
}

async fn get_all_prices(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let store = state.pricing_store.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Pricing store unavailable"})),
        )
    })?;

    let prices = store.get_all_prices().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"detail": format!("{}", e)})),
        )
    })?;

    Ok(Json(serde_json::to_value(prices).unwrap_or_default()))
}

async fn get_price(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<LLMPriceConfig>, (StatusCode, Json<serde_json::Value>)> {
    let model_name = model_name.strip_prefix('/').unwrap_or(&model_name);

    let store = state.pricing_store.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Pricing store unavailable"})),
        )
    })?;

    let price = store
        .get_price(&ModelName::new(model_name))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
        })?;

    Ok(Json(price))
}

async fn set_price(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
    headers: HeaderMap,
    Json(config): Json<LLMPriceConfig>,
) -> Result<Json<LLMPriceConfig>, (StatusCode, Json<serde_json::Value>)> {
    let model_name = model_name.strip_prefix('/').unwrap_or(&model_name);

    // Verify admin token
    verify_admin_token(&state, &headers)?;

    // Validate prices are non-negative
    if config.prompt_tokens_price < 0.0
        || config.completion_tokens_price < 0.0
        || config.web_search_cost < 0.0
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": "Prices must be non-negative"})),
        ));
    }

    let store = state.pricing_store.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Pricing store unavailable"})),
        )
    })?;

    store
        .set_price(&ModelName::new(model_name), &config)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
        })?;

    Ok(Json(config))
}

async fn delete_price(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
    headers: HeaderMap,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let model_name = model_name.strip_prefix('/').unwrap_or(&model_name);

    verify_admin_token(&state, &headers)?;

    if model_name == "default" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": "Cannot delete default pricing"})),
        ));
    }

    let store = state.pricing_store.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Pricing store unavailable"})),
        )
    })?;

    let existed = store
        .delete_price(&ModelName::new(model_name))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
        })?;

    if !existed {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"detail": "Price not found"})),
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

fn verify_admin_token(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let admin_token = state.config.auth.admin_token.as_ref().ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"detail": "Admin token not configured"})),
        )
    })?;

    let auth_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"detail": "Missing authorization"})),
            )
        })?;

    let token = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"detail": "Invalid authorization format"})),
        )
    })?;

    if token != admin_token {
        return Err((
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"detail": "Invalid admin token"})),
        ));
    }

    Ok(())
}
