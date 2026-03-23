use crate::state::AppState;
use axum::{extract::State, http::StatusCode, routing::get, Json, Router};

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/models", get(list_models))
}

async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let registry = state.model_registry.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Model registry unavailable"})),
        )
    })?;

    let models = registry.discover_models(None, None).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"detail": format!("{}", e)})),
        )
    })?;

    let data: Vec<_> = models.values().map(|ep| &ep.metadata).collect();

    Ok(Json(serde_json::json!(data)))
}
