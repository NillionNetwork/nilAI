use crate::state::AppState;
use axum::{extract::State, http::StatusCode, routing::get, Extension, Json, Router};
use nilai_domain::auth::AuthenticationInfo;
use nilai_domain::usage::Usage;

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/usage", get(get_usage))
}

async fn get_usage(
    State(state): State<AppState>,
    Extension(auth_info): Extension<AuthenticationInfo>,
) -> Result<Json<Usage>, (StatusCode, Json<serde_json::Value>)> {
    let log_store = state.query_log_store.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Query log store unavailable"})),
        )
    })?;

    let usage = log_store
        .get_user_usage(&auth_info.user.user_id)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
        })?;

    Ok(Json(usage))
}
