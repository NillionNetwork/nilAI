use crate::state::AppState;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::Deserialize;

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/delegation", get(get_delegation_token))
}

#[derive(Deserialize)]
struct DelegationQuery {
    prompt_delegation_request: String,
}

async fn get_delegation_token(
    State(state): State<AppState>,
    Query(query): Query<DelegationQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let vault = state.secret_vault.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "nilDB service unavailable"})),
        )
    })?;

    let did = nilai_domain::ids::Did::new(&query.prompt_delegation_request);
    let token = vault
        .create_delegation_token("/nil/db", &did, 60)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
        })?;

    Ok(Json(serde_json::json!({
        "token": token,
        "did": query.prompt_delegation_request
    })))
}
