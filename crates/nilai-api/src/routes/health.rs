use crate::state::AppState;
use axum::{extract::State, routing::get, Json, Router};
use nilai_domain::health::HealthCheckResponse;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/health", get(health_check))
        .route("/v1/public_key", get(get_public_key))
}

async fn health_check(State(state): State<AppState>) -> Json<HealthCheckResponse> {
    Json(HealthCheckResponse {
        status: "ok".to_string(),
        uptime: state.uptime_secs(),
    })
}

async fn get_public_key(State(state): State<AppState>) -> String {
    state.keypair.public_key_b64().as_str().to_string()
}
