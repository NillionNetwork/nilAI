use crate::state::AppState;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use nilai_domain::attestation::AttestationReport;
use nilai_domain::ids::Nonce;
use serde::Deserialize;

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/attestation/report", get(get_attestation_report))
}

#[derive(Deserialize)]
struct AttestationQuery {
    nonce: String,
}

async fn get_attestation_report(
    State(state): State<AppState>,
    Query(query): Query<AttestationQuery>,
) -> Result<Json<AttestationReport>, (StatusCode, Json<serde_json::Value>)> {
    // Validate nonce is 64-char hex
    if query.nonce.len() != 64 || !query.nonce.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": "Nonce must be a 64-character hex string"})),
        ));
    }

    let attester = state.attester.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Attestation service unavailable"})),
        )
    })?;

    let nonce = Nonce::new(&query.nonce);
    let mut report = attester.get_report(&nonce).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"detail": format!("{}", e)})),
        )
    })?;

    // Override verifying_key with our public key
    report.verifying_key = state.keypair.public_key_b64().as_str().to_string();

    Ok(Json(report))
}
