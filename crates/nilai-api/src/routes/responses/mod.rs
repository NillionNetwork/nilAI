mod handler;
mod stream;

use crate::state::AppState;
use axum::{routing::post, Router};

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/responses", post(handler::create_response))
}
