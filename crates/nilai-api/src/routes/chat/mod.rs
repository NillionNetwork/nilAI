mod handler;
mod stream;

use axum::{routing::post, Router};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(handler::chat_completion))
}
