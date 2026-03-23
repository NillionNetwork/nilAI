pub mod attestation;
pub mod chat;
pub mod delegation;
pub mod health;
pub mod models;
pub mod pricing;
pub mod responses;
pub mod usage;

use crate::state::AppState;
use axum::Router;

pub fn public_routes() -> Router<AppState> {
    Router::new().merge(health::router())
}

pub fn private_routes() -> Router<AppState> {
    Router::new()
        .merge(models::router())
        .merge(attestation::router())
        .merge(usage::router())
        .merge(pricing::router())
        .merge(delegation::router())
        .merge(responses::router())
        .merge(chat::router())
}
