use axum::{middleware, Router};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::routes;
use crate::state::AppState;

pub fn create_app(state: AppState) -> Router {
    let public_routes = routes::public_routes();
    let private_routes = routes::private_routes();

    // Apply auth middleware to private routes if credit_service and user_store are available
    let private_routes = if state.credit_service.is_some() && state.user_store.is_some() {
        let auth_state = nilai_auth::middleware::AuthState {
            config: Arc::new(nilai_infra::config::AuthConfig {
                auth_strategy: state.config.auth.auth_strategy.clone(),
                nilauth_trusted_root_issuers: state
                    .config
                    .auth
                    .nilauth_trusted_root_issuers
                    .clone(),
                credit_api_token: state.config.auth.credit_api_token.clone(),
                auth_token: state.config.auth.auth_token.clone(),
                admin_token: state.config.auth.admin_token.clone(),
            }),
            credit_service: state.credit_service.clone().unwrap(),
            user_store: state.user_store.clone().unwrap(),
            trusted_root_keys: vec![], // TODO: Load from config when NUC root keys are provisioned
            audience_did: None,        // TODO: Derive from server's public key
        };
        private_routes.route_layer(middleware::from_fn_with_state(
            auth_state,
            nilai_auth::middleware::auth_middleware,
        ))
    } else {
        tracing::warn!("Auth middleware not applied - credit_service or user_store unavailable");
        private_routes
    };

    Router::new()
        .merge(public_routes)
        .merge(private_routes)
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}
