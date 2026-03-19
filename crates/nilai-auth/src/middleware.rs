use std::sync::Arc;

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json, Router,
};

use nilai_domain::auth::{AuthStrategy, AuthenticationInfo};
use nilai_domain::error::NilaiError;
use nilai_domain::ids::ApiKey;
use nilai_domain::ports::{CreditService, UserStore};
use nilai_infra::config::AuthConfig;

use crate::strategy;

/// State needed by the auth middleware.
#[derive(Clone)]
pub struct AuthState {
    pub config: Arc<AuthConfig>,
    pub credit_service: Arc<dyn CreditService>,
    pub user_store: Arc<dyn UserStore>,
    /// Trusted root issuer public keys (compressed secp256k1, 33 bytes each).
    /// Used for NUC token validation.
    pub trusted_root_keys: Vec<[u8; 33]>,
    /// Our server's DID (audience for NUC invocation tokens).
    pub audience_did: Option<nillion_nucs::did::Did>,
}

/// Axum middleware function for authentication.
///
/// Extracts the Bearer token from the Authorization header, dispatches to the
/// appropriate authentication strategy (API key or NUC), and inserts the
/// resulting `AuthenticationInfo` as a request extension.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Extract bearer token
    let token = match extract_bearer_token(&request) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"detail": "Missing or invalid Authorization header"})),
            )
                .into_response();
        }
    };

    // Check for docs token bypass
    if let Some(ref docs_token) = state.config.auth_token {
        if token == *docs_token {
            let auth_info = AuthenticationInfo {
                user: nilai_domain::user::UserData {
                    user_id: nilai_domain::ids::UserId::new("docs-user"),
                    rate_limits: nilai_domain::user::RateLimits::default(),
                },
                token_rate_limit: None,
                prompt_document: None,
            };
            request.extensions_mut().insert(auth_info);
            return next.run(request).await;
        }
    }

    // Dispatch to strategy
    let strategy = if state.config.auth_strategy == "nuc" {
        AuthStrategy::Nuc
    } else {
        AuthStrategy::ApiKey
    };

    let auth_result = match strategy {
        AuthStrategy::ApiKey => {
            strategy::api_key_strategy(
                &ApiKey::new(&token),
                state.credit_service.as_ref(),
                state.user_store.as_ref(),
            )
            .await
        }
        AuthStrategy::Nuc => {
            strategy::nuc_strategy(
                &token,
                &state.trusted_root_keys,
                state.audience_did.as_ref(),
                state.credit_service.as_ref(),
                state.user_store.as_ref(),
            )
            .await
        }
        _ => Err(NilaiError::Unauthorized(
            "Unsupported auth strategy".to_string(),
        )),
    };

    match auth_result {
        Ok(auth_info) => {
            request.extensions_mut().insert(auth_info);
            next.run(request).await
        }
        Err(e) => {
            tracing::warn!("Authentication failed: {}", e);
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"detail": format!("{}", e)})),
            )
                .into_response()
        }
    }
}

fn extract_bearer_token(request: &Request) -> Option<String> {
    let auth_header = request.headers().get(header::AUTHORIZATION)?;
    let auth_str = auth_header.to_str().ok()?;
    auth_str.strip_prefix("Bearer ").map(|s| s.to_string())
}

/// Apply auth middleware to a router.
pub fn with_auth<S: Clone + Send + Sync + 'static>(
    router: Router<S>,
    auth_state: AuthState,
) -> Router<S> {
    router.route_layer(middleware::from_fn_with_state(auth_state, auth_middleware))
}
