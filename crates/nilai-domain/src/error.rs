/// Central error type for the NilAI domain.
#[derive(Debug, thiserror::Error)]
pub enum NilaiError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("unauthorized: {0}")]
    Unauthorized(String),

    #[error("rate limited (retry after {retry_after_ms}ms)")]
    RateLimited { retry_after_ms: u64 },

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("config error: {0}")]
    ConfigError(String),

    #[error("crypto error: {0}")]
    CryptoError(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error("external service error [{service}]: {message}")]
    ExternalService { service: String, message: String },
}

pub type NilaiResult<T> = Result<T, NilaiError>;
