use chrono::Utc;
use nilai_domain::auth::{TokenRateLimit, TokenRateLimits};
use nilai_domain::error::{NilaiError, NilaiResult};
use nillion_nucs::token::NucToken;

/// Extract token rate limits from NUC token chain.
///
/// Traverses the proof chain in reverse order (from root toward invocation)
/// and extracts `usage_limit` metadata from each proof. Each subsequent limit
/// must be a valid reduction of the previous one (0 < new <= previous).
///
/// The rate limit key for each proof is derived from its nonce (hex-encoded).
pub fn extract_token_rate_limits(
    main_token: &NucToken,
    proofs: &[NucToken],
) -> NilaiResult<Option<TokenRateLimits>> {
    // Build full chain: main token + proofs (proofs are ordered from token's proof to root)
    let all_tokens: Vec<&NucToken> = std::iter::once(main_token).chain(proofs.iter()).collect();

    let mut limits = Vec::new();
    let mut previous_limit: Option<u64> = None;

    // Iterate in reverse (from root toward invocation)
    for token in all_tokens.iter().rev() {
        if let Some(ref meta) = token.meta {
            if let Some(usage_value) = meta.get("usage_limit") {
                let usage_limit = match usage_value {
                    serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| {
                        NilaiError::BadRequest(
                            "Invalid usage limit type. Usage limit must be a positive integer."
                                .to_string(),
                        )
                    })?,
                    _ => {
                        return Err(NilaiError::BadRequest(
                            "Invalid usage limit type. Usage limit must be an integer.".to_string(),
                        ));
                    }
                };

                // Validate reduction: each limit must be <= previous
                if let Some(prev) = previous_limit {
                    if usage_limit > prev || usage_limit == 0 {
                        return Err(NilaiError::BadRequest(
                            "Inconsistent usage limit across proofs".to_string(),
                        ));
                    }
                }
                previous_limit = Some(usage_limit);

                // Use hex-encoded nonce as the rate limit key
                let signature_key = hex::encode(&token.nonce);

                // Use token expiration, or a past date if none
                let expires_at = token
                    .expires_at
                    .unwrap_or_else(|| Utc::now() - chrono::Duration::seconds(1));

                limits.push(TokenRateLimit {
                    signature: signature_key,
                    expires_at,
                    usage_limit: Some(usage_limit),
                });
            }
        }
    }

    if limits.is_empty() {
        Ok(None)
    } else {
        Ok(Some(TokenRateLimits { limits }))
    }
}
