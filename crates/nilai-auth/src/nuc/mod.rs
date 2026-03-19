pub mod prompt_document;
pub mod rate_limits;

use nillion_nucs::{
    envelope::NucTokenEnvelope,
    token::NucToken,
    validator::{NucValidator, TokenTypeRequirements, ValidatedNucToken, ValidationParameters},
};
use std::collections::HashMap;

use nilai_domain::error::{NilaiError, NilaiResult};

/// Information extracted from a validated NUC token.
pub struct NucInfo {
    /// The subscription holder (subject of the token chain).
    pub subscription_holder: String,
    /// The actual user (issuer of the invocation token).
    pub user: String,
    /// The validated token (for further extraction).
    pub validated: ValidatedNucToken,
}

/// Validate a NUC token and extract principal information.
///
/// Parses the token envelope, validates signatures and the proof chain,
/// checks the command is under `/nil/ai`, and extracts the subscription holder
/// (subject) and user (issuer of the invocation token).
pub fn validate_nuc(
    token_str: &str,
    trusted_root_keys: &[[u8; 33]],
    audience_did: Option<&nillion_nucs::did::Did>,
) -> NilaiResult<NucInfo> {
    if token_str.is_empty() {
        return Err(NilaiError::Unauthorized("Empty NUC token".to_string()));
    }

    // Parse the token envelope
    let envelope = NucTokenEnvelope::decode(token_str)
        .map_err(|e| NilaiError::Unauthorized(format!("Failed to parse NUC token: {}", e)))?;

    // Build validator with trusted root keys
    let validator = NucValidator::new(trusted_root_keys.iter().copied())
        .map_err(|e| NilaiError::Internal(format!("Failed to create NUC validator: {:?}", e)))?;

    // Set validation parameters
    let token_requirements = match audience_did {
        Some(did) => TokenTypeRequirements::Invocation(*did),
        None => TokenTypeRequirements::None,
    };

    let params = ValidationParameters {
        max_chain_length: 5,
        max_policy_width: 10,
        max_policy_depth: 5,
        token_requirements,
    };

    let context = HashMap::new();
    let validated = validator
        .validate(envelope, params, &context)
        .map_err(|e| NilaiError::Unauthorized(format!("NUC token validation failed: {}", e)))?;

    // Extract principals:
    // - subject = subscription holder (consistent throughout chain)
    // - issuer of the invocation token = the actual user
    let subscription_holder = validated.token.subject.to_string();
    let user = validated.token.issuer.to_string();

    Ok(NucInfo {
        subscription_holder,
        user,
        validated,
    })
}

/// Parse a NUC token envelope without full validation.
/// Used for extracting metadata (rate limits, prompt documents) from the proof chain.
pub fn parse_envelope(token_str: &str) -> NilaiResult<(NucToken, Vec<NucToken>)> {
    let envelope = NucTokenEnvelope::decode(token_str)
        .map_err(|e| NilaiError::Unauthorized(format!("Failed to parse NUC token: {}", e)))?;

    // Extract tokens (consuming envelope)
    let (main_token, proofs) = envelope.into_parts();
    let main = main_token.into_token();
    let proof_tokens: Vec<NucToken> = proofs.into_iter().map(|p| p.into_token()).collect();

    Ok((main, proof_tokens))
}
