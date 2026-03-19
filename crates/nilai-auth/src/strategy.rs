use nilai_domain::auth::AuthenticationInfo;
use nilai_domain::error::NilaiResult;
use nilai_domain::ids::ApiKey;
use nilai_domain::ports::{CreditService, UserStore};

use crate::nuc;

/// API Key authentication strategy.
///
/// Validates the API key with the credit service and retrieves or creates the
/// user in the user store.
pub async fn api_key_strategy(
    api_key: &ApiKey,
    credit_service: &dyn CreditService,
    user_store: &dyn UserStore,
) -> NilaiResult<AuthenticationInfo> {
    // Validate credential with credit service
    let user_id = credit_service.validate_credential(api_key, false).await?;

    // Get or create user
    let user = match user_store.get_user(&user_id).await? {
        Some(user) => user,
        None => user_store.insert_user(&user_id).await?,
    };

    Ok(AuthenticationInfo {
        user,
        token_rate_limit: None,
        prompt_document: None,
    })
}

/// NUC (Nillion User Credential) authentication strategy.
///
/// Parses the NUC token, validates the proof chain against trusted root keys,
/// extracts rate limits and prompt document metadata, and validates the
/// subscription holder with the credit service.
pub async fn nuc_strategy(
    token: &str,
    trusted_root_keys: &[[u8; 33]],
    audience_did: Option<&nillion_nucs::did::Did>,
    credit_service: &dyn CreditService,
    user_store: &dyn UserStore,
) -> NilaiResult<AuthenticationInfo> {
    // Parse and validate the NUC token
    let nuc_info = nuc::validate_nuc(token, trusted_root_keys, audience_did)?;

    // Extract rate limits and prompt document from the validated token chain
    let token_rate_limits = nuc::rate_limits::extract_token_rate_limits(
        &nuc_info.validated.token,
        &nuc_info.validated.proofs,
    )?;

    let prompt_document = nuc::prompt_document::extract_prompt_document(
        &nuc_info.validated.token,
        &nuc_info.validated.proofs,
    )?;

    // Validate the subscription holder with credit service
    let subscription_key = ApiKey::new(&nuc_info.subscription_holder);
    let user_id = credit_service
        .validate_credential(&subscription_key, true)
        .await?;

    // Get or create user
    let user = match user_store.get_user(&user_id).await? {
        Some(user) => user,
        None => user_store.insert_user(&user_id).await?,
    };

    Ok(AuthenticationInfo {
        user,
        token_rate_limit: token_rate_limits,
        prompt_document,
    })
}
