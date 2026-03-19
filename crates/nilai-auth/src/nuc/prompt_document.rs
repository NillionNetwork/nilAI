use nilai_domain::auth::PromptDocument;
use nilai_domain::error::NilaiResult;
use nillion_nucs::token::{NucToken, TokenBody};

/// Extract prompt document information from a NUC token chain.
///
/// Traverses the proof chain (excluding the invocation token) looking
/// for the first proof with `document_id` and `document_owner_did` metadata.
/// Validates that `document_owner_did` matches the proof's issuer.
pub fn extract_prompt_document(
    _main_token: &NucToken,
    proofs: &[NucToken],
) -> NilaiResult<Option<PromptDocument>> {
    // Skip the main (invocation) token -- it's created by the user and should not
    // be trusted for document metadata. Only look at delegation proofs.
    for proof in proofs.iter().rev() {
        // Only consider delegation tokens (not invocations)
        if !matches!(proof.body, TokenBody::Delegation(_)) {
            continue;
        }

        if let Some(ref meta) = proof.meta {
            let document_id = meta.get("document_id").and_then(|v| v.as_str());
            let owner_did = meta.get("document_owner_did").and_then(|v| v.as_str());

            if let (Some(doc_id), Some(owner)) = (document_id, owner_did) {
                // Validate that the document_owner_did matches the proof's issuer
                let issuer_str = proof.issuer.to_string();
                if owner != issuer_str {
                    tracing::warn!(
                        "document_owner_did '{}' does not match proof issuer '{}'",
                        owner,
                        issuer_str
                    );
                    continue;
                }

                return Ok(Some(PromptDocument {
                    document_id: doc_id.to_string(),
                    owner_did: owner.to_string(),
                }));
            }
        }
    }

    Ok(None)
}
