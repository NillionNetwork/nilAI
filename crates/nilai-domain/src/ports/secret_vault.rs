use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::{Did, DocumentId};

#[async_trait]
pub trait SecretVaultClient: Send + Sync {
    async fn read_document(
        &self,
        document_id: &DocumentId,
        delegation_token: &str,
    ) -> NilaiResult<String>;
    async fn create_delegation_token(
        &self,
        command: &str,
        audience_did: &Did,
        ttl_secs: u64,
    ) -> NilaiResult<String>;
}
