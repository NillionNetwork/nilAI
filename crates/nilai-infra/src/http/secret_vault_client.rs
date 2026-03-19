use async_trait::async_trait;
use reqwest::Client;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::{Did, DocumentId};
use nilai_domain::ports::SecretVaultClient;

pub struct NilDbClient {
    #[allow(dead_code)]
    client: Client,
    #[allow(dead_code)]
    nodes: Vec<String>,
}

impl NilDbClient {
    pub fn new(nodes: Vec<String>) -> Self {
        Self {
            client: Client::new(),
            nodes,
        }
    }
}

#[async_trait]
impl SecretVaultClient for NilDbClient {
    async fn read_document(
        &self,
        _document_id: &DocumentId,
        _delegation_token: &str,
    ) -> NilaiResult<String> {
        // TODO: Implement nilDB read protocol
        Err(NilaiError::Internal(
            "nilDB read not yet implemented".to_string(),
        ))
    }

    async fn create_delegation_token(
        &self,
        _command: &str,
        _audience_did: &Did,
        _ttl_secs: u64,
    ) -> NilaiResult<String> {
        // TODO: Implement nilDB delegation token protocol
        Err(NilaiError::Internal(
            "nilDB delegation token not yet implemented".to_string(),
        ))
    }
}
