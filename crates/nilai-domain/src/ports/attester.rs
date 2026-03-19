use async_trait::async_trait;

use crate::attestation::AttestationReport;
use crate::error::NilaiResult;
use crate::ids::Nonce;

#[async_trait]
pub trait Attester: Send + Sync {
    async fn get_report(&self, nonce: &Nonce) -> NilaiResult<AttestationReport>;
}
