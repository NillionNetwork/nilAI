use async_trait::async_trait;
use reqwest::Client;

use nilai_domain::attestation::AttestationReport;
use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::Nonce;
use nilai_domain::ports::Attester;

pub struct NilccAttesterClient {
    client: Client,
    base_url: String,
}

impl NilccAttesterClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }
}

#[async_trait]
impl Attester for NilccAttesterClient {
    async fn get_report(&self, nonce: &Nonce) -> NilaiResult<AttestationReport> {
        let url = format!(
            "{}/attestation/report?nonce={}",
            self.base_url,
            nonce.as_str()
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "Attester".to_string(),
                message: format!("Request failed: {}", e),
            })?;

        resp.json::<AttestationReport>()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "Attester".to_string(),
                message: format!("Parse error: {}", e),
            })
    }
}
