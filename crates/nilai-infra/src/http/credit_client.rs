use async_trait::async_trait;
use reqwest::Client;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::{ApiKey, Credits, LockId, UserId};
use nilai_domain::ports::CreditService;

pub struct NilauthCreditClient {
    client: Client,
    base_url: String,
    api_token: String,
}

impl NilauthCreditClient {
    pub fn new(base_url: String, api_token: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url,
            api_token,
        }
    }
}

#[async_trait]
impl CreditService for NilauthCreditClient {
    async fn validate_credential(
        &self,
        credential: &ApiKey,
        is_public: bool,
    ) -> NilaiResult<UserId> {
        // TODO: Reverse-engineer the exact nilauth-credit-middleware HTTP protocol
        let resp = self
            .client
            .post(format!("{}/validate", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&serde_json::json!({
                "credential": credential.as_str(),
                "is_public": is_public,
            }))
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Validate credential failed: {}", e),
            })?;

        if !resp.status().is_success() {
            return Err(NilaiError::Unauthorized("Invalid credential".to_string()));
        }

        #[derive(serde::Deserialize)]
        struct ValidateResponse {
            user_id: String,
        }

        let body: ValidateResponse =
            resp.json().await.map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Parse error: {}", e),
            })?;

        Ok(UserId::new(body.user_id))
    }

    async fn lock_credits(&self, user_id: &UserId, estimated_cost: Credits) -> NilaiResult<LockId> {
        // TODO: Reverse-engineer the exact lock protocol
        let resp = self
            .client
            .post(format!("{}/lock", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&serde_json::json!({
                "user_id": user_id.as_str(),
                "estimated_cost": estimated_cost.as_f64(),
            }))
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Lock credits failed: {}", e),
            })?;

        #[derive(serde::Deserialize)]
        struct LockResponse {
            lock_id: String,
        }

        let body: LockResponse = resp.json().await.map_err(|e| NilaiError::ExternalService {
            service: "nilauth".to_string(),
            message: format!("Parse error: {}", e),
        })?;

        Ok(LockId::new(body.lock_id))
    }

    async fn settle_credits(&self, lock_id: &LockId, actual_cost: Credits) -> NilaiResult<()> {
        // TODO: Reverse-engineer the exact settle protocol
        let resp = self
            .client
            .post(format!("{}/settle", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&serde_json::json!({
                "lock_id": lock_id.as_str(),
                "actual_cost": actual_cost.as_f64(),
            }))
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Settle credits failed: {}", e),
            })?;

        if !resp.status().is_success() {
            return Err(NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: "Settle failed".to_string(),
            });
        }

        Ok(())
    }
}
