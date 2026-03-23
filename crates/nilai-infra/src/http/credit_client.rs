use async_trait::async_trait;
use reqwest::Client;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::{ApiKey, Credits, LockId, UserId};
use nilai_domain::ports::CreditService;

pub struct NilauthCreditClient {
    client: Client,
    base_url: String,
}

impl NilauthCreditClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url,
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
        let resp = self
            .client
            .post(format!("{}/v1/credential/validate", self.base_url))
            .json(&serde_json::json!({
                "credential_key": credential.as_str(),
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
            #[allow(dead_code)]
            is_valid: bool,
            #[allow(dead_code)]
            is_public: bool,
        }

        let body: ValidateResponse =
            resp.json().await.map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Parse error: {}", e),
            })?;

        Ok(UserId::new(body.user_id))
    }

    async fn lock_credits(
        &self,
        credential: &ApiKey,
        estimated_cost: Credits,
        is_public: bool,
    ) -> NilaiResult<LockId> {
        let endpoint = if is_public {
            "v1/balance/lock/public"
        } else {
            "v1/balance/lock/private"
        };

        let resp = self
            .client
            .post(format!("{}/{}", self.base_url, endpoint))
            .json(&serde_json::json!({
                "credential": credential.as_str(),
                "amount": estimated_cost.as_f64(),
            }))
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: format!("Lock credits failed: {}", e),
            })?;

        if !resp.status().is_success() {
            return Err(NilaiError::ExternalService {
                service: "nilauth".to_string(),
                message: "Lock credits failed".to_string(),
            });
        }

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
        let resp = self
            .client
            .post(format!("{}/v1/balance/unlock", self.base_url))
            .json(&serde_json::json!({
                "lock_id": lock_id.as_str(),
                "real_cost": actual_cost.as_f64(),
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
