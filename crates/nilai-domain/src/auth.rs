use crate::user::UserData;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationInfo {
    pub user: UserData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_rate_limit: Option<TokenRateLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_document: Option<PromptDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthStrategy {
    ApiKey,
    Jwt,
    Nuc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRateLimit {
    pub signature: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_limit: Option<u64>,
}

impl TokenRateLimit {
    pub fn ms_remaining(&self) -> u64 {
        let now = chrono::Utc::now();
        let remaining = self.expires_at - now;
        remaining.num_milliseconds().max(0) as u64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRateLimits {
    pub limits: Vec<TokenRateLimit>,
}

impl TokenRateLimits {
    pub fn last(&self) -> Option<&TokenRateLimit> {
        self.limits.last()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDocument {
    pub document_id: String,
    pub owner_did: String,
}
