use crate::ids::UserId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_rate_limit_day: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_rate_limit_hour: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_rate_limit_minute: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_rate_limit_day: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_rate_limit_hour: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_rate_limit_minute: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_rate_limit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_rate_limit: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserData {
    pub user_id: UserId,
    pub rate_limits: RateLimits,
}
