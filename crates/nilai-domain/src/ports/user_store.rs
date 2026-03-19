use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::UserId;
use crate::user::{RateLimits, UserData};

#[async_trait]
pub trait UserStore: Send + Sync {
    async fn get_user(&self, user_id: &UserId) -> NilaiResult<Option<UserData>>;
    async fn insert_user(&self, user_id: &UserId) -> NilaiResult<UserData>;
    async fn update_rate_limits(&self, user_id: &UserId, limits: &RateLimits) -> NilaiResult<()>;
}
