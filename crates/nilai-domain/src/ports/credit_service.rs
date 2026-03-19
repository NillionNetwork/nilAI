use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::{ApiKey, Credits, LockId, UserId};

#[async_trait]
pub trait CreditService: Send + Sync {
    async fn validate_credential(
        &self,
        credential: &ApiKey,
        is_public: bool,
    ) -> NilaiResult<UserId>;
    async fn lock_credits(&self, user_id: &UserId, estimated_cost: Credits) -> NilaiResult<LockId>;
    async fn settle_credits(&self, lock_id: &LockId, actual_cost: Credits) -> NilaiResult<()>;
}
