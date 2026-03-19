use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::UserId;
use crate::usage::{QueryLog, Usage};

#[async_trait]
pub trait QueryLogStore: Send + Sync {
    async fn log_query(&self, log: &QueryLog) -> NilaiResult<()>;
    async fn get_user_usage(&self, user_id: &UserId) -> NilaiResult<Usage>;
}
