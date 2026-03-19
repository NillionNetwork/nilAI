use async_trait::async_trait;

use crate::error::NilaiResult;

#[async_trait]
pub trait CodeExecutor: Send + Sync {
    async fn execute_python(&self, code: &str) -> NilaiResult<String>;
}
