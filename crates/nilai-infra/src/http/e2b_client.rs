use async_trait::async_trait;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::CodeExecutor;

pub struct E2bClient {
    // TODO: Integrate with e2b Rust SDK
}

impl E2bClient {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for E2bClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CodeExecutor for E2bClient {
    async fn execute_python(&self, _code: &str) -> NilaiResult<String> {
        Err(NilaiError::Internal(
            "e2b code execution not yet implemented".to_string(),
        ))
    }
}
