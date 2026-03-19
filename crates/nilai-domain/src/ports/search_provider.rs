use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::search::SearchResult;

#[async_trait]
pub trait SearchProvider: Send + Sync {
    async fn search(&self, query: &str, count: u32) -> NilaiResult<Vec<SearchResult>>;
}
