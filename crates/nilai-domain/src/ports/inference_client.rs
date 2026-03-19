use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;

use crate::chat::{ChatCompletion, ChatRequest};
use crate::error::NilaiResult;

#[async_trait]
pub trait InferenceClient: Send + Sync {
    async fn chat_completion(
        &self,
        base_url: &url::Url,
        req: &ChatRequest,
    ) -> NilaiResult<ChatCompletion>;

    async fn chat_completion_stream(
        &self,
        base_url: &url::Url,
        req: &ChatRequest,
    ) -> NilaiResult<Pin<Box<dyn Stream<Item = NilaiResult<Bytes>> + Send>>>;
}
