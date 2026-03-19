use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::Client;

use nilai_domain::chat::{ChatCompletion, ChatRequest};
use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::InferenceClient;

pub struct VllmClient {
    client: Client,
}

impl VllmClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl Default for VllmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceClient for VllmClient {
    async fn chat_completion(
        &self,
        base_url: &url::Url,
        req: &ChatRequest,
    ) -> NilaiResult<ChatCompletion> {
        let url = base_url
            .join("/v1/chat/completions")
            .map_err(|e| NilaiError::InferenceError(format!("Invalid URL: {}", e)))?;

        let resp = self
            .client
            .post(url)
            .json(req)
            .send()
            .await
            .map_err(|e| NilaiError::InferenceError(format!("vLLM request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(NilaiError::InferenceError(format!(
                "vLLM returned {}: {}",
                status, body
            )));
        }

        resp.json::<ChatCompletion>().await.map_err(|e| {
            NilaiError::InferenceError(format!("Failed to parse vLLM response: {}", e))
        })
    }

    async fn chat_completion_stream(
        &self,
        base_url: &url::Url,
        req: &ChatRequest,
    ) -> NilaiResult<Pin<Box<dyn Stream<Item = NilaiResult<Bytes>> + Send>>> {
        let url = base_url
            .join("/v1/chat/completions")
            .map_err(|e| NilaiError::InferenceError(format!("Invalid URL: {}", e)))?;

        let mut stream_req = req.clone();
        stream_req.stream = Some(true);

        let resp = self
            .client
            .post(url)
            .json(&stream_req)
            .send()
            .await
            .map_err(|e| {
                NilaiError::InferenceError(format!("vLLM stream request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(NilaiError::InferenceError(format!(
                "vLLM returned {}: {}",
                status, body
            )));
        }

        let stream = resp.bytes_stream().map(|result| {
            result.map_err(|e| NilaiError::InferenceError(format!("Stream error: {}", e)))
        });

        Ok(Box::pin(stream))
    }
}
