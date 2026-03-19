use std::pin::Pin;
use std::sync::Arc;
use futures::stream::{self, Stream, StreamExt};
use bytes::Bytes;
use nilai_domain::chat::ChatRequest;
use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::InferenceClient;

/// Create an SSE stream from the inference client.
/// Each event is formatted as `data: {json}\n\n`.
pub async fn create_sse_stream(
    client: Arc<dyn InferenceClient>,
    base_url: url::Url,
    req: ChatRequest,
) -> NilaiResult<Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>> {
    let byte_stream = client.chat_completion_stream(&base_url, &req).await?;

    // The vLLM client returns raw SSE bytes from the upstream.
    // We pass them through directly since vLLM already formats as SSE.
    let mapped = byte_stream.map(|result| {
        match result {
            Ok(bytes) => Ok(bytes),
            Err(e) => {
                let error_event = format!("data: {{\"error\": \"stream_failed\", \"message\": \"{}\"}}\n\n", e);
                Ok(Bytes::from(error_event))
            }
        }
    });

    Ok(Box::pin(mapped))
}
