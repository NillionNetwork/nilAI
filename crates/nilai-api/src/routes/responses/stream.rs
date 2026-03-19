// Streaming for the Responses API is handled inline in handler.rs
// via reqwest byte stream passthrough, since the vLLM server
// already produces SSE-formatted events.
