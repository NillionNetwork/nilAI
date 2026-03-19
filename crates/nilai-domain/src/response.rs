use crate::search::Source;
use serde::{Deserialize, Serialize};

/// Simplified ResponseRequest - the full OpenAI Responses API is complex.
/// We define the fields we actually use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<bool>,
}

/// A signed response wrapping the OpenAI Response type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedResponse {
    #[serde(flatten)]
    pub response: serde_json::Value,
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<Source>>,
}
