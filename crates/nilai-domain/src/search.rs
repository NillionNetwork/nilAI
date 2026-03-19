use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub body: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ResultContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultContent {
    pub text: String,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchContext {
    pub prompt: String,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub topic: String,
    pub needs_search: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicResponse {
    pub topics: Vec<Topic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicQuery {
    pub topic: String,
    pub query: String,
}
