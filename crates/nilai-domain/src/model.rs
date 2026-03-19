use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    #[serde(default = "ModelMetadata::default_id")]
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub source: String,
    pub supported_features: Vec<String>,
    pub tool_support: bool,
    #[serde(default)]
    pub multimodal_support: bool,
}

impl ModelMetadata {
    fn default_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpoint {
    pub url: String,
    pub metadata: ModelMetadata,
}
