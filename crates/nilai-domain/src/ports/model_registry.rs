use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::NilaiResult;
use crate::ids::{ModelId, ModelName};
use crate::model::ModelEndpoint;

#[async_trait]
pub trait ModelRegistry: Send + Sync {
    async fn get_model(&self, model_id: &ModelId) -> NilaiResult<Option<ModelEndpoint>>;
    async fn discover_models(
        &self,
        name: Option<&ModelName>,
        feature: Option<&str>,
    ) -> NilaiResult<HashMap<String, ModelEndpoint>>;
    async fn register_model(&self, endpoint: &ModelEndpoint) -> NilaiResult<String>;
    async fn unregister_model(&self, model_id: &ModelId) -> NilaiResult<()>;
}
