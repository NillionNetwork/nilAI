use nilai_domain::ports::{
    Attester, CodeExecutor, CreditService, InferenceClient, ModelRegistry, PricingStore,
    QueryLogStore, RateLimiter, SearchProvider, SecretVaultClient, UserStore,
};
use nilai_infra::config::NilaiConfig;
use nilai_infra::crypto::KeyPair;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub keypair: Arc<KeyPair>,
    pub config: Arc<NilaiConfig>,
    pub start_time: Instant,
    // Ports - all optional during incremental migration
    pub model_registry: Option<Arc<dyn ModelRegistry>>,
    pub user_store: Option<Arc<dyn UserStore>>,
    pub query_log_store: Option<Arc<dyn QueryLogStore>>,
    pub rate_limiter: Option<Arc<dyn RateLimiter>>,
    pub pricing_store: Option<Arc<dyn PricingStore>>,
    pub inference_client: Option<Arc<dyn InferenceClient>>,
    pub search_provider: Option<Arc<dyn SearchProvider>>,
    pub attester: Option<Arc<dyn Attester>>,
    pub credit_service: Option<Arc<dyn CreditService>>,
    pub code_executor: Option<Arc<dyn CodeExecutor>>,
    pub secret_vault: Option<Arc<dyn SecretVaultClient>>,
}

impl AppState {
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}
