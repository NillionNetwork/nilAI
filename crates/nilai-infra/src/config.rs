use figment::{
    providers::{Env, Format, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NilaiConfig {
    #[serde(default = "default_environment")]
    pub environment: String,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub docs: DocsConfig,
    #[serde(default)]
    pub database: Option<DatabaseConfig>,
    #[serde(default)]
    pub discovery: Option<DiscoveryConfig>,
    #[serde(default)]
    pub redis: Option<RedisConfig>,
    #[serde(default)]
    pub web_search: WebSearchConfig,
    #[serde(default)]
    pub rate_limiting: RateLimitingConfig,
    #[serde(default)]
    pub llm_pricing: LLMPricingConfig,
    #[serde(default)]
    pub nildb: Option<NilDbConfig>,
}

fn default_environment() -> String {
    "mainnet".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthConfig {
    #[serde(default = "default_auth_strategy", alias = "strategy")]
    pub auth_strategy: String,
    #[serde(default)]
    pub nilauth_trusted_root_issuers: Vec<String>,
    #[serde(default)]
    pub credit_api_token: String,
    pub auth_token: Option<String>,
    pub admin_token: Option<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            auth_strategy: default_auth_strategy(),
            nilauth_trusted_root_issuers: vec!["http://nilauth-credit-server:3000".to_string()],
            credit_api_token: String::new(),
            auth_token: None,
            admin_token: None,
        }
    }
}

impl AuthConfig {
    pub fn credit_service_url(&self) -> Option<&str> {
        self.nilauth_trusted_root_issuers
            .first()
            .map(|s| s.as_str())
    }
}

fn default_auth_strategy() -> String {
    "api_key".to_string()
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DocsConfig {
    pub token: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseConfig {
    pub user: String,
    pub password: String,
    pub host: String,
    pub port: u16,
    pub db: String,
}

impl DatabaseConfig {
    pub fn connection_string(&self) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, self.db
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DiscoveryConfig {
    #[serde(default = "default_discovery_url")]
    pub url: String,
}

fn default_discovery_url() -> String {
    "redis://localhost:6379".to_string()
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            url: default_discovery_url(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RedisConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebSearchConfig {
    pub api_key: Option<String>,
    #[serde(default = "default_brave_api_path")]
    pub api_path: String,
    #[serde(default = "default_search_count")]
    pub count: u32,
    #[serde(default = "default_lang")]
    pub lang: String,
    #[serde(default = "default_country")]
    pub country: String,
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: u32,
    #[serde(default = "default_rps")]
    pub rps: u64,
}

fn default_brave_api_path() -> String {
    "https://api.search.brave.com/res/v1/web/search".to_string()
}
fn default_search_count() -> u32 {
    3
}
fn default_lang() -> String {
    "en".to_string()
}
fn default_country() -> String {
    "us".to_string()
}
fn default_timeout() -> f64 {
    20.0
}
fn default_max_concurrent() -> u32 {
    20
}
fn default_rps() -> u64 {
    20
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_path: default_brave_api_path(),
            count: default_search_count(),
            lang: default_lang(),
            country: default_country(),
            timeout: default_timeout(),
            max_concurrent_requests: default_max_concurrent(),
            rps: default_rps(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateLimitingConfig {
    pub user_rate_limit_minute: Option<u64>,
    pub user_rate_limit_hour: Option<u64>,
    pub user_rate_limit_day: Option<u64>,
    pub web_search_rate_limit_minute: Option<u64>,
    pub web_search_rate_limit_hour: Option<u64>,
    pub web_search_rate_limit_day: Option<u64>,
    #[serde(default = "default_concurrent_limits")]
    pub model_concurrent_rate_limit: HashMap<String, u64>,
    pub user_rate_limit: Option<u64>,
    pub web_search_rate_limit: Option<u64>,
}

fn default_concurrent_limits() -> HashMap<String, u64> {
    let mut m = HashMap::new();
    m.insert("default".to_string(), 50);
    m
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            user_rate_limit_minute: Some(100),
            user_rate_limit_hour: Some(1000),
            user_rate_limit_day: Some(10000),
            web_search_rate_limit_minute: Some(6),
            web_search_rate_limit_hour: Some(18),
            web_search_rate_limit_day: Some(72),
            model_concurrent_rate_limit: default_concurrent_limits(),
            user_rate_limit: None,
            web_search_rate_limit: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct LLMPricingConfig {
    #[serde(default)]
    pub default: nilai_domain::usage::LLMPriceConfig,
    #[serde(default)]
    pub models: HashMap<String, nilai_domain::usage::LLMPriceConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NilDbConfig {
    pub nilchain_url: String,
    pub nilauth_url: String,
    pub nodes: Vec<String>,
    pub builder_private_key: String,
    pub collection: String,
}

impl NilaiConfig {
    /// Load configuration from YAML file and environment variables.
    /// YAML is loaded first, then env vars override.
    pub fn load(config_path: Option<&Path>) -> Result<Self, Box<figment::Error>> {
        let mut figment = Figment::new();

        if let Some(path) = config_path {
            figment = figment.merge(Yaml::file(path));
        }

        // Env vars: POSTGRES_* for database, REDIS_* for redis, etc.
        figment = figment
            .merge(
                Env::prefixed("POSTGRES_")
                    .map(|key| format!("database.{}", key.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::prefixed("DISCOVERY_")
                    .map(|key| format!("discovery.{}", key.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::prefixed("REDIS_")
                    .map(|key| format!("redis.{}", key.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::prefixed("WEB_SEARCH_")
                    .map(|key| format!("web_search.{}", key.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::prefixed("NILDB_")
                    .map(|key| format!("nildb.{}", key.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::raw()
                    .only(&["ENVIRONMENT"])
                    .map(|_| "environment".into()),
            )
            .merge(
                Env::raw()
                    .only(&["ADMIN_TOKEN"])
                    .map(|_| "auth.admin_token".into()),
            )
            .merge(
                Env::raw()
                    .only(&["DOCS_TOKEN"])
                    .map(|_| "docs.token".into()),
            )
            .merge(
                Env::raw()
                    .only(&["BRAVE_SEARCH_API"])
                    .map(|_| "web_search.api_key".into()),
            );

        figment.extract().map_err(Box::new)
    }
}
