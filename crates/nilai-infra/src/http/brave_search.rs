use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ports::SearchProvider;
use nilai_domain::search::SearchResult;

pub struct BraveSearchClient {
    client: Client,
    api_key: String,
    api_path: String,
}

impl BraveSearchClient {
    pub fn new(api_key: String, api_path: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            api_path,
        }
    }
}

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    results: Vec<BraveWebResult>,
}

#[derive(Deserialize)]
struct BraveWebResult {
    title: String,
    description: String,
    url: String,
}

#[async_trait]
impl SearchProvider for BraveSearchClient {
    async fn search(&self, query: &str, count: u32) -> NilaiResult<Vec<SearchResult>> {
        let resp = self
            .client
            .get(&self.api_path)
            .header("X-Subscription-Token", &self.api_key)
            .query(&[("q", query), ("count", &count.to_string())])
            .send()
            .await
            .map_err(|e| NilaiError::ExternalService {
                service: "Brave Search".to_string(),
                message: format!("Request failed: {}", e),
            })?;

        if !resp.status().is_success() {
            return Err(NilaiError::ExternalService {
                service: "Brave Search".to_string(),
                message: format!("HTTP {}", resp.status()),
            });
        }

        let brave_resp: BraveResponse =
            resp.json().await.map_err(|e| NilaiError::ExternalService {
                service: "Brave Search".to_string(),
                message: format!("Parse error: {}", e),
            })?;

        Ok(brave_resp
            .web
            .map(|w| {
                w.results
                    .into_iter()
                    .map(|r| SearchResult {
                        title: r.title,
                        body: r.description,
                        url: r.url,
                        content: None,
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
}
