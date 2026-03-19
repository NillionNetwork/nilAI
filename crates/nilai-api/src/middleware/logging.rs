use nilai_domain::ports::QueryLogStore;
use nilai_domain::usage::QueryLog;
use std::sync::Arc;
use std::time::Instant;

/// Context for tracking query metrics and logging to database.
pub struct QueryLogContext {
    log_store: Arc<dyn QueryLogStore>,
    pub user_id: String,
    pub lock_id: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub web_searches: u64,
    start_time: Instant,
}

impl QueryLogContext {
    pub fn new(log_store: Arc<dyn QueryLogStore>, user_id: String) -> Self {
        Self {
            log_store,
            user_id,
            lock_id: String::new(),
            model: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            web_searches: 0,
            start_time: Instant::now(),
        }
    }

    pub fn set_usage(&mut self, prompt_tokens: u64, completion_tokens: u64, web_searches: u64) {
        self.prompt_tokens = prompt_tokens;
        self.completion_tokens = completion_tokens;
        self.total_tokens = prompt_tokens + completion_tokens;
        self.web_searches = web_searches;
    }

    /// Commit the log entry to the database. Intended to be called as a background task.
    pub async fn commit(self) {
        let log = QueryLog {
            user_id: self.user_id,
            model: self.model,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            web_searches: self.web_searches,
            lock_id: self.lock_id,
        };

        if let Err(e) = self.log_store.log_query(&log).await {
            tracing::error!("Failed to commit query log: {}", e);
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}
