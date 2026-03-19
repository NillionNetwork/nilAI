use nilai_domain::error::NilaiResult;
use nilai_domain::ids::{Credits, LockId, UserId};
use nilai_domain::ports::CreditService;
use std::sync::Arc;

/// Metering context for credit lock/settle flow.
/// Call `lock()` before processing, `settle()` after.
pub struct MeteringContext {
    credit_service: Arc<dyn CreditService>,
    lock_id: Option<LockId>,
    user_id: UserId,
}

impl MeteringContext {
    pub fn new(credit_service: Arc<dyn CreditService>, user_id: UserId) -> Self {
        Self {
            credit_service,
            lock_id: None,
            user_id,
        }
    }

    /// Lock estimated credits before inference.
    pub async fn lock(&mut self, estimated_cost: f64) -> NilaiResult<()> {
        let lock_id = self
            .credit_service
            .lock_credits(&self.user_id, Credits::new(estimated_cost))
            .await?;
        self.lock_id = Some(lock_id);
        Ok(())
    }

    /// Settle actual cost after inference completes.
    pub async fn settle(&self, actual_cost: f64) -> NilaiResult<()> {
        if let Some(ref lock_id) = self.lock_id {
            self.credit_service
                .settle_credits(lock_id, Credits::new(actual_cost))
                .await?;
        }
        Ok(())
    }

    pub fn lock_id(&self) -> Option<&LockId> {
        self.lock_id.as_ref()
    }
}

/// No-op metering context for requests that skip metering (e.g., docs token).
pub struct NoOpMeteringContext;

impl NoOpMeteringContext {
    pub fn lock_id_str(&self) -> &str {
        "noop-lock-id"
    }
}
