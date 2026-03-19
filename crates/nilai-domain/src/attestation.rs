use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    pub nonce: String,
    pub verifying_key: String,
    pub cpu_attestation: String,
    pub gpu_attestation: String,
}
