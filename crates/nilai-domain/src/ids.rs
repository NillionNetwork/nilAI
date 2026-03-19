use serde::{Deserialize, Serialize};
use std::fmt;

/// Macro to define a newtype string identifier with common derives and impls.
macro_rules! newtype_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

newtype_id!(
    /// User identifier
    UserId
);

newtype_id!(
    /// Raw API key from Bearer header
    ApiKey
);

newtype_id!(
    /// Credit metering lock
    LockId
);

newtype_id!(
    /// Admin token
    AdminToken
);

newtype_id!(
    /// Docs bypass token
    DocsToken
);

newtype_id!(
    /// Model UUID
    ModelId
);

newtype_id!(
    /// Model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
    ModelName
);

newtype_id!(
    /// 64-char hex attestation nonce
    Nonce
);

newtype_id!(
    /// Base64-encoded ECDSA signature
    Signature
);

newtype_id!(
    /// Base64-encoded public key
    PublicKeyB64
);

newtype_id!(
    /// nilDB document identifier
    DocumentId
);

newtype_id!(
    /// Decentralized identifier
    Did
);

// ---------------------------------------------------------------------------
// Value objects
// ---------------------------------------------------------------------------

/// Token cost in credits
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Credits(f64);

impl Credits {
    pub fn new(value: f64) -> Self {
        debug_assert!(value >= 0.0);
        Self(value)
    }

    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl std::ops::Add for Credits {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

/// Price per million tokens
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PricePerMillion(f64);

impl PricePerMillion {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn cost_for(&self, token_count: u64) -> Credits {
        Credits::new(self.0 * token_count as f64 / 1_000_000.0)
    }
}

/// Token count
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCount(u64);

impl TokenCount {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}
