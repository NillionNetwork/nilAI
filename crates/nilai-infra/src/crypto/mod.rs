use std::fs;
use std::path::Path;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use fs2::FileExt;
use k256::ecdsa::signature::{Signer, Verifier};
use k256::ecdsa::{Signature, SigningKey, VerifyingKey};
use k256::elliptic_curve::rand_core::OsRng;

use nilai_domain::error::{NilaiError, NilaiResult};
use nilai_domain::ids::PublicKeyB64;

pub struct KeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    b64_public_key: PublicKeyB64,
}

impl KeyPair {
    /// Generate or load a key pair from file, with file locking.
    pub fn generate_or_load(key_path: &str) -> NilaiResult<Self> {
        let lock_path = format!("{}.lock", key_path);

        // Ensure lock file exists
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&lock_path)
            .map_err(|e| NilaiError::CryptoError(format!("Failed to create lock file: {}", e)))?;

        let lock_file = fs::File::open(&lock_path)
            .map_err(|e| NilaiError::CryptoError(format!("Failed to open lock file: {}", e)))?;
        lock_file
            .lock_exclusive()
            .map_err(|e| NilaiError::CryptoError(format!("Failed to acquire lock: {}", e)))?;

        let signing_key = if Path::new(key_path).exists() {
            let key_bytes = fs::read(key_path)
                .map_err(|e| NilaiError::CryptoError(format!("Failed to read key file: {}", e)))?;
            if key_bytes.is_empty() {
                return Err(NilaiError::CryptoError(
                    "Private key file is empty or corrupted".to_string(),
                ));
            }
            SigningKey::from_bytes(key_bytes.as_slice().into())
                .map_err(|e| NilaiError::CryptoError(format!("Invalid private key: {}", e)))?
        } else {
            let signing_key = SigningKey::random(&mut OsRng);
            fs::write(key_path, signing_key.to_bytes())
                .map_err(|e| NilaiError::CryptoError(format!("Failed to write key file: {}", e)))?;
            signing_key
        };

        lock_file
            .unlock()
            .map_err(|e| NilaiError::CryptoError(format!("Failed to release lock: {}", e)))?;

        let verifying_key = *signing_key.verifying_key();
        let compressed = verifying_key.to_encoded_point(true);
        let b64 = BASE64.encode(compressed.as_bytes());

        Ok(Self {
            signing_key,
            verifying_key,
            b64_public_key: PublicKeyB64::new(b64),
        })
    }

    /// Sign a message (UTF-8 string). Returns DER-encoded signature bytes.
    pub fn sign_message(&self, message: &str) -> NilaiResult<Vec<u8>> {
        let signature: Signature = self.signing_key.sign(message.as_bytes());
        Ok(signature.to_der().to_bytes().to_vec())
    }

    /// Sign and return base64-encoded DER signature.
    pub fn sign_message_b64(&self, message: &str) -> NilaiResult<String> {
        let sig_bytes = self.sign_message(message)?;
        Ok(BASE64.encode(&sig_bytes))
    }

    /// Verify a DER-encoded signature against a message.
    pub fn verify_signature(&self, message: &str, der_bytes: &[u8]) -> bool {
        match Signature::from_der(der_bytes) {
            Ok(sig) => self.verifying_key.verify(message.as_bytes(), &sig).is_ok(),
            Err(_) => false,
        }
    }

    /// Get the base64-encoded public key.
    pub fn public_key_b64(&self) -> &PublicKeyB64 {
        &self.b64_public_key
    }

    /// Get the raw verifying key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("test_key.key");
        let kp = KeyPair::generate_or_load(key_path.to_str().unwrap()).unwrap();

        let message = "hello world";
        let sig = kp.sign_message(message).unwrap();
        assert!(kp.verify_signature(message, &sig));
        assert!(!kp.verify_signature("wrong message", &sig));
    }

    #[test]
    fn test_key_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("test_key.key");
        let path_str = key_path.to_str().unwrap();

        let kp1 = KeyPair::generate_or_load(path_str).unwrap();
        let kp2 = KeyPair::generate_or_load(path_str).unwrap();

        assert_eq!(kp1.public_key_b64().as_str(), kp2.public_key_b64().as_str());
    }

    #[test]
    fn test_b64_signature() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("test_key.key");
        let kp = KeyPair::generate_or_load(key_path.to_str().unwrap()).unwrap();

        let sig = kp.sign_message_b64("test").unwrap();
        assert!(!sig.is_empty());
        // Verify it's valid base64
        BASE64.decode(&sig).unwrap();
    }
}
