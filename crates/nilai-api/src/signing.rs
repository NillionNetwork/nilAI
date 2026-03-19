use nilai_infra::crypto::KeyPair;

/// Sign a JSON response string and return the base64 signature.
pub fn sign_response(keypair: &KeyPair, response_json: &str) -> String {
    keypair.sign_message_b64(response_json).unwrap_or_else(|e| {
        tracing::error!("Failed to sign response: {}", e);
        String::new()
    })
}
