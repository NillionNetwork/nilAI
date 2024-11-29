from base64 import b64decode

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec

from nilai.crypto import generate_key_pair, sign_message, verify_signature


def test_generate_key_pair():
    # Generate keys
    private_key, public_key, verifying_key = generate_key_pair()

    # Check private_key and public_key are instances of the expected types
    assert isinstance(private_key, ec.EllipticCurvePrivateKey)
    assert isinstance(public_key, ec.EllipticCurvePublicKey)

    # Ensure the verifying_key is a valid PEM-encoded public key
    decoded_key = b64decode(verifying_key)
    assert b"BEGIN PUBLIC KEY" in decoded_key
    assert b"END PUBLIC KEY" in decoded_key


def test_sign_message():
    # Generate keys
    private_key, _, _ = generate_key_pair()

    # Message to sign
    message = "Test message"

    # Generate a signature
    signature = sign_message(private_key, message)

    # Ensure the signature is a byte string
    assert isinstance(signature, bytes)
    assert len(signature) > 0


def test_verify_signature_valid():
    # Generate keys
    private_key, public_key, _ = generate_key_pair()

    # Message to sign and verify
    message = "Valid message"

    # Generate a signature
    signature = sign_message(private_key, message)

    # Verify the signature (should not raise an exception)
    try:
        verify_signature(public_key, message, signature)
    except Exception as e:
        pytest.fail(f"Verification failed: {e}")


def test_verify_signature_invalid_message():
    # Generate keys
    private_key, public_key, _ = generate_key_pair()

    # Original and tampered messages
    message = "Original message"
    tampered_message = "Tampered message"

    # Generate a signature for the original message
    signature = sign_message(private_key, message)

    # Verify the tampered message (should raise InvalidSignature)
    with pytest.raises(InvalidSignature):
        verify_signature(public_key, tampered_message, signature)


def test_verify_signature_invalid_signature():
    # Generate keys
    private_key, public_key, _ = generate_key_pair()

    # Message
    message = "Message"

    # Generate a valid signature
    signature = sign_message(private_key, message)

    # Tamper with the signature
    tampered_signature = signature[:-1] + b"\x00"

    # Verify the tampered signature (should raise InvalidSignature)
    with pytest.raises(InvalidSignature):
        verify_signature(public_key, message, tampered_signature)
