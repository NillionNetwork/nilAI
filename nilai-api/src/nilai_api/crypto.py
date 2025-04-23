from base64 import b64encode
import os

from secp256k1 import PrivateKey, PublicKey

PRIVATE_KEY_PATH = "private_key.key"


def generate_key_pair() -> tuple[PrivateKey, PublicKey, str]:
    """
    Generate a new key pair and return the private key, public key, and base64 encoded public key.

    Returns:
        tuple[PrivateKey, PublicKey, str]: A tuple containing the private key, public key, and base64 encoded public key.
    """
    private_key: PrivateKey
    if os.path.exists(PRIVATE_KEY_PATH):
        with open(PRIVATE_KEY_PATH, "rb") as f:
            private_key = PrivateKey(f.read())
    else:
        private_key = PrivateKey()
        with open(PRIVATE_KEY_PATH, "wb") as f:
            private_key_bytes: bytes = private_key.private_key  # type: ignore
            f.write(private_key_bytes)

    public_key = private_key.pubkey
    if public_key is None:
        raise ValueError("Keypair generation failed:Public key is None")
    b64_public_key: str = b64encode(public_key.serialize()).decode()

    return private_key, public_key, b64_public_key


def sign_message(private_key: PrivateKey, message: str) -> bytes:
    """
    Sign a message using the private key.

    Args:
        private_key (PrivateKey): The private key to sign the message with.
        message (str): The message to sign.

    Returns:
        bytes: The signature of the message.
    """
    signature = private_key.ecdsa_sign(message.encode())
    serialized_signature: bytes = private_key.ecdsa_serialize(signature)
    return serialized_signature


def verify_signature(public_key: PublicKey, message: str, signature: bytes) -> bool:
    """
    Verify a signature using the public key.

    Args:
        public_key (PublicKey): The public key to verify the signature with.
        message (str): The message to verify the signature with.
        signature (bytes): The signature to verify.

    Returns:
        bool: True if the signature is valid, False otherwise.
    """
    sig = public_key.ecdsa_deserialize(signature)
    return public_key.ecdsa_verify(message.encode(), sig)
