import os
import fcntl
from base64 import b64encode

from secp256k1 import PrivateKey, PublicKey

PRIVATE_KEY_PATH = "private_key.key"


def generate_key_pair() -> tuple[PrivateKey, PublicKey, str]:
    """
    Generate or load a key pair safely, preventing concurrent access using fcntl.

    Returns:
        tuple[PrivateKey, PublicKey, str]: Private key, public key, and base64-encoded public key.
    """
    private_key: PrivateKey

    # Use a separate lock file to avoid corrupting the key file itself
    lock_path = PRIVATE_KEY_PATH + ".lock"

    # Ensure the lock file exists
    open(lock_path, "a").close()

    with open(lock_path, "r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if os.path.exists(PRIVATE_KEY_PATH):
            with open(PRIVATE_KEY_PATH, "rb") as f:
                private_key_bytes: bytes = f.read()
                if not private_key_bytes:
                    raise ValueError("Private key file is empty or corrupted.")
                private_key = PrivateKey(private_key_bytes)
        else:
            private_key = PrivateKey()
            with open(PRIVATE_KEY_PATH, "wb") as f:
                private_key_bytes: bytes = private_key.private_key  # type: ignore
                f.write(private_key_bytes)

        # Release the lock
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    public_key = private_key.pubkey
    if public_key is None:
        raise ValueError("Keypair generation failed: Public key is None")

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
