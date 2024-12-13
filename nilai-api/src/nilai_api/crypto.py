from base64 import b64encode

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def generate_key_pair():
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    verifying_key = b64encode(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).decode()
    return private_key, public_key, verifying_key


def sign_message(private_key, message):
    return private_key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))


def verify_signature(public_key, message, signature):
    public_key.verify(signature, message.encode(), ec.ECDSA(hashes.SHA256()))
