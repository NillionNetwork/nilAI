import datetime
from nuc.envelope import NucTokenEnvelope
from nuc.token import Command, NucToken, Did
from nuc.builder import NucTokenBuilder, DelegationBody
from secp256k1 import PrivateKey


def is_expired(token_envelope: NucTokenEnvelope) -> bool:
    """
    Check if a token envelope is expired.

    Args:
        token_envelope (NucTokenEnvelope): The token envelope to check.

    Returns:
        bool: True if the token envelope is expired, False otherwise.
    """
    token: NucToken = token_envelope.token.token
    if token.expires_at is None:
        return False
    return token.expires_at < datetime.datetime.now(datetime.timezone.utc)


def new_root_token(private_key: PrivateKey) -> NucTokenEnvelope:
    """
    Force the creation of a new root token.
    """
    hex_public_key = private_key.pubkey
    if hex_public_key is None:
        raise ValueError("Public key is None")
    hex_public_key = hex_public_key.serialize()
    root_token = NucTokenBuilder(
        body=DelegationBody([]),
        audience=Did(hex_public_key),
        subject=Did(hex_public_key),
        expires_at=datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
        command=Command(["nil", "ai", "generate"]),
    ).build(private_key)
    return NucTokenEnvelope.parse(root_token)
