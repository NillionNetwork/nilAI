from datetime import datetime, timedelta, timezone
from nuc_helpers import (
    get_wallet_and_private_key,
    pay_for_subscription,
    get_root_token,
    get_nilai_public_key,
    get_invocation_token as nuc_helpers_get_invocation_token,
    validate_token,
    InvocationToken,
    RootToken,
    NilAuthPublicKey,
    NilAuthPrivateKey,
    get_delegation_token,
    DelegationToken,
)
from nuc.nilauth import NilauthClient, BlindModule
from nuc.token import Did
from nuc.validate import ValidationParameters, InvocationRequirement

# These correspond to the key used to test with nilAuth. Otherwise the OWNER DID would not match the issuer
DOCUMENT_ID = "bb93f3a4-ba4c-4e20-8f2e-c0650c75a372"
DOCUMENT_OWNER_DID = (
    "did:nil:030923f2e7120c50e42905b857ddd2947f6ecced6bb02aab64e63b28e9e2e06d10"
)


def get_nuc_token(
    usage_limit: int | None = None,
    expires_at: datetime | None = None,
    blind_module: BlindModule = BlindModule.NILAI,
    document_id: str | None = None,
    document_owner_did: str | None = None,
    create_delegation: bool = False,
    create_invalid_delegation: bool = False,
) -> InvocationToken:
    """
    Unified function to get NUC tokens with various configurations.

    Args:
        usage_limit: Optional usage limit for delegation tokens
        expires_at: Optional expiration time for delegation tokens
        blind_module: Optional blind module to use for the token
        create_delegation: Whether to create a delegation token (for rate limiting)
        create_invalid_delegation: Whether to create an invalid delegation chain (for testing)

    Returns:
        InvocationToken: The generated token
    """
    # Constants
    PRIVATE_KEY = "l/SYifzu2Iqc3dsWoWHRP2oSMHwrORY/PDw5fDwtJDQ="  # Example private key for testing devnet
    NILAI_ENDPOINT = "localhost:8080"
    NILAUTH_ENDPOINT = "localhost:30921"
    NILCHAIN_GRPC = "localhost:26649"

    # Setup server private key and client
    server_wallet, server_keypair, server_private_key = get_wallet_and_private_key(
        PRIVATE_KEY
    )

    print("Public key: ", server_private_key.pubkey)
    nilauth_client = NilauthClient(f"http://{NILAUTH_ENDPOINT}")

    if not server_private_key.pubkey:
        raise Exception("Failed to get public key")

    # Pay for subscription
    pay_for_subscription(
        nilauth_client,
        server_wallet,
        server_keypair,
        server_private_key.pubkey,
        f"http://{NILCHAIN_GRPC}",
        blind_module=blind_module,
    )

    # Create root token
    root_token: RootToken = get_root_token(
        nilauth_client,
        server_private_key,
        blind_module=blind_module,
    )

    # Get Nilai public key
    nilai_public_key: NilAuthPublicKey = get_nilai_public_key(
        f"http://{NILAI_ENDPOINT}"
    )

    # Handle delegation token creation if requested
    if create_delegation or create_invalid_delegation:
        # Create user private key and public key
        user_private_key = NilAuthPrivateKey()
        user_public_key = user_private_key.pubkey

        if user_public_key is None:
            raise Exception("Failed to get user public key")

        # Set default values for delegation
        delegation_usage_limit = usage_limit if usage_limit is not None else 3
        delegation_expires_at = (
            expires_at
            if expires_at is not None
            else datetime.now(timezone.utc) + timedelta(minutes=5)
        )

        # Create delegation token
        delegation_token: DelegationToken = get_delegation_token(
            root_token,
            server_private_key,
            user_public_key,
            usage_limit=delegation_usage_limit,
            expires_at=delegation_expires_at,
            document_id=document_id,
            document_owner_did=document_owner_did,
        )

        # Create invalid delegation chain if requested (for testing)
        if create_invalid_delegation:
            delegation_token = get_delegation_token(
                delegation_token,
                user_private_key,
                user_public_key,
                usage_limit=5,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                document_id=document_id,
                document_owner_did=document_owner_did,
            )

        # Create invocation token from delegation
        invocation_token: InvocationToken = nuc_helpers_get_invocation_token(
            delegation_token,
            nilai_public_key,
            user_private_key,
        )
    else:
        # Create invocation token directly from root token
        invocation_token: InvocationToken = nuc_helpers_get_invocation_token(
            root_token,
            nilai_public_key,
            server_private_key,
        )

    # Validate the token
    default_validation_parameters = ValidationParameters.default()
    default_validation_parameters.token_requirements = InvocationRequirement(
        audience=Did(nilai_public_key.serialize())
    )

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        invocation_token.token,
        default_validation_parameters,
    )

    return invocation_token


def get_rate_limited_nuc_token(rate_limit: int = 3) -> InvocationToken:
    """Convenience function for getting rate-limited tokens."""
    return get_nuc_token(
        usage_limit=rate_limit,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        create_delegation=True,
    )


def get_document_id_nuc_token() -> InvocationToken:
    """Convenience function for getting NILDB NUC tokens."""
    print("DOCUMENT_ID", DOCUMENT_ID)
    return get_nuc_token(
        create_delegation=True,
        document_id=DOCUMENT_ID,
        document_owner_did=DOCUMENT_OWNER_DID,
    )


def get_invalid_rate_limited_nuc_token() -> InvocationToken:
    """Convenience function for getting invalid rate-limited tokens (for testing)."""
    return get_nuc_token(
        usage_limit=3,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        create_delegation=True,
        create_invalid_delegation=True,
    )


def get_nildb_nuc_token() -> InvocationToken:
    """Convenience function for getting NILDB NUC tokens."""
    return get_nuc_token(
        blind_module=BlindModule.NILDB,
    )
