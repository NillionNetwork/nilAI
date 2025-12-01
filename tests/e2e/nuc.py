from nilai_api.auth.nuc_helpers import (
    NilAuthPrivateKey,
)

from nuc.builder import NucTokenBuilder
from nuc.token import Did, InvocationBody, Command

from nilai_py import (
    Client,
    DelegationTokenServer,
    DelegationServerConfig,
    AuthType,
    DelegationTokenRequest,
    DelegationTokenResponse,
    PromptDocumentInfo,
)
from openai import DefaultHttpxClient


# These correspond to the key used to test with nilAuth. Otherwise the OWNER DID would not match the issuer
DOCUMENT_ID = "bb93f3a4-ba4c-4e20-8f2e-c0650c75a372"
DOCUMENT_OWNER_DID = (
    "did:nil:030923f2e7120c50e42905b857ddd2947f6ecced6bb02aab64e63b28e9e2e06d10"
)

PRIVATE_KEY = "97f49889fceed88a9cdddb16a161d13f6a12307c2b39163f3c3c397c3c2d2434"  # Example private key for testing devnet


def get_nuc_client(
    usage_limit: int | None = None,
    expires_in: int | None = None,
    document_id: str | None = None,
    document_owner_did: str | None = None,
    create_invalid_delegation: bool = False,
) -> Client:
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
    # We use a key that is not registered to nilauth-credit for the invalid delegation token

    private_key = (
        PRIVATE_KEY
        if not create_invalid_delegation
        else NilAuthPrivateKey().serialize()
    )

    config = DelegationServerConfig(
        expiration_time=expires_in if expires_in else 10 * 60 * 60,  # 10 hours
        token_max_uses=usage_limit if usage_limit else 10,
        prompt_document=PromptDocumentInfo(
            doc_id=document_id if document_id else DOCUMENT_ID,
            owner_did=document_owner_did if document_owner_did else DOCUMENT_OWNER_DID,
        )
        if document_id or document_owner_did
        else None,
    )

    root_server = DelegationTokenServer(
        private_key=private_key,
        config=config,
    )

    # >>> Client initializes a client
    # The client is responsible for making requests to the Nilai API.
    # We do not provide an API key but we set the auth type to DELEGATION_TOKEN
    http_client = DefaultHttpxClient(verify=False)
    client = Client(
        base_url="https://localhost/nuc/v1",
        auth_type=AuthType.DELEGATION_TOKEN,
        http_client=http_client,
        api_key=private_key,
    )

    delegation_request: DelegationTokenRequest = client.get_delegation_request()

    # <<< Server creates a delegation token
    delegation_token: DelegationTokenResponse = root_server.create_delegation_token(
        delegation_request
    )
    # >>> Client sets internally the delegation token
    client.update_delegation(delegation_token)

    return client


def get_rate_limited_nuc_client(rate_limit: int = 3) -> Client:
    return get_nuc_client(
        usage_limit=rate_limit,
        expires_in=5 * 60,  # 5 minutes
    )


def get_rate_limited_nuc_token(rate_limit: int = 3) -> str:
    """Convenience function for getting rate-limited tokens."""
    return get_rate_limited_nuc_client(rate_limit)._get_invocation_token()


def get_invalid_rate_limited_nuc_client() -> Client:
    return get_nuc_client(
        usage_limit=3,
        expires_in=5 * 60,  # 5 minutes
        create_invalid_delegation=True,
    )


def get_invalid_rate_limited_nuc_token() -> str:
    return get_invalid_rate_limited_nuc_client()._get_invocation_token()


def get_document_id_nuc_client() -> Client:
    return get_nuc_client(
        document_id=DOCUMENT_ID,
        document_owner_did=DOCUMENT_OWNER_DID,
    )


def get_document_id_nuc_token() -> str:
    """Convenience function for getting NILDB NUC tokens."""
    return get_document_id_nuc_client()._get_invocation_token()


def get_invalid_nildb_nuc_token() -> str:
    """Convenience function for getting NILDB NUC tokens."""
    private_key = NilAuthPrivateKey(bytes.fromhex(PRIVATE_KEY))
    http_client = DefaultHttpxClient(verify=False)
    client = Client(
        base_url="https://localhost/nuc/v1",
        auth_type=AuthType.API_KEY,
        http_client=http_client,
        api_key=PRIVATE_KEY,
    )

    invocation_token: str = (
        NucTokenBuilder.extending(client.root_token)
        .body(InvocationBody(args={}))
        .audience(Did(client.nilai_public_key.serialize()))
        .command(Command(["nil", "db", "generate"]))
        .build(private_key)
    )
    return invocation_token
