from fastapi import FastAPI
from nuc.nilauth import NilauthClient
from pydantic import BaseModel
from secp256k1 import PublicKey as NilAuthPublicKey
import base64
from nilai_auth_server.config import NILAUTH_TRUSTED_ROOT_ISSUER

from nuc_helpers import (
    RootToken,
    DelegationToken,
    pay_for_subscription,
    get_wallet_and_private_key,
    get_root_token,
    get_delegation_token,
)

app = FastAPI()

PRIVATE_KEY = "l/SYifzu2Iqc3dsWoWHRP2oSMHwrORY/PDw5fDwtJDQ="  # This is an example private key with funds for testing devnet, and should not be used in production
NILCHAIN_GRPC = "localhost:26649"


class DelegateRequest(BaseModel):
    user_public_key: str


@app.post("/v1/delegate/")
def delegate(request: DelegateRequest) -> DelegationToken:
    """
    Delegate the root token to the delegated key

    Args:
        request: The request body
    """

    server_wallet, server_keypair, server_private_key = get_wallet_and_private_key(
        PRIVATE_KEY
    )
    nilauth_client = NilauthClient(f"http://{NILAUTH_TRUSTED_ROOT_ISSUER}")

    # Pay for the subscription
    pay_for_subscription(
        nilauth_client,
        server_wallet,
        server_keypair,
        server_private_key,
        f"http://{NILCHAIN_GRPC}",
    )

    # Create a root token
    root_token: RootToken = get_root_token(nilauth_client, server_private_key)

    user_public_key = NilAuthPublicKey(
        base64.b64decode(request.user_public_key), raw=True
    )

    delegation_token: DelegationToken = get_delegation_token(
        root_token,
        server_private_key,
        user_public_key,
    )
    return delegation_token


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
