from fastapi import FastAPI
from nuc.payer import Payer
from nuc.builder import NucTokenBuilder
from nuc.nilauth import NilauthClient
from nuc.envelope import NucTokenEnvelope
from nuc.token import Command, Did
from cosmpy.crypto.keypairs import PrivateKey as NilchainPrivateKey
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from pydantic import BaseModel
from secp256k1 import PrivateKey as NilAuthPrivateKey, PublicKey as NilAuthPublicKey
import base64
import datetime
from functools import lru_cache
from nilai_auth_server.config import NILAUTH_TRUSTED_ROOT_ISSUERS

app = FastAPI()

PRIVATE_KEY = "l/SYifzu2Iqc3dsWoWHRP2oSMHwrORY/PDw5fDwtJDQ="
NILCHAIN_GRPC = "http://localhost:26649"


@lru_cache(maxsize=1)
def get_wallet_and_private_key():
    """
    Get the wallet and private key

    Returns:
        wallet: The wallet
        keypair: The keypair
        private_key: The private key
    """
    # FIXME: Use a real wallet, and don't hardcode the private key
    keypair = NilchainPrivateKey(PRIVATE_KEY)
    wallet = LocalWallet(keypair, prefix="nillion")
    private_key = NilAuthPrivateKey(base64.b64decode(PRIVATE_KEY))
    return wallet, keypair, private_key


def get_root_token(nilauth_client, private_key) -> NucTokenEnvelope:
    """
    Get the root token from nilauth

    Args:
        nilauth_client: The nilauth client
        private_key: The private key of the user

    Returns:
        The root token
    """

    ## Getting the root token from nilauth
    root_token: str = nilauth_client.request_token(key=private_key)

    root_token_envelope = NucTokenEnvelope.parse(root_token)

    return root_token_envelope


def get_unil_balance(address) -> int:
    """
    Get the UNIL balance of the user

    Args:
        address: The address of the user
    """

    cfg = NetworkConfig(
        chain_id="nillion-chain-devnet",
        url="grpc+" + NILCHAIN_GRPC,
        fee_minimum_gas_price=1,
        fee_denomination="unil",
        staking_denomination="unil",
    )
    ledger_client = LedgerClient(cfg)
    balance = ledger_client.query_bank_balance(address, "unil")
    return balance


def pay_for_subscription(nilauth_client, keypair, private_key):
    """
    Pay for the subscription using the Nilchain keypair if the user is not subscribed

    Args:
        nilauth_client: The nilauth client
        keypair: The Nilchain keypair
        private_key: The NilAuth private key of the user
    """

    payer = Payer(
        wallet_private_key=keypair,
        chain_id="nillion-chain-devnet",
        grpc_endpoint=NILCHAIN_GRPC,
        gas_limit=1000000000000,
    )

    # Pretty print the subscription details
    subscription_details = nilauth_client.subscription_status(private_key)
    print(f"IS SUBSCRIBED: {subscription_details.subscribed}")

    if not subscription_details.subscribed:
        print("[>] Paying for subscription")
        nilauth_client.pay_subscription(
            key=private_key,
            payer=payer,
        )
    else:
        print("[>] Subscription is already paid for")

        print(
            f"EXPIRES IN: {subscription_details.details.expires_at - datetime.datetime.now(datetime.timezone.utc)}"
        )
        print(
            f"CAN BE RENEWED IN: {subscription_details.details.renewable_at - datetime.datetime.now(datetime.timezone.utc)}"
        )


class DelegateRequest(BaseModel):
    user_public_key: str


class DelegationToken(BaseModel):
    token: str


@app.post("/v1/delegate/")
def delegate(request: DelegateRequest) -> DelegationToken:
    """
    Delegate the root token to the delegated key

    Args:
        request: The request body
    """

    wallet, keypair, private_key = get_wallet_and_private_key()
    balance = get_unil_balance(wallet.address())

    print(f"Wallet balance: {balance} unil")
    print("[>] Creating nilauth client")
    nilauth_clients = [NilauthClient(url) for url in NILAUTH_TRUSTED_ROOT_ISSUERS]
    nilauth_client = nilauth_clients[0]
    pay_for_subscription(nilauth_client, keypair, private_key)

    root_token = get_root_token(nilauth_client, private_key)

    user_public_key = NilAuthPublicKey(
        base64.b64decode(request.user_public_key), raw=True
    )

    delegated_token = (
        NucTokenBuilder.extending(root_token)
        .audience(Did(user_public_key.serialize()))
        .command(Command(["nil", "ai", "generate"]))
        .build(private_key)
    )
    return DelegationToken(token=delegated_token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
