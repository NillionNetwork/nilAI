import base64
import datetime
import logging
from functools import lru_cache
from typing import Tuple
import httpx

# Importing the pydantic library dependencies
from pydantic import BaseModel

# Importing the secp256k1 library dependencies
from secp256k1 import PrivateKey as NilAuthPrivateKey, PublicKey as NilAuthPublicKey

# Importing the nuc library dependencies
from nuc.payer import Payer
from nuc.builder import NucTokenBuilder
from nuc.nilauth import NilauthClient
from nuc.envelope import NucTokenEnvelope
from nuc.token import Command, Did, InvocationBody
from nuc.validate import NucTokenValidator, ValidationParameters

# Importing the cosmpy library dependencies
from cosmpy.crypto.keypairs import PrivateKey as NilchainPrivateKey
from cosmpy.aerial.wallet import LocalWallet, Address
from cosmpy.aerial.client import LedgerClient, NetworkConfig

logger = logging.getLogger(__name__)

## Pydantic models


class RootToken(BaseModel):
    token: str


class DelegationToken(BaseModel):
    token: str


class InvocationToken(BaseModel):
    token: str


## Helpers
@lru_cache(maxsize=1)
def get_wallet_and_private_key(
    private_key_bytes: str | bytes | None = None,
) -> Tuple[LocalWallet, NilchainPrivateKey, NilAuthPrivateKey]:
    """
    Get the wallet and private key

    Args:
        private_key_bytes: The private key bytes to use for the wallet

    Returns:
        wallet: The wallet
        keypair: The keypair
        private_key: The private key
    """
    keypair = NilchainPrivateKey(private_key_bytes)
    wallet = LocalWallet(keypair, prefix="nillion")
    private_key = NilAuthPrivateKey(base64.b64decode(keypair.private_key))
    return wallet, keypair, private_key


def get_root_token(
    nilauth_client: NilauthClient, private_key: NilAuthPrivateKey
) -> RootToken:
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

    return RootToken(token=root_token)


def get_unil_balance(address: Address, grpc_endpoint: str) -> int:
    """
    Get the UNIL balance of the user

    Args:
        address: The address of the user
        grpc_endpoint: The endpoint of the grpc server

    Returns:
        The balance of the user in UNIL
    """
    print("grpc_endpoint", grpc_endpoint)

    cfg = NetworkConfig(
        chain_id="nillion-chain-devnet",
        url="grpc+" + grpc_endpoint,
        fee_minimum_gas_price=1,
        fee_denomination="unil",
        staking_denomination="unil",
    )
    ledger_client = LedgerClient(cfg)
    balance = ledger_client.query_bank_balance(address, "unil")  # type: ignore
    return balance


def pay_for_subscription(
    nilauth_client: NilauthClient,
    wallet: LocalWallet,
    keypair: NilchainPrivateKey,
    private_key: NilAuthPrivateKey,
    grpc_endpoint: str,
) -> None:
    """
    Pay for the subscription using the Nilchain keypair if the user is not subscribed

    Args:
        nilauth_client: The nilauth client
        keypair: The Nilchain keypair
        private_key: The NilAuth private key of the user
        grpc_endpoint: The endpoint of the grpc server
    """

    if get_unil_balance(wallet.address(), grpc_endpoint=grpc_endpoint) < 0:
        raise RuntimeError("User does not have enough UNIL to pay for the subscription")

    payer = Payer(
        wallet_private_key=keypair,
        chain_id="nillion-chain-devnet",
        grpc_endpoint=grpc_endpoint,
        gas_limit=1000000000000,
    )

    # Pretty print the subscription details
    subscription_details = nilauth_client.subscription_status(private_key)
    logger.info(f"IS SUBSCRIBED: {subscription_details.subscribed}")
    if not subscription_details or subscription_details.subscribed is None:
        raise RuntimeError(
            f"User subscription details could not be retrieved: {subscription_details}, {subscription_details.subscribed}, {subscription_details.details}"
        )

    if not subscription_details.subscribed:
        logger.info("[>] Paying for subscription")
        nilauth_client.pay_subscription(
            key=private_key,
            payer=payer,
        )
    else:
        logger.info("[>] Subscription is already paid for")

        if subscription_details.details is None:
            raise RuntimeError(
                f"Subscription details could not be retrieved: {subscription_details}"
            )

        logger.info(
            f"EXPIRES IN: {subscription_details.details.expires_at - datetime.datetime.now(datetime.timezone.utc)}"
        )
        logger.info(
            f"CAN BE RENEWED IN: {subscription_details.details.renewable_at - datetime.datetime.now(datetime.timezone.utc)}"
        )


def get_delegation_token(
    root_token: RootToken,
    private_key: NilAuthPrivateKey,
    user_public_key: NilAuthPublicKey,
) -> DelegationToken:
    """
    Delegate the root token to the delegated key

    Args:
        user_public_key_b64: The base64 encoded public key of the user
        nilauth_url: The URL of the nilauth server
        grpc_endpoint: The endpoint of the grpc server
    Returns:
        The delegation token
    """

    root_token_envelope = NucTokenEnvelope.parse(root_token.token)
    delegated_token = (
        NucTokenBuilder.extending(root_token_envelope)
        .audience(Did(user_public_key.serialize()))
        .command(Command(["nil", "ai", "generate"]))
        .build(private_key)
    )
    return DelegationToken(token=delegated_token)


def get_nilai_public_key(nilai_url: str) -> NilAuthPublicKey:
    """
    Get the nilai public key from the nilai server

    Args:
        nilai_url: The URL of the nilai server

    Returns:
        The nilai public key
    """
    response = httpx.get(f"{nilai_url}/v1/public_key")
    public_key = NilAuthPublicKey(base64.b64decode(response.text), raw=True)
    logger.info(f"Nilai public key: {public_key.serialize().hex()}")
    return public_key


def get_invocation_token(
    delegation_token: RootToken | DelegationToken,
    nilai_public_key: NilAuthPublicKey,
    delegated_key: NilAuthPrivateKey,
) -> InvocationToken:
    """
    Make an invocation token for the given delegated token and nilai public key

    Args:
        delegated_token: The delegated token
        nilai_public_key: The nilai public key
        delegated_key: The private key
    """
    print("Delegation token: ", delegation_token)
    delegated_token_envelope = NucTokenEnvelope.parse(delegation_token.token)

    invocation = (
        NucTokenBuilder.extending(delegated_token_envelope)
        .body(InvocationBody(args={}))
        .audience(Did(nilai_public_key.serialize()))
        .build(delegated_key)
    )
    return InvocationToken(token=invocation)


def get_nilauth_public_key(nilauth_url: str) -> Did:
    """
    Get the nilauth public key from the nilauth server

    Args:
        nilauth_url: The URL of the nilauth server

    Returns:
        The nilauth public key as a Did
    """
    nilauth_client = NilauthClient(nilauth_url)
    nilauth_public_key = Did(nilauth_client.about().public_key.serialize())
    return nilauth_public_key


def validate_token(
    nilauth_url: str, token: str, validation_parameters: ValidationParameters
):
    """
    Validate a token

    Args:
        token: The token to validate
        validation_parameters: The validation parameters
    """
    validator = NucTokenValidator([get_nilauth_public_key(nilauth_url)])

    validator.validate(NucTokenEnvelope.parse(token), validation_parameters)

    print("[>] Token validated")
