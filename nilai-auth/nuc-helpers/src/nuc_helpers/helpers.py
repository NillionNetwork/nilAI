import base64
import datetime
import logging
from typing import Tuple
import httpx

# Importing the types
from nuc_helpers.types import RootToken, DelegationToken, InvocationToken, ChainId

# Importing the secp256k1 library dependencies
from secp256k1 import PrivateKey as NilAuthPrivateKey, PublicKey as NilAuthPublicKey

# Importing the nuc library dependencies
from nuc.payer import Payer
from nuc.builder import NucTokenBuilder
from nuc.nilauth import NilauthClient, BlindModule
from nuc.envelope import NucTokenEnvelope
from nuc.token import Command, Did, InvocationBody
from nuc.validate import NucTokenValidator, ValidationParameters

# Importing the cosmpy library dependencies
from cosmpy.crypto.keypairs import PrivateKey as NilchainPrivateKey
from cosmpy.aerial.wallet import LocalWallet, Address
from cosmpy.aerial.client import LedgerClient, NetworkConfig

logger = logging.getLogger(__name__)


def get_wallet_and_private_key_from_mnemonic(
    mnemonic: str,
) -> Tuple[LocalWallet, NilchainPrivateKey, NilAuthPrivateKey]:
    """
    Get the wallet and private key from a mnemonic

    Args:
        mnemonic: The mnemonic to use for the wallet

    Returns:
        wallet: The wallet of the user to use for payments on the nilchain
        keypair: The keypair of the wallet
        private_key: The private key of the keypair to use for nilauth
    """

    wallet = LocalWallet.from_mnemonic(mnemonic, prefix="nillion")
    keypair = NilchainPrivateKey(base64.b64decode(wallet.signer().private_key))
    private_key = NilAuthPrivateKey(base64.b64decode(keypair.private_key))
    return wallet, keypair, private_key


## Helpers
def get_wallet_and_private_key(
    private_key_bytes: str | bytes | None = None,
) -> Tuple[LocalWallet, NilchainPrivateKey, NilAuthPrivateKey]:
    """
    Get the wallet and private key from a private key bytes

    Args:
        private_key_bytes: The private key bytes to use for the wallet

    Returns:
        wallet: The wallet of the user to use for payments on the nilchain
        keypair: The keypair of the wallet
        private_key: The private key of the keypair to use for nilauth
    """
    keypair = NilchainPrivateKey(private_key_bytes)
    wallet = LocalWallet(keypair, prefix="nillion")
    private_key = NilAuthPrivateKey(base64.b64decode(keypair.private_key))
    return wallet, keypair, private_key


def get_root_token(
    nilauth_client: NilauthClient,
    private_key: NilAuthPrivateKey,
    blind_module: BlindModule = BlindModule.NILAI,
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
    root_token: str = nilauth_client.request_token(
        key=private_key, blind_module=blind_module
    )

    return RootToken(token=root_token)


def get_unil_balance(
    address: Address,
    grpc_endpoint: str,
    chain_id: ChainId = ChainId.NILLION_CHAIN_DEVNET,
) -> int:
    """
    Get the UNIL balance of the user

    Args:
        address: The address of the user
        grpc_endpoint: The endpoint of the grpc server
        chain_id: The chain id of the nilchain (default is devnet)

    Returns:
        The balance of the user in UNIL
    """
    cfg = NetworkConfig(
        chain_id=chain_id.value,
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
    public_key: NilAuthPublicKey,
    grpc_endpoint: str,
    blind_module: BlindModule = BlindModule.NILAI,
    chain_id: ChainId = ChainId.NILLION_CHAIN_DEVNET,
) -> None:
    """
    Pay for the subscription using the Nilchain keypair if the user is not subscribed

    Args:
        nilauth_client: The nilauth client
        keypair: The Nilchain keypair
        private_key: The NilAuth private key of the user
        grpc_endpoint: The endpoint of the grpc server
        chain_id: The chain id of the nilchain (default is devnet)
    """

    payer = Payer(
        wallet_private_key=keypair,
        chain_id=chain_id.value,
        grpc_endpoint=grpc_endpoint,
        gas_limit=1000000000000,
    )

    # Pretty print the subscription details
    subscription_details = nilauth_client.subscription_status(public_key, blind_module)
    logger.info(f"IS SUBSCRIBED: {subscription_details.subscribed}")
    if not subscription_details or subscription_details.subscribed is None:
        raise RuntimeError(
            f"User subscription details could not be retrieved: {subscription_details}, {subscription_details.subscribed}, {subscription_details.details}"
        )

    if not subscription_details.subscribed:
        if get_unil_balance(
            wallet.address(), grpc_endpoint=grpc_endpoint
        ) < nilauth_client.subscription_cost(blind_module=blind_module):
            raise RuntimeError(
                "User does not have enough UNIL to pay for the subscription"
            )
        logger.info("[>] Paying for subscription")
        nilauth_client.pay_subscription(
            pubkey=public_key,
            payer=payer,
            blind_module=blind_module,
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
    root_token: RootToken | DelegationToken,
    private_key: NilAuthPrivateKey,
    user_public_key: NilAuthPublicKey,
    usage_limit: int | None = None,
    expires_at: datetime.datetime | None = None,
    document_id: str | None = None,
    document_owner_did: str | None = None,
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
    if bool(document_id) != bool(document_owner_did):
        raise ValueError(
            f"If Document ID or document owner DID provided, the other must also be provided: Document ID: {document_id} Document Owner DID: {document_owner_did}"
        )

    root_token_envelope = NucTokenEnvelope.parse(root_token.token)
    delegated_token = (
        NucTokenBuilder.extending(root_token_envelope)
        .expires_at(
            expires_at
            if expires_at
            else datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=5)
        )
        .audience(Did(user_public_key.serialize()))
        .command(Command(["nil", "ai", "generate"]))
        .meta(
            {
                "usage_limit": usage_limit,
                "document_id": document_id,
                "document_owner_did": document_owner_did,
            }
        )
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
    delegated_token_envelope = NucTokenEnvelope.parse(delegation_token.token)

    invocation = (
        NucTokenBuilder.extending(delegated_token_envelope)
        .body(InvocationBody(args={}))
        .audience(Did(nilai_public_key.serialize()))
        .expires_at(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=30)
        )
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
    token_envelope = NucTokenEnvelope.parse(token)
    validator = NucTokenValidator([get_nilauth_public_key(nilauth_url)])

    validator.validate(token_envelope, context={}, parameters=validation_parameters)
