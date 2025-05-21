from nuc_helpers import (
    get_wallet_and_private_key,
    pay_for_subscription,
    get_root_token,
    get_nilai_public_key,
    get_invocation_token,
    validate_token,
    InvocationToken,
    RootToken,
    NilAuthPublicKey,
)
from nuc.nilauth import NilauthClient
from nuc.token import Did
from nuc.validate import ValidationParameters, InvocationRequirement


def get_nuc_token() -> InvocationToken:
    # Services must be running for this to work
    PRIVATE_KEY = "l/SYifzu2Iqc3dsWoWHRP2oSMHwrORY/PDw5fDwtJDQ="  # This is an example private key with funds for testing devnet, and should not be used in production
    NILAI_ENDPOINT = "localhost:8080"
    NILAUTH_ENDPOINT = "localhost:30921"
    NILCHAIN_GRPC = "localhost:26649"

    # Server private key
    server_wallet, server_keypair, server_private_key = get_wallet_and_private_key(
        PRIVATE_KEY
    )
    nilauth_client = NilauthClient(f"http://{NILAUTH_ENDPOINT}")

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

    nilai_public_key: NilAuthPublicKey = get_nilai_public_key(
        f"http://{NILAI_ENDPOINT}"
    )
    invocation_token: InvocationToken = get_invocation_token(
        root_token,
        nilai_public_key,
        server_private_key,
    )

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
