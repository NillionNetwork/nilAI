from nuc_helpers import (
    get_wallet_and_private_key,
    pay_for_subscription,
    get_root_token,
    get_delegation_token,
    get_nilai_public_key,
    get_invocation_token,
    validate_token,
    InvocationToken,
    RootToken,
    DelegationToken,
    NilAuthPublicKey,
    NilAuthPrivateKey,
)
from nuc.nilauth import NilauthClient
from nuc.token import Did
from nuc.validate import ValidationParameters, InvocationRequirement


def b2b2b2c_test():
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
    if not server_private_key.pubkey:
        raise Exception("Failed to get public key")

    pay_for_subscription(
        nilauth_client,
        server_wallet,
        server_keypair,
        server_private_key.pubkey,
        f"http://{NILCHAIN_GRPC}",
    )

    # Create a root token
    root_token: RootToken = get_root_token(nilauth_client, server_private_key)

    # Create a user private key and public key
    user_private_key = NilAuthPrivateKey()
    user_public_key = user_private_key.pubkey

    if user_public_key is None:
        raise Exception("Failed to get public key")
    # b64_public_key = base64.b64encode(public_key.serialize()).decode("utf-8")

    delegation_token: DelegationToken = get_delegation_token(
        root_token,
        server_private_key,
        user_public_key,
    )

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        delegation_token.token,
        ValidationParameters.default(),
    )
    for i in range(2):
        delegation_token: DelegationToken = get_delegation_token(
            delegation_token,
            user_private_key,
            user_public_key,
        )
        validate_token(
            f"http://{NILAUTH_ENDPOINT}",
            delegation_token.token,
            ValidationParameters.default(),
        )
        print("[>] Validated delegation token: ", type(delegation_token))

    nilai_public_key: NilAuthPublicKey = get_nilai_public_key(
        f"http://{NILAI_ENDPOINT}"
    )

    invocation_token: InvocationToken = get_invocation_token(
        delegation_token,
        nilai_public_key,
        user_private_key,
    )

    print("Root token type: ", type(root_token))
    default_validation_parameters = ValidationParameters.default()
    default_validation_parameters.token_requirements = InvocationRequirement(
        audience=Did(nilai_public_key.serialize())
    )

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        invocation_token.token,
        default_validation_parameters,
    )


def b2b2c_test():
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
    if not server_private_key.pubkey:
        raise Exception("Failed to get public key")

    pay_for_subscription(
        nilauth_client,
        server_wallet,
        server_keypair,
        server_private_key.pubkey,
        f"http://{NILCHAIN_GRPC}",
    )

    # Create a root token
    root_token: RootToken = get_root_token(nilauth_client, server_private_key)

    # Create a user private key and public key
    user_private_key = NilAuthPrivateKey()
    user_public_key = user_private_key.pubkey

    if user_public_key is None:
        raise Exception("Failed to get public key")
    # b64_public_key = base64.b64encode(public_key.serialize()).decode("utf-8")

    delegation_token: DelegationToken = get_delegation_token(
        root_token,
        server_private_key,
        user_public_key,
    )

    print("Delegation token: ", delegation_token, type(delegation_token))
    nilai_public_key: NilAuthPublicKey = get_nilai_public_key(
        f"http://{NILAI_ENDPOINT}"
    )
    invocation_token: InvocationToken = get_invocation_token(
        delegation_token,
        nilai_public_key,
        user_private_key,
    )

    print("Root token type: ", type(root_token))
    default_validation_parameters = ValidationParameters.default()
    default_validation_parameters.token_requirements = InvocationRequirement(
        audience=Did(nilai_public_key.serialize())
    )

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        invocation_token.token,
        default_validation_parameters,
    )


def b2c_test():
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
    if not server_private_key.pubkey:
        raise Exception("Failed to get public key")

    pay_for_subscription(
        nilauth_client,
        server_wallet,
        server_keypair,
        server_private_key.pubkey,
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

    print("Root token type: ", type(root_token))
    default_validation_parameters = ValidationParameters.default()
    default_validation_parameters.token_requirements = InvocationRequirement(
        audience=Did(nilai_public_key.serialize())
    )

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        invocation_token.token,
        default_validation_parameters,
    )


def main():
    """
    Main function to test the helpers
    """
    b2b2b2c_test()
    # b2b2c_test()
    # b2c_test()


if __name__ == "__main__":
    main()
