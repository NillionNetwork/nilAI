# Do an HTTP request to the nilai-auth-server
import httpx
from secp256k1 import PrivateKey as NilAuthPrivateKey
from nuc.validate import ValidationParameters, InvocationRequirement

import base64

from nuc.token import Did

import openai

from nuc_helpers import (
    DelegationToken,
    InvocationToken,
    get_nilai_public_key,
    get_invocation_token,
    validate_token,
)

SERVICE_ENDPOINT = "localhost:8100"
NILAI_ENDPOINT = "localhost:8080"
NILAUTH_ENDPOINT = "localhost:30921"


def retrieve_delegation_token(b64_public_key: str) -> DelegationToken:
    """
    Get a delegation token for the given public key

    Args:
        b64_public_key: The base64 encoded public key

    Returns:
        delegation_token: The delegation token
    """
    response = httpx.post(
        f"http://{SERVICE_ENDPOINT}/v1/delegate/",
        json={"user_public_key": b64_public_key},
    )
    return DelegationToken(**response.json())


def main():
    """
    Main function
    """
    # Create a user private key and public key
    user_private_key = NilAuthPrivateKey()
    user_public_key = user_private_key.pubkey

    if user_public_key is None:
        raise Exception("Failed to get public key")

    b64_public_key = base64.b64encode(user_public_key.serialize()).decode("utf-8")

    delegation_token = retrieve_delegation_token(b64_public_key)

    validate_token(
        f"http://{NILAUTH_ENDPOINT}",
        delegation_token.token,
        ValidationParameters.default(),
    )
    nilai_public_key = get_nilai_public_key(f"http://{NILAI_ENDPOINT}")
    if nilai_public_key is None:
        raise Exception("Failed to get nilai public key")

    invocation_token: InvocationToken = get_invocation_token(
        delegation_token,
        nilai_public_key,
        user_private_key,
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
    client = openai.OpenAI(
        base_url=f"http://{NILAI_ENDPOINT}/v1", api_key=invocation_token.token
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.2,
        max_tokens=100,
    )
    print(response)


if __name__ == "__main__":
    main()
