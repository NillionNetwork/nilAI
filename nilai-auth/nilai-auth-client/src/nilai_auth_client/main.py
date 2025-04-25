# Do an HTTP request to the nilai-auth-server
import httpx
from secp256k1 import PrivateKey as NilAuthPrivateKey, PublicKey as NilAuthPublicKey
from nuc.validate import NucTokenValidator, ValidationParameters, InvocationRequirement

import base64

from pydantic import BaseModel
from nuc.envelope import NucTokenEnvelope
from nuc.builder import NucTokenBuilder
from nuc.token import InvocationBody, Did
from nuc.nilauth import NilauthClient

import openai

SERVICE_ENDPOINT = "localhost:8100"
NILAI_ENDPOINT = "localhost:8080"
NILAUTH_ENDPOINT = "localhost:30921"


class DelegationToken(BaseModel):
    token: str


def get_delegation_token(b64_public_key: str):
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
    return DelegationToken(**response.json()).token


def get_nilai_public_key():
    """
    Get the nilai public key
    """
    response = httpx.get(f"http://{NILAI_ENDPOINT}/v1/public_key")
    public_key = NilAuthPublicKey(base64.b64decode(response.text), raw=True)
    print(f"Nilai public key: {public_key.serialize().hex()}")
    return public_key


def make_invocation_token(
    delegated_token: str,
    nilai_public_key: NilAuthPublicKey,
    delegated_key: NilAuthPrivateKey,
):
    """
    Make an invocation token for the given delegated token and nilai public key

    Args:
        delegated_token: The delegated token
        nilai_public_key: The nilai public key
        delegated_key: The private key
    """
    delegated_token_envelope = NucTokenEnvelope.parse(delegated_token)

    invocation = (
        NucTokenBuilder.extending(delegated_token_envelope)
        .body(InvocationBody(args={}))
        .audience(Did(nilai_public_key.serialize()))
        .build(delegated_key)
    )
    return invocation


def get_nilauth_public_key():
    """
    Get the nilauth public key
    """
    nilauth_client = NilauthClient(f"http://{NILAUTH_ENDPOINT}")
    nilauth_public_key = Did(nilauth_client.about().public_key.serialize())
    return nilauth_public_key


def validate_token(token: str, validation_parameters: ValidationParameters):
    validator = NucTokenValidator([get_nilauth_public_key()])

    validator.validate(NucTokenEnvelope.parse(token), validation_parameters)

    print("[>] Token validated")


def main():
    """
    Main function
    """
    private_key = NilAuthPrivateKey()
    public_key = private_key.pubkey

    if public_key is None:
        raise Exception("Failed to get public key")

    b64_public_key = base64.b64encode(public_key.serialize()).decode("utf-8")

    delegated_token = get_delegation_token(b64_public_key)

    validate_token(delegated_token, ValidationParameters.default())
    nilai_public_key = get_nilai_public_key()
    if nilai_public_key is None:
        raise Exception("Failed to get nilai public key")

    invocation_token = make_invocation_token(
        delegated_token, nilai_public_key, private_key
    )

    print(f"Invocation token: {invocation_token}")

    default_parameters = ValidationParameters.default()
    default_parameters.token_requirements = InvocationRequirement(
        audience=Did(nilai_public_key.serialize())
    )
    validation_parameters = default_parameters

    validate_token(invocation_token, validation_parameters)
    client = openai.OpenAI(
        base_url=f"http://{NILAI_ENDPOINT}/v1", api_key=invocation_token
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
