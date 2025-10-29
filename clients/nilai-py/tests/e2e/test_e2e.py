import pytest
import openai
from nilai_py import (
    Client,
    DelegationTokenServer,
    AuthType,
    DelegationServerConfig,
    DelegationTokenRequest,
    DelegationTokenResponse,
)

from . import get_api_key


def test_e2e_api_key():
    client = Client(
        base_url="https://nilai-a779.nillion.network/nuc/v1/",
        api_key=get_api_key(),
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "user", "content": "Hello! Can you help me with something?"}
        ],
    )

    print(f"Response: {response.choices[0].message.content}")


def test_e2e_delegation_token():
    server = DelegationTokenServer(
        private_key=get_api_key(),
        config=DelegationServerConfig(
            expiration_time=10,  # 1 second
            token_max_uses=1,  # 1 use
        ),
    )

    client = Client(
        base_url="https://nilai-a779.nillion.network/nuc/v1/",
        auth_type=AuthType.DELEGATION_TOKEN,
    )

    # Client produces a delegation request
    delegation_request: DelegationTokenRequest = client.get_delegation_request()
    # Server creates a delegation token
    delegation_token: DelegationTokenResponse = server.create_delegation_token(
        delegation_request
    )
    # Client updates the delegation token
    client.update_delegation(delegation_token)
    # Client uses the delegation token to make a request
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "user", "content": "Hello! Can you help me with something?"}
        ],
    )

    print(f"Response: {response.choices[0].message.content}")


def test_e2e_delegation_token_expired():
    server = DelegationTokenServer(
        private_key=get_api_key(),
        config=DelegationServerConfig(
            expiration_time=0,  # 0 seconds validity -> token is expired
            token_max_uses=1,  # 1 use
        ),
    )

    client = Client(
        base_url="https://nilai-a779.nillion.network/nuc/v1/",
        auth_type=AuthType.DELEGATION_TOKEN,
    )

    # Client produces a delegation request
    delegation_request: DelegationTokenRequest = client.get_delegation_request()
    # Server creates a delegation token
    delegation_token: DelegationTokenResponse = server.create_delegation_token(
        delegation_request
    )
    # Client updates the delegation token
    client.update_delegation(delegation_token)

    with pytest.raises(openai.AuthenticationError):
        # Client uses the delegation token to make a request
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "user", "content": "Hello! Can you help me with something?"}
            ],
        )

        print(f"Response: {response.choices[0].message.content}")


def test_e2e_delegation_token_max_uses():
    server = DelegationTokenServer(
        private_key=get_api_key(),
        config=DelegationServerConfig(
            expiration_time=10,  # 10 seconds validity -> token is not expired
            token_max_uses=1,  # 1 use -> token can be used once
        ),
    )

    client = Client(
        base_url="https://nilai-a779.nillion.network/nuc/v1/",
        auth_type=AuthType.DELEGATION_TOKEN,
    )

    # Client produces a delegation request
    delegation_request: DelegationTokenRequest = client.get_delegation_request()
    # Server creates a delegation token
    delegation_token: DelegationTokenResponse = server.create_delegation_token(
        delegation_request
    )
    # Client updates the delegation token
    client.update_delegation(delegation_token)
    # Client uses the delegation token to make a request
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "user", "content": "Hello! Can you help me with something?"}
        ],
    )

    print(f"Response: {response.choices[0].message.content}")
    with pytest.raises(openai.RateLimitError):
        # Client uses the delegation token to make a request
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "user", "content": "Hello! Can you help me with something?"}
            ],
        )

        print(f"Response: {response.choices[0].message.content}")
