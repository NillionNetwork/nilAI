from openai import DefaultHttpxClient
from nilai_py import (
    Client,
    DelegationTokenServer,
    AuthType,
    DelegationServerConfig,
    DelegationTokenRequest,
    DelegationTokenResponse,
)

from config import API_KEY


def main():
    # >>> Server initializes a delegation token server
    # The server is responsible for creating delegation tokens
    # and managing their expiration and usage.
    http_client = DefaultHttpxClient(verify=False)

    server = DelegationTokenServer(
        private_key=API_KEY,
        config=DelegationServerConfig(
            expiration_time=10,  # 10 seconds validity of delegation tokens
            token_max_uses=1,  # 1 use of a delegation token
        ),
        # For production instances, use the following:
        # nilauth_instance=NilAuthInstance.PRODUCTION,
    )

    # >>> Client initializes a client
    # The client is responsible for making requests to the Nilai API.
    # We do not provide an API key but we set the auth type to DELEGATION_TOKEN
    client = Client(
        base_url="https://localhost/nuc/v1",
        auth_type=AuthType.DELEGATION_TOKEN,
        http_client=http_client,
        # For production instances, use the following:
        # nilauth_instance=NilAuthInstance.PRODUCTION,
    )
    for i in range(100):
        # >>> Client produces a delegation request
        delegation_request: DelegationTokenRequest = client.get_delegation_request()

        # <<< Server creates a delegation token
        delegation_token: DelegationTokenResponse = server.create_delegation_token(
            delegation_request
        )

        # >>> Client sets internally the delegation token
        client.update_delegation(delegation_token)

        # >>> Client uses the delegation token to make a request
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Hello! Can you help me with something?"}
            ],
        )

        print(f"Response {i}: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
