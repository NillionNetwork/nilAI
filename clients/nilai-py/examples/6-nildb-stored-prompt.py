"""
Example 6: Using stored prompts from NilDB with delegation token flow

This example demonstrates how to:
1. Load private keys and stored prompt data from files
2. Set up a delegation token chain between subscription owner and prompt data owner
3. Use a client with delegation tokens to access stored prompts

Key components:
- Subscription owner server (creates delegation tokens for API access)
- Prompt data owner server (manages access to stored prompt documents)
- Client (makes requests using delegation tokens)
"""

import json
from typing import Dict, Any

from nilai_py import (
    Client,
    DelegationTokenServer,
    AuthType,
    DelegationServerConfig,
    PromptDocumentInfo,
    DelegationTokenServerType,
)

from config import API_KEY


class FileLoader:
    """Utility class for loading configuration files."""

    @staticmethod
    def load_private_key(filename: str) -> str:
        """Load private key from JSON file."""
        with open(filename, "r") as f:
            key_data = json.load(f)
        return key_data["key"]

    @staticmethod
    def load_stored_prompt_data(filename: str) -> Dict[str, str]:
        """Load stored prompt data including DID and document IDs."""
        with open(filename, "r") as f:
            prompt_data = json.load(f)
        return {
            "did": prompt_data["did"],
            "doc_id": prompt_data["document_ids"][0],  # Use the first document ID
        }


class DelegationServerManager:
    """Manages delegation token servers for the stored prompt flow."""

    def __init__(
        self,
        api_key: str,
    ):
        self.api_key = api_key

    def create_subscription_owner_server(self) -> DelegationTokenServer:
        """Create server for the subscription owner (manages API access)."""
        return DelegationTokenServer(
            private_key=self.api_key,
            config=DelegationServerConfig(
                expiration_time=10 * 60 * 60,  # 10 hours
                token_max_uses=10,
            ),
        )

    def create_prompt_data_owner_server(
        self, private_key: str, prompt_data: Dict[str, str]
    ) -> DelegationTokenServer:
        """Create server for the prompt data owner (manages document access)."""
        return DelegationTokenServer(
            private_key=private_key,
            config=DelegationServerConfig(
                mode=DelegationTokenServerType.DELEGATION_ISSUER,
                expiration_time=10,  # 10 seconds
                token_max_uses=1,
                prompt_document=PromptDocumentInfo(
                    doc_id=prompt_data["doc_id"], owner_did=prompt_data["did"]
                ),
            ),
        )


class StoredPromptClient:
    """Client for making requests using stored prompts with delegation tokens."""

    def __init__(
        self,
        base_url: str = "https://nilai-f910.nillion.network/nuc/v1/",
    ):
        self.client = Client(
            base_url=base_url,
            auth_type=AuthType.DELEGATION_TOKEN,
        )

    def setup_delegation(self, delegation_server: DelegationTokenServer) -> None:
        """Set up delegation token for the client."""
        delegation_request = self.client.get_delegation_request()
        delegation_token = delegation_server.create_delegation_token(delegation_request)
        self.client.update_delegation(delegation_token)

    def create_completion(self, model: str, messages: list) -> Any:
        """Create a chat completion using the configured client."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
        )


def setup_delegation_chain(
    subscription_server: DelegationTokenServer, prompt_server: DelegationTokenServer
) -> None:
    """Set up the delegation chain between servers."""
    prompt_request = prompt_server.get_delegation_request()
    delegation_token = subscription_server.create_delegation_token(prompt_request)
    prompt_server.update_delegation_token(delegation_token.delegation_token)


def main():
    """Main execution flow for stored prompt example."""

    # Load configuration files
    loader = FileLoader()
    private_key = loader.load_private_key("keys/private_key_20250922_165315.json")
    stored_prompt_data = loader.load_stored_prompt_data(
        "stored_prompts/stored_prompts-9bb6bb19-54a8-4992-a85a-faac3ea98637.json"
    )

    # Initialize server manager
    server_manager = DelegationServerManager(API_KEY)

    # Create delegation servers
    subscription_owner_server = server_manager.create_subscription_owner_server()
    prompt_data_owner_server = server_manager.create_prompt_data_owner_server(
        private_key, stored_prompt_data
    )

    # Set up delegation chain
    setup_delegation_chain(subscription_owner_server, prompt_data_owner_server)

    # Initialize client and set up delegation
    stored_prompt_client = StoredPromptClient()
    stored_prompt_client.setup_delegation(prompt_data_owner_server)

    # Make request using stored prompt
    response = stored_prompt_client.create_completion(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": "Hello! Can you help me with something?"}
        ],
    )

    print(
        "Your response, if using the previous stored prompt should have a cheese answer:"
    )
    print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
