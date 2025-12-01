from typing import List, Optional
import httpx

from secretvaults import SecretVaultUserClient
import uuid

from nilai_py.nildb.models import (
    DocumentReference,
    UserSetupResult,
    PromptDelegationToken,
)
from nilai_py.nildb.config import NilDBConfig, DefaultNilDBConfig
from nilai_py.nildb.user import create_user_if_not_exists
from nilai_py.nildb.document import (
    create_document_core,
    list_data_references_core,
)


class NilDBPromptManager(object):
    """Manager for handling document prompts in NilDB"""

    def __init__(self, nilai_url: str, nildb_config: NilDBConfig = DefaultNilDBConfig):
        self.nilai_url = nilai_url
        self.nildb_config = nildb_config
        self._client: Optional[SecretVaultUserClient] = None
        self._user_result: Optional[UserSetupResult] = None

    @staticmethod
    async def init(
        nilai_url: str, nildb_config: NilDBConfig = DefaultNilDBConfig
    ) -> "NilDBPromptManager":
        """Async initializer to setup user and client"""
        instance = NilDBPromptManager(nilai_url, nildb_config)
        instance._user_result = await instance.setup_user()
        instance._client = instance.user_result.user_client
        return instance

    @property
    def client(self) -> SecretVaultUserClient:
        if not self._client:
            raise RuntimeError("Client not initialized. Call setup_user() first.")
        return self._client

    @property
    def user_result(self) -> UserSetupResult:
        if not self._user_result:
            raise RuntimeError("User not initialized. Call setup_user() first.")
        return self._user_result

    async def setup_user(self, keys_dir: str = "keys") -> UserSetupResult:
        """Setup user keypair and client with configuration validation and error handling"""
        result = await create_user_if_not_exists(
            config=self.nildb_config, keys_dir=keys_dir
        )

        if not result.success:
            raise RuntimeError(f"User setup failed: {result.error}")
        else:
            if result.keypair is not None:
                print(
                    f"🎉 User setup successful! 🎉\n  🔑 Keys saved to: {result.keys_saved_to}\n  🔐 Public Key: {result.keypair.public_key_hex(compressed=True)}\n  🆔 DID: {result.keypair.to_did_string()}"
                )
        return result

    async def request_nildb_delegation_token(self, token=None) -> PromptDelegationToken:
        # Use provided token, or fall back to env variable, or use default

        if self.user_result.keypair is None:
            raise RuntimeError("User keypair is not initialized")

        prompt_delegation_token = httpx.get(
            f"{self.nilai_url}delegation",
            params={
                "prompt_delegation_request": self.user_result.keypair.to_did_string()
            },
            verify=False,
            headers={"Authorization": f"Bearer {token}"},
        )

        print(
            f"Delegation token response status: {prompt_delegation_token.status_code}"
        )

        if prompt_delegation_token.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve the delegation token: {prompt_delegation_token.text}"
            )

        return PromptDelegationToken(**prompt_delegation_token.json())

    async def list_prompts(self) -> None:
        """List all document references for the user"""
        try:
            result = await list_data_references_core(user_client=self.client)

            print(
                "\n=== List Document References ==="
                "\nListing all document references owned by the user:"
                "\n" + "=" * 60
            )
            if result.success and result.data:
                print("Document References:")
                for ref in result.data:
                    print(f" - Collection: {ref.collection}, Document: {ref.document}")
            else:
                print("No document references found.")
        except Exception as e:
            print(f"An error occurred while listing document references: {str(e)}")

    async def create_prompt(
        self, prompt: str, nilai_invocation_token: str
    ) -> List[DocumentReference]:
        """Store a new document prompt with the given content based on the document ID"""
        print(
            f"\n=== Create Document on {self.nildb_config.collection} for prompt: {prompt} ==="
        )

        try:
            print(
                f"\n📝 Creating document in collection {self.nildb_config.collection}"
            )
            print("=" * 60)

            # Load delegation token from file
            print("🔑 Loading delegation token...")
            delegation_token = await self.request_nildb_delegation_token(
                token=nilai_invocation_token
            )

            # Fixed sample document data

            id = str(uuid.uuid4())
            data = {"_id": id, "prompt": {"%allot": prompt}}
            print(f"📝 Using document data: {data}")

            result = await create_document_core(
                self.client,
                self.nildb_config.collection,
                data,
                delegation_token.token,
                delegation_token.did,
            )
            if result.success:
                print(f"✅ Document created successfully! Document ID: {id}")
            else:
                print(f"❌ Failed to create document: {result.error or result.message}")
            return result.data if result.success and result.data else []
        except Exception as e:
            print(f"An error occurred while creating the document: {str(e)}")
            return []

    async def close(self):
        """Close the underlying client connection"""
        if self._client:
            await self._client.close()
            self._client = None
