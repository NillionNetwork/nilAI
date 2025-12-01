import json
import os
import openai
from typing_extensions import override
from typing import List, Optional


import base64
import httpx
import asyncio
import datetime


from nuc.envelope import NucTokenEnvelope
from nuc.token import Did, InvocationBody
from nuc.builder import NucTokenBuilder
from nilai_py.nildb import NilDBPromptManager

from nilai_py.niltypes import (
    DelegationTokenRequest,
    DelegationTokenResponse,
    NilAuthPrivateKey,
    NilAuthPublicKey,
    AuthType,
)

from nilai_py.common import is_expired, new_root_token


class Client(openai.Client):
    def __init__(self, *args, **kwargs):
        self.auth_type: AuthType = kwargs.pop("auth_type", AuthType.API_KEY)

        match self.auth_type:
            case AuthType.API_KEY:
                self._api_key_init(*args, **kwargs)
            case AuthType.DELEGATION_TOKEN:
                self._delegation_token_init(*args, **kwargs)
                kwargs["api_key"] = (
                    "<placeholder>"  # This is a placeholder to avoid the api key being used in the super call
                )

        # Remove the nilauth_url from the kwargs
        super().__init__(*args, **kwargs)

        # Retrieve the public key from the nilai server
        try:
            self.nilai_public_key = self._get_nilai_public_key()
            print(
                "Retrieved nilai public key:", self.nilai_public_key.serialize().hex()
            )
        except Exception as e:
            print(f"Failed to retrieve the nilai public key: {e}")
            raise e

    def _api_key_init(self, *args, **kwargs):
        # Initialize the nilauth private key with the subscription
        self.api_key: str | None = kwargs.get("api_key", None)  # pyright: ignore[reportIncompatibleVariableOverride]
        if self.api_key is None:
            raise ValueError("In API key mode, api_key is required")

        self.nilauth_private_key: NilAuthPrivateKey = NilAuthPrivateKey(  # pyright: ignore[reportRedeclaration]
            bytes.fromhex(self.api_key)
        )
        # Initialize the root token envelope
        self._root_token_envelope: Optional[NucTokenEnvelope] = None

    def _delegation_token_init(self, *args, **kwargs):
        # Generate a new private key for the client
        api_key = kwargs.get("api_key", None)
        if api_key is not None:
            self.nilauth_private_key = NilAuthPrivateKey(bytes.fromhex(api_key))
        else:
            self.nilauth_private_key: NilAuthPrivateKey = NilAuthPrivateKey()

    @property
    def root_token(self) -> NucTokenEnvelope:
        """
        Get the root token envelope. If the root token is expired, it will be refreshed.
        The root token is used to create delegation tokens.

        Returns:
            NucTokenEnvelope: The root token envelope.
        """
        if self.auth_type != AuthType.API_KEY:
            raise RuntimeError("Root token is only available in API key mode")

        if self._root_token_envelope is None or is_expired(self._root_token_envelope):
            self._root_token_envelope = new_root_token(self.nilauth_private_key)

        return self._root_token_envelope

    def _get_nilai_public_key(self) -> NilAuthPublicKey:
        """
        Retrieve the nilai public key from the nilai server.

        Returns:
            NilAuthPublicKey: The nilai public key.

        Raises:
            RuntimeError: If the nilai public key cannot be retrieved.
        """
        try:
            public_key_response = httpx.get(f"{self.base_url}public_key", verify=False)
            if public_key_response.status_code != 200:
                raise RuntimeError(
                    f"Failed to retrieve the nilai public key: {public_key_response.text}"
                )
            return NilAuthPublicKey(
                base64.b64decode(public_key_response.text), raw=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve the nilai public key: {e}")

    def get_delegation_request(self) -> DelegationTokenRequest:
        """
        Get the delegation request for the client.

        Returns:
            DelegationTokenRequest: The delegation request.
        """
        if self.nilauth_private_key.pubkey is None:
            raise ValueError("Public key is None")

        delegation_request: DelegationTokenRequest = DelegationTokenRequest(
            public_key=self.nilauth_private_key.pubkey.serialize().hex()
        )
        return delegation_request

    def update_delegation(self, delegation_token_response: DelegationTokenResponse):
        """
        Update the delegation token for the client.
        """
        self.delegation_token = NucTokenEnvelope.parse(
            delegation_token_response.delegation_token
        )

    def _get_invocation_token(self) -> str:
        """
        Get the invocation token for the client.

        Returns:
            str: The invocation token.
        """
        match self.auth_type:
            case AuthType.API_KEY:
                return self._get_invocation_token_with_api_key()
            case AuthType.DELEGATION_TOKEN:
                return self._get_invocation_token_with_delegation()
            case _:
                raise RuntimeError("Invalid auth type")

    def _get_invocation_token_with_delegation(self) -> str:
        """
        Get the invocation token for the client with delegation.
        """
        if self.auth_type != AuthType.DELEGATION_TOKEN:
            raise RuntimeError(
                "Invocation token is only available through API key mode only"
            )

        invocation_token: str = (
            NucTokenBuilder.extending(self.delegation_token)
            .body(InvocationBody(args={}))
            .audience(Did(self.nilai_public_key.serialize()))
            .build(self.nilauth_private_key)
        )
        return invocation_token

    def _get_invocation_token_with_api_key(self) -> str:
        """
        Get the invocation token for the client with API key.
        """
        if self.auth_type != AuthType.API_KEY:
            raise RuntimeError(
                "Invocation token is only available through Delegation Token mode only"
            )

        invocation_token: str = (
            NucTokenBuilder.extending(self.root_token)
            .body(InvocationBody(args={}))
            .audience(Did(self.nilai_public_key.serialize()))
            .build(self.nilauth_private_key)
        )
        return invocation_token

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self._get_invocation_token()
        return {"Authorization": f"Bearer {api_key}"}

    async def async_list_prompts_from_nildb(self) -> None:
        prompt_manager = await NilDBPromptManager.init(nilai_url=str(self.base_url))
        await prompt_manager.list_prompts()
        await prompt_manager.close()

    def list_prompts_from_nildb(self) -> None:
        return asyncio.run(self.async_list_prompts_from_nildb())

    async def async_store_prompt_to_nildb(self, prompt: str, dir: str) -> List[str]:
        prompt_manager = await NilDBPromptManager.init(nilai_url=str(self.base_url))

        invocation_token = self._get_invocation_token()
        result = await prompt_manager.create_prompt(
            prompt=prompt, nilai_invocation_token=invocation_token
        )

        await prompt_manager.close()

        # Extract document IDs from the result for storage
        document_ids = []
        if result and hasattr(result, "root"):
            for node_name, response in result.root.items():  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(response, "data") and hasattr(response.data, "created"):
                    document_ids.extend(response.data.created)

        # Store the created document IDs to a json file
        did = prompt_manager.user_result.keypair
        if did is None:
            raise ValueError("DID is None")
        did = did.to_did_string()

        os.makedirs(dir, exist_ok=True)
        storage_data = {
            "prompt": prompt,
            "created_at": datetime.datetime.now().isoformat(),
            "did": did,
            "document_ids": document_ids,
        }
        with open(f"{dir}/stored_prompts-{document_ids[0]}.json", "w+") as f:
            json.dump(storage_data, f, indent=4)

        return document_ids

    def store_prompt_to_nildb(self, prompt: str, dir="./stored_prompts") -> List[str]:
        return asyncio.run(self.async_store_prompt_to_nildb(prompt, dir=dir))
