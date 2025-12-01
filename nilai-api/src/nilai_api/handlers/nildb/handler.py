from typing import Optional
from nilai_api.config import CONFIG

from secretvaults import SecretVaultBuilderClient, SecretVaultUserClient
from secretvaults.common.keypair import Keypair
from secretvaults.common.blindfold import BlindfoldFactoryConfig, BlindfoldOperation

from secretvaults.common.utils import into_seconds_from_now
from nuc.builder import NucTokenBuilder
from nuc.token import Command, Did
from secretvaults.common.nuc_cmd import NucCmd
from secretvaults.common.types import Uuid
from secretvaults.dto.users import (
    ReadDataRequestParams,
)

from nilai_api.auth.common import PromptDocument
from nilai_api.handlers.nildb.api_model import PromptDelegationToken

import datetime


BUILDER_CLIENT: Optional[SecretVaultBuilderClient] = None
USER_CLIENT: Optional[SecretVaultUserClient] = None


async def create_builder_client():
    """Create and return a builder client using proper initialization pattern"""
    global BUILDER_CLIENT
    if BUILDER_CLIENT is not None:
        return BUILDER_CLIENT

    # Create keypair from private key
    keypair = Keypair.from_hex(CONFIG.nildb.builder_private_key)

    # Prepare URLs for the builder client
    urls = {
        "chain": [CONFIG.nildb.nilchain_url],
        "auth": CONFIG.nildb.nilauth_url,
        "dbs": CONFIG.nildb.nodes,
    }

    # Create SecretVaultBuilderClient with proper initialization
    BUILDER_CLIENT = await SecretVaultBuilderClient.from_options(
        keypair=keypair,
        urls=urls,
        blindfold=BlindfoldFactoryConfig(
            operation=BlindfoldOperation.STORE, use_cluster_key=True
        ),
    )

    # Get root token for use in other functions
    await BUILDER_CLIENT.refresh_root_token()

    return BUILDER_CLIENT


async def create_user_client() -> SecretVaultUserClient:
    """Create and return a user client using proper initialization pattern"""
    global USER_CLIENT
    if USER_CLIENT is not None:
        return USER_CLIENT

    # Create keypair from private key
    keypair = Keypair.from_hex(CONFIG.nildb.builder_private_key)
    USER_CLIENT = await SecretVaultUserClient.from_options(
        keypair=keypair,
        base_urls=CONFIG.nildb.nodes,
        blindfold=BlindfoldFactoryConfig(
            operation=BlindfoldOperation.STORE, use_cluster_key=True
        ),
    )

    return USER_CLIENT


async def get_nildb_delegation_token(user_did: str) -> PromptDelegationToken:
    """Get a delegation token for the builder - core functionality without UI concerns"""
    # Get builder's root token
    builder_client = await create_builder_client()
    root_token_envelope = builder_client.root_token

    if not root_token_envelope:
        raise ValueError("Couldn't extract root NUC token from nilDB profile")

    # Create delegation token extending the root token envelope
    delegation_token = (
        NucTokenBuilder.extending(root_token_envelope)
        .command(Command(NucCmd.NIL_DB_DATA_CREATE.value.split(".")))
        .audience(Did.parse(user_did))
        .expires_at(datetime.datetime.fromtimestamp(into_seconds_from_now(60)))
        .build(builder_client.keypair.private_key())
    )

    builder_did = builder_client.keypair.to_did_string()
    return PromptDelegationToken(token=delegation_token, did=builder_did)


async def get_prompt_from_nildb(prompt_document: PromptDocument) -> str:
    """Read a specific document - core functionality"""

    read_params = ReadDataRequestParams(
        collection=CONFIG.nildb.collection,
        document=Uuid(prompt_document.document_id),
        subject=Uuid(prompt_document.owner_did),
    )
    user_client = await create_user_client()
    document_response = await user_client.read_data(read_params)

    if not document_response:
        raise ValueError("Couldn't get document response from nilDB nodes")

    # Check if response has data attribute (wrapped response)
    if hasattr(document_response, "data") and document_response.data:
        document_data = document_response.data
    else:
        document_data = document_response

    # Convert to dict to avoid pyright attribute errors based on flexible typing of output dictionary
    if hasattr(document_data, "__dict__"):
        data_dict = document_data.__dict__
    elif hasattr(document_data, "model_dump"):
        data_dict = document_data.model_dump()
    else:
        data_dict = dict(document_data) if document_data else {}
    if data_dict.get("owner", None) != str(prompt_document.owner_did):
        raise ValueError(
            "Non-owning entity trying to invoke access to a document resource"
        )

    if "prompt" not in data_dict:
        raise ValueError("Couldn't find prompt field in document response from nilDB")

    prompt = data_dict.get("prompt")
    if prompt is None:
        raise ValueError("Prompt field is None in document response from nilDB")
    return prompt
