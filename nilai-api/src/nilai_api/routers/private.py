import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status

from nilai_api.attestation import get_attestation_report
from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.handlers.nildb.api_model import (
    PromptDelegationRequest,
    PromptDelegationToken,
)
from nilai_api.handlers.nildb.handler import get_nildb_delegation_token
from nilai_api.routers.endpoints.chat import chat_completion_router
from nilai_api.routers.endpoints.responses import responses_router
from nilai_api.state import state

from nilai_common import (
    AttestationReport,
    ModelMetadata,
    Nonce,
    Usage,
)


logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(chat_completion_router)
router.include_router(responses_router)


@router.get("/v1/delegation")
async def get_prompt_store_delegation(
    prompt_delegation_request: PromptDelegationRequest,
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> PromptDelegationToken:
    if not auth_info.user.is_subscription_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Prompt storage is reserved to subscription owners: {auth_info.user} is not a subscription owner, apikey: {auth_info.user}",
        )

    try:
        return await get_nildb_delegation_token(prompt_delegation_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server unable to produce delegation tokens: {str(e)}",
        )


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(auth_info: AuthenticationInfo = Depends(get_auth_info)) -> Usage:
    """
    Retrieve the current token usage for the authenticated user.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Usage statistics for the user's token consumption

    ### Example
    ```python
    # Retrieves token usage for the logged-in user
    usage = await get_usage(user)
    ```
    """
    return Usage(
        prompt_tokens=auth_info.user.prompt_tokens,
        completion_tokens=auth_info.user.completion_tokens,
        total_tokens=auth_info.user.prompt_tokens + auth_info.user.completion_tokens,
        queries=auth_info.user.queries,  # type: ignore # FIXME this field is not part of Usage
    )


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> AttestationReport:
    """
    Generate a cryptographic attestation report.

    - **attestation_request**: Attestation request containing a nonce
    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Attestation details for service verification

    ### Attestation Details
    - `verifying_key`: Public key for cryptographic verification
    - `cpu_attestation`: CPU environment verification
    - `gpu_attestation`: GPU environment verification

    ### Security Note
    Provides cryptographic proof of the service's integrity and environment.
    """

    attestation_report = await get_attestation_report()
    attestation_report.verifying_key = state.b64_public_key
    return attestation_report


@router.get("/v1/models", tags=["Model"])
async def get_models(
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> List[ModelMetadata]:
    """
    List all available models in the system.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Dictionary of available models

    ### Example
    ```python
    # Retrieves list of available models
    models = await get_models(user)
    ```
    """
    return [endpoint.metadata for endpoint in (await state.models).values()]
