import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.config import CONFIG
from nilai_api.config.pricing import LLMPriceConfig
from nilai_api.pricing_service import get_pricing_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/pricing", tags=["Pricing"])
admin_bearer_scheme = HTTPBearer()


async def verify_admin_token(
    credentials: HTTPAuthorizationCredentials = Security(admin_bearer_scheme),
) -> None:
    """Dependency to verify that the request has a valid admin token."""
    admin_token = CONFIG.auth.admin_token
    if not admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin operations are disabled (no admin token configured)",
        )

    if credentials.credentials != admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )


@router.get("", response_model=Dict[str, LLMPriceConfig])
async def get_all_prices(
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> Dict[str, LLMPriceConfig]:
    """
    Get all model prices.

    Returns a dictionary mapping model names to their pricing configurations.
    """
    pricing_service = get_pricing_service()
    return await pricing_service.get_all_prices()


@router.get("/{model_name:path}", response_model=LLMPriceConfig)
async def get_model_price(
    model_name: str,
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> LLMPriceConfig:
    """
    Get price for a specific model.

    - **model_name**: The model name (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
    - **Returns**: Pricing configuration for the model

    If no specific pricing is set for the model, returns default pricing.
    """
    pricing_service = get_pricing_service()
    return await pricing_service.get_price(model_name)


@router.put("/{model_name:path}", response_model=LLMPriceConfig)
async def update_model_price(
    model_name: str,
    price_config: LLMPriceConfig,
    _: None = Depends(verify_admin_token),
) -> LLMPriceConfig:
    """
    Update price for a specific model (admin only).

    - **model_name**: The model name (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
    - **price_config**: New pricing configuration

    Requires admin token in Authorization header.
    """

    # Validate price values
    if price_config.prompt_tokens_price < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_tokens_price must be non-negative",
        )
    if price_config.completion_tokens_price < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="completion_tokens_price must be non-negative",
        )
    if price_config.web_search_cost < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="web_search_cost must be non-negative",
        )

    pricing_service = get_pricing_service()
    await pricing_service.set_price(model_name, price_config)

    logger.info(f"Admin updated pricing for model '{model_name}'")
    return price_config


@router.delete("/{model_name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_price(
    model_name: str,
    _: None = Depends(verify_admin_token),
) -> None:
    """
    Delete custom price for a model (admin only).

    - **model_name**: The model name to delete pricing for

    After deletion, the model will use default pricing.
    Requires admin token in Authorization header.
    """

    if model_name == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default pricing",
        )

    pricing_service = get_pricing_service()
    existed = await pricing_service.delete_price(model_name)

    if not existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No custom pricing found for model '{model_name}'",
        )

    logger.info(f"Admin deleted pricing for model '{model_name}'")
