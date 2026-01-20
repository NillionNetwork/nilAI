from typing import Dict
from pydantic import BaseModel, Field


class LLMPriceConfig(BaseModel):
    """Pricing configuration for a single LLM model."""

    prompt_tokens_price: float = Field(
        default=2.0, description="Cost per 1M prompt tokens"
    )
    completion_tokens_price: float = Field(
        default=2.0, description="Cost per 1M completion tokens"
    )
    web_search_cost: float = Field(default=0.05, description="Cost per web search")


class LLMPricingConfig(BaseModel):
    """Container for all LLM pricing configurations."""

    default: LLMPriceConfig = Field(default_factory=LLMPriceConfig)
    models: Dict[str, LLMPriceConfig] = Field(default_factory=dict)
