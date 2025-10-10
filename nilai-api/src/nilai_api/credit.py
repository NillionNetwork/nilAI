import logging
from typing import Awaitable, Callable, Optional
from fastapi import Request
from typing import TypeAlias
from pydantic import BaseModel

from nilauth_credit_middleware import (
    CreditClientSingleton,
    create_metering_dependency,
    UserIdExtractors,
)

from nilai_api.config import CONFIG

logger = logging.getLogger(__name__)


class LLMCost(BaseModel):
    prompt_tokens_price: float
    completion_tokens_price: float
    web_search_cost: float

    def __init__(
        self,
        prompt_tokens_price: float,
        completion_tokens_price: float,
        web_search_cost: float,
    ):
        super().__init__(
            prompt_tokens_price=prompt_tokens_price / 1_000_000,
            completion_tokens_price=completion_tokens_price / 1_000_000,
            web_search_cost=web_search_cost,
        )

    @staticmethod
    def default() -> "LLMCost":
        return LLMCost(
            prompt_tokens_price=2.0, completion_tokens_price=2.0, web_search_cost=0.05
        )

    def total_cost(
        self, prompt_tokens: int, completion_tokens: int, web_searches: int
    ) -> float:
        logger.info(" == Cost Summary == ")
        logger.info(
            f"Prompt Tokens: {prompt_tokens} cost: {self.prompt_tokens_price * prompt_tokens}"
        )
        logger.info(
            f"Completion Tokens: {completion_tokens} cost: {self.completion_tokens_price * completion_tokens}"
        )
        logger.info(
            f"Web Searches: {web_searches} cost: {self.web_search_cost * web_searches}"
        )
        logger.info(
            f"Total Cost: {self.prompt_tokens_price * prompt_tokens + self.completion_tokens_price * completion_tokens + self.web_search_cost * web_searches}"
        )
        return (
            self.prompt_tokens_price * prompt_tokens
            + self.completion_tokens_price * completion_tokens
            + self.web_search_cost * web_searches
        )


class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    web_searches: int


class LLMResponse(BaseModel):
    usage: LLMUsage
    data: dict


LLMCostDict: TypeAlias = dict[str, LLMCost]


MyCostDictionary: LLMCostDict = {
    "meta-llama/Llama-3.2-1B-Instruct": LLMCost(
        prompt_tokens_price=3.0, completion_tokens_price=3.0, web_search_cost=0.05
    ),
    "default": LLMCost.default(),
}

# Configure the singleton credit client
CreditClientSingleton.configure(
    base_url=CONFIG.auth.credit_service_url,
    api_token=CONFIG.auth.credit_api_token,
    timeout=10.0,
)


def user_id_extractor() -> Callable[[Request], Awaitable[str]]:
    if CONFIG.auth.auth_strategy == "nuc":
        return UserIdExtractors.from_nuc_bearer_token()
    else:
        extractor = UserIdExtractors.from_header("Authorization")

        async def wrapper(request: Request) -> str:
            information = await extractor(request)
            if information.startswith("Bearer "):
                information = information[7:]
                return information
            else:
                raise ValueError("Authorization header does not start with Bearer")

        return wrapper


def llm_cost_calculator(llm_cost_dict: LLMCostDict):
    async def calculator(request: Request, response_data: dict) -> float:
        model_name = getattr(request, "model", "default")
        llm_cost = llm_cost_dict.get(model_name, LLMCost.default())
        total_cost = 0.0
        usage: Optional[LLMUsage] = response_data.get("usage", None)
        if usage is None:
            logger.error(f"Usage not found in response data: {response_data}")
            return total_cost
        total_cost += llm_cost.total_cost(
            usage.prompt_tokens, usage.completion_tokens, usage.web_searches
        )
        return total_cost

    return calculator


LLMMeter = create_metering_dependency(
    user_id_extractor=user_id_extractor(),
    estimated_cost=2.0,
    cost_calculator=llm_cost_calculator(MyCostDictionary),
)
