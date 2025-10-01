from typing import Tuple, Literal, Dict, Type
from fastapi import HTTPException, Request
from pydantic import BaseModel
from nilai_api.config import CONFIG
from nilai_common import ChatRequest, ResponseRequest


SCHEMAS: Dict[str, Type[BaseModel]] = {
    "chat": ChatRequest,
    "response": ResponseRequest,
}


async def _model_concurrent_rate_limit(
    request: Request, schema: Type[BaseModel], key_prefix: str
) -> Tuple[int, str]:
    body = await request.json()
    try:
        req_model = schema(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    model = getattr(req_model, "model")
    key = f"{key_prefix}:{model}"
    limit = CONFIG.rate_limiting.model_concurrent_rate_limit.get(
        model,
        CONFIG.rate_limiting.model_concurrent_rate_limit.get("default", 50),
    )
    return limit, key


async def _model_web_search_rate_limit(
    request: Request, schema: Type[BaseModel]
) -> bool:
    body = await request.json()
    try:
        req_model = schema(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    return getattr(req_model, "web_search", False)


def concurrent_rate_limit(kind: Literal["chat", "response"]):
    async def dep(request: Request) -> Tuple[int, str]:
        schema = SCHEMAS[kind]
        return await _model_concurrent_rate_limit(request, schema, key_prefix=kind)
    return dep


def web_search_rate_limit(kind: Literal["chat", "response"]):
    async def dep(request: Request) -> bool:
        schema = SCHEMAS[kind]
        return await _model_web_search_rate_limit(request, schema)
    return dep

