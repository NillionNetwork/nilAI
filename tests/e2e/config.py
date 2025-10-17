from .nuc import get_nuc_token
from nilai_api.config import CONFIG

ENVIRONMENT = CONFIG.environment.environment
# Left for API key for backwards compatibility
AUTH_TOKEN = CONFIG.auth.auth_token
AUTH_STRATEGY = CONFIG.auth.auth_strategy

match AUTH_STRATEGY:
    case "nuc":
        BASE_URL = "https://localhost/nuc/v1"
    case "api_key":
        BASE_URL = "https://localhost/v1"
    case _:
        raise ValueError(f"Invalid AUTH_STRATEGY: {AUTH_STRATEGY}")


def api_key_getter() -> str:
    if AUTH_STRATEGY == "nuc":
        return get_nuc_token().token
    elif AUTH_STRATEGY == "api_key":
        if AUTH_TOKEN is None:
            raise ValueError("Expected AUTH_TOKEN to be set")
        return AUTH_TOKEN
    else:
        raise ValueError(f"Invalid AUTH_STRATEGY: {AUTH_STRATEGY}")


print(f"USING {AUTH_STRATEGY}")
models = {
    "mainnet": [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ],
    "testnet": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    "ci": ["openai/gpt-oss-20b"],
}

if ENVIRONMENT not in models:
    ENVIRONMENT = "ci"
    print(
        f"Environment {ENVIRONMENT} not found in models, using {ENVIRONMENT} as default"
    )
test_models = models[ENVIRONMENT]
