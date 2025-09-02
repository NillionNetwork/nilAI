import os
from .nuc import get_nuc_token

ENVIRONMENT = os.getenv("ENVIRONMENT", "ci")
# Left for API key for backwards compatibility
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
AUTH_STRATEGY = os.getenv("AUTH_STRATEGY", "nuc")

match AUTH_STRATEGY:
    case "nuc":
        BASE_URL = "https://localhost/nuc/v1"

        def api_key_getter():
            return get_nuc_token().token
    case "api_key":
        BASE_URL = "https://localhost/v1"

        def api_key_getter():
            return AUTH_TOKEN
    case _:
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
    "ci": ["google/gemma-3-4b-it"],
}

if ENVIRONMENT not in models:
    ENVIRONMENT = "ci"
    print(
        f"Environment {ENVIRONMENT} not found in models, using {ENVIRONMENT} as default"
    )
test_models = models[ENVIRONMENT]
