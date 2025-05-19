import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "ci")
# Left for API key for backwards compatibility
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

BASE_URL = "https://localhost/nuc/v1"


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
    "ci": [
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
}

if ENVIRONMENT not in models:
    ENVIRONMENT = "ci"
    print(
        f"Environment {ENVIRONMENT} not found in models, using {ENVIRONMENT} as default"
    )
test_models = models[ENVIRONMENT]
