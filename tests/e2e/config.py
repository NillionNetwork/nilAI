import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
# Left for API key for backwards compatibility
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

BASE_URL = "http://127.0.0.1:8080/v1"


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

test_models = models["ci"]
