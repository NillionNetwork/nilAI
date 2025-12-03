import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_api_key() -> str:
    api_key: str | None = os.getenv("API_KEY", None)
    if api_key is None:
        raise ValueError("API_KEY is not set")
    return api_key


API_KEY: str = get_api_key()
