import os
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY is not set")
    return api_key
