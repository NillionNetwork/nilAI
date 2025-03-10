import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
AUTH_TOKEN = os.environ["AUTH_TOKEN"]

if ENVIRONMENT == "dev":
    BASE_URL = "http://localhost:8080/v1"
elif ENVIRONMENT == "ci":
    BASE_URL = "http://127.0.0.1:8080/v1"
elif ENVIRONMENT == "prod":
    BASE_URL = "https://nilai-e176.nillion.network/v1"
else:
    raise ValueError(f"Invalid environment: {ENVIRONMENT}")
