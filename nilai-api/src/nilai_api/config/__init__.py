import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT: str = os.getenv("ENVIRONMENT", "testnet")


ETCD_HOST: str = os.getenv("ETCD_HOST", "localhost")
ETCD_PORT: int = int(os.getenv("ETCD_PORT", 2379))


REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

DOCS_TOKEN: str | None = os.getenv("DOCS_TOKEN", None)

DB_USER: str = os.getenv("POSTGRES_USER", "postgres")
DB_PASS: str = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME: str = os.getenv("POSTGRES_DB", "nilai_users")


NILAUTH_TRUSTED_ROOT_ISSUERS: List[str] = os.getenv(
    "NILAUTH_TRUSTED_ROOT_ISSUERS", ""
).split(",")

AUTH_STRATEGY: str = os.getenv("AUTH_STRATEGY", "api_key")

# Defined by default but re-defined in testnet.py and mainnet.py
USER_RATE_LIMIT_MINUTE: int | None = 100
USER_RATE_LIMIT_HOUR: int | None = 1000
USER_RATE_LIMIT_DAY: int | None = 10000
WEB_SEARCH_RATE_LIMIT_HOUR: int | None = 3

if ENVIRONMENT == "mainnet":
    from .mainnet import *  # noqa
elif ENVIRONMENT == "testnet":
    from .testnet import *  # noqa
else:
    # default to mainnet with no limits
    from .mainnet import *  # noqa
