import os

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


ETCD_HOST = os.getenv("ETCD_HOST", "localhost")
ETCD_PORT = int(os.getenv("ETCD_PORT", 2379))


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "nilai_users")


AUTH_STRATEGY = "api_key"


if ENVIRONMENT == "mainnet":
    from .mainnet import *  # noqa
elif ENVIRONMENT == "testnet":
    from .testnet import *  # noqa
