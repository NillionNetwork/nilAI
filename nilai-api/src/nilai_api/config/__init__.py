import os

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "testnet")


ETCD_HOST = os.getenv("ETCD_HOST", "localhost")
ETCD_PORT = int(os.getenv("ETCD_PORT", 2379))

GPUVERIFIER_API = os.getenv("GPUVERIFIER_API", "http://localhost:8000")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "nilai_users")


AUTH_STRATEGY = "api_key"


if ENVIRONMENT == "mainnet":
    from .mainnet import *  # noqa
elif ENVIRONMENT == "testnet":
    from .testnet import *  # noqa
