import os
from typing import List, Dict, Any, Optional
import yaml
from dotenv import load_dotenv

load_dotenv()

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

# Default values
USER_RATE_LIMIT_MINUTE: Optional[int] = 100
USER_RATE_LIMIT_HOUR: Optional[int] = 1000
USER_RATE_LIMIT_DAY: Optional[int] = 10000
MODEL_CONCURRENT_RATE_LIMIT: Dict[str, int] = {}


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


config_file: str = "config.yaml"
config_path = os.path.join(os.path.dirname(__file__), config_file)

if not os.path.exists(config_path):
    config_file = "config.yaml"
    config_path = os.path.join(os.path.dirname(__file__), config_file)

config_data = load_config_from_yaml(config_path)

# Overwrite with values from yaml
if config_data:
    USER_RATE_LIMIT_MINUTE = config_data.get(
        "user_rate_limit_minute", USER_RATE_LIMIT_MINUTE
    )
    USER_RATE_LIMIT_HOUR = config_data.get("user_rate_limit_hour", USER_RATE_LIMIT_HOUR)
    USER_RATE_LIMIT_DAY = config_data.get("user_rate_limit_day", USER_RATE_LIMIT_DAY)
    MODEL_CONCURRENT_RATE_LIMIT = config_data.get(
        "model_concurrent_rate_limit", MODEL_CONCURRENT_RATE_LIMIT
    )
