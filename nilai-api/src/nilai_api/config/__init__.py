# Import all configuration models
import json
from .environment import EnvironmentConfig
from .database import DatabaseConfig, EtcdConfig, RedisConfig
from .auth import AuthConfig, DocsConfig
from .nildb import NilDBConfig
from .web_search import WebSearchSettings
from .rate_limiting import RateLimitingConfig
from .utils import create_config_model, CONFIG_DATA
from pydantic import BaseModel
import logging


class NilAIConfig(BaseModel):
    """Centralized configuration container for the Nilai API."""

    environment: EnvironmentConfig = create_config_model(
        EnvironmentConfig, "", CONFIG_DATA
    )
    database: DatabaseConfig = create_config_model(
        DatabaseConfig, "database", CONFIG_DATA, "POSTGRES_"
    )
    etcd: EtcdConfig = create_config_model(EtcdConfig, "etcd", CONFIG_DATA, "ETCD_")
    redis: RedisConfig = create_config_model(
        RedisConfig, "redis", CONFIG_DATA, "REDIS_"
    )
    auth: AuthConfig = create_config_model(AuthConfig, "auth", CONFIG_DATA)
    docs: DocsConfig = create_config_model(DocsConfig, "docs", CONFIG_DATA, "DOCS_")
    web_search: WebSearchSettings = create_config_model(
        WebSearchSettings, "web_search", CONFIG_DATA, "WEB_SEARCH_"
    )
    rate_limiting: RateLimitingConfig = create_config_model(
        RateLimitingConfig, "rate_limiting", CONFIG_DATA
    )
    nildb: NilDBConfig = create_config_model(
        NilDBConfig, "nildb", CONFIG_DATA, "NILDB_"
    )

    def prettify(self):
        """Print the config in a pretty format removing passwords and other sensitive information"""
        config_dict = self.model_dump()
        keywords = ["pass", "token", "key"]
        for key, value in config_dict.items():
            if isinstance(value, str):
                for keyword in keywords:
                    print(key, keyword, keyword in key)
                    if keyword in key and value is not None:
                        config_dict[key] = "***************"
            if isinstance(value, dict):
                for k, v in value.items():
                    for keyword in keywords:
                        if keyword in k and v is not None:
                            value[k] = "***************"
        return json.dumps(config_dict, indent=4)


# Global config instance
CONFIG = NilAIConfig()
__all__ = [
    # Main config object
    "CONFIG"
]

logging.info(CONFIG.prettify())
