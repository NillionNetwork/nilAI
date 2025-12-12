# Import all configuration models
import json
import logging
from pydantic import BaseModel
from .environment import EnvironmentConfig
from .database import DatabaseConfig, DiscoveryConfig, RedisConfig
from .auth import AuthConfig, DocsConfig
from .nildb import NilDBConfig
from .web_search import WebSearchSettings
from .rate_limiting import RateLimitingConfig
from .utils import create_config_model, CONFIG_DATA


class NilAIConfig(BaseModel):
    """Centralized configuration container for the Nilai API."""

    environment: EnvironmentConfig = create_config_model(
        EnvironmentConfig, "", CONFIG_DATA
    )
    database: DatabaseConfig = create_config_model(
        DatabaseConfig, "database", CONFIG_DATA, "POSTGRES_"
    )
    discovery: DiscoveryConfig = create_config_model(
        DiscoveryConfig, "discovery", CONFIG_DATA, "DISCOVERY_"
    )
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
        config_dict = self.model_dump(mode="json")

        keywords = {"pass", "token", "key"}
        for key, value in list(config_dict.items()):
            if (
                isinstance(value, str)
                and any(k in key for k in keywords)
                and value is not None
            ):
                config_dict[key] = "***************"
            elif isinstance(value, dict):
                for k, v in list(value.items()):
                    if (
                        isinstance(v, str)
                        and any(kw in k for kw in keywords)
                        and v is not None
                    ):
                        value[k] = "***************"

        # Remove the password from the redis url anything between : and @

        if "discovery" in config_dict:
            if "url" in config_dict["discovery"]:
                value = config_dict["discovery"]["url"]
                # Remove the password from the redis url anything between : and @ if it exists
                split_value = value.split(":")
                if len(split_value) > 2 and "@" in split_value[2]:
                    password = split_value[2].split("@")[0]
                    config_dict["discovery"]["url"] = value.replace(
                        password, "***************"
                    )

        if "redis" in config_dict:
            if "url" in config_dict["redis"]:
                value = config_dict["redis"]["url"]
                # Remove the password from the redis url anything between : and @ if it exists
                split_value = value.split(":")
                if len(split_value) > 2 and "@" in split_value[2]:
                    password = split_value[2].split("@")[0]
                    config_dict["redis"]["url"] = value.replace(
                        password, "***************"
                    )

        return json.dumps(config_dict, indent=4)


# Global config instance
CONFIG = NilAIConfig()
__all__ = [
    # Main config object
    "CONFIG"
]

logging.info(CONFIG.prettify())
print(CONFIG.prettify())
