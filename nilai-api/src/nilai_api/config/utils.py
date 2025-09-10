import os
from typing import Dict, Any, Optional, Type, TypeVar, get_origin
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    """Get nested value from dict using dot notation."""
    value = data
    for key in key_path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def create_config_model(
    model_class: Type[T],
    yaml_section: str,
    config_data: Dict[str, Any],
    env_prefix: str = "",
    custom_env_mapping: Optional[Dict[str, str]] = None,
) -> T:
    """Create Pydantic model instance with YAML-first, env override approach."""
    # Get YAML section data
    yaml_data = get_nested_value(config_data, yaml_section) or {}

    # Prepare data dict with environment overrides
    model_data = {}
    custom_env_mapping = custom_env_mapping or {}

    # Get model fields
    for field_name, field_info in model_class.model_fields.items():
        # Determine environment variable key
        if field_name in custom_env_mapping:
            # Use custom mapping first
            env_keys = [custom_env_mapping[field_name]]
        else:
            # Use standard prefix logic
            env_keys = [
                f"{env_prefix}{field_name.upper()}"
                if env_prefix
                else field_name.upper()
            ]

        # Add special case for api_key -> BRAVE_SEARCH_API for backward compatibility
        if (
            field_name == "api_key"
            and "BRAVE_SEARCH_API"
            not in [custom_env_mapping.get(field_name, "")] + env_keys
        ):
            env_keys.append("BRAVE_SEARCH_API")

        # Try environment variables in order
        env_value = None
        for env_key in env_keys:
            env_value = os.getenv(env_key)
            if env_value is not None:
                break

        if env_value is not None:
            # Handle type conversion for environment variables
            field_type = field_info.annotation
            if field_type is bool:
                model_data[field_name] = env_value.lower() in ("true", "1", "yes", "on")
            elif field_type is int:
                model_data[field_name] = int(env_value)
            elif field_type is float:
                model_data[field_name] = float(env_value)
            elif get_origin(field_type) is list:
                model_data[field_name] = env_value.split(",") if env_value else []
            elif field_type is dict or get_origin(field_type) is dict:
                try:
                    model_data[field_name] = json.loads(env_value)
                except json.JSONDecodeError:
                    model_data[field_name] = {}
            else:
                model_data[field_name] = env_value
        elif field_name in yaml_data:
            # Use YAML value
            model_data[field_name] = yaml_data[field_name]
        # If neither env nor yaml has the value, let Pydantic handle defaults
    return model_class(**model_data)


def get_required_env_var(name: str) -> str:
    """Get a required environment variable, raising an error if not set."""
    value: Optional[str] = os.getenv(name, None)
    if value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


# Load shared config data
config_file: str = "config.yaml"
config_path = os.path.join(os.path.dirname(__file__), config_file)
CONFIG_DATA = load_config_from_yaml(config_path)
