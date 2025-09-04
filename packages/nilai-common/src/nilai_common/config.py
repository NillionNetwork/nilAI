import os
from pydantic import BaseModel, Field


class HostSettings(BaseModel):
    host: str = "localhost"
    port: int = 8000
    etcd_host: str = "localhost"
    etcd_port: int = 2379
    tool_support: bool = False
    multimodal_support: bool = False
    gunicorn_workers: int = 10
    attestation_host: str = "localhost"
    attestation_port: int = 8081


class ModelSettings(BaseModel):
    num_retries: int = Field(default=30, ge=-1)
    timeout: int = Field(default=10, ge=1)


def to_bool(value: str) -> bool:
    """Convert a string to a boolean."""
    return value.lower() in ("true", "1", "t", "y", "yes")


SETTINGS: HostSettings = HostSettings(
    host=str(os.getenv("SVC_HOST", "localhost")),
    port=int(os.getenv("SVC_PORT", 8000)),
    etcd_host=str(os.getenv("ETCD_HOST", "localhost")),
    etcd_port=int(os.getenv("ETCD_PORT", 2379)),
    tool_support=to_bool(os.getenv("TOOL_SUPPORT", "False")),
    multimodal_support=to_bool(os.getenv("MULTIMODAL_SUPPORT", "False")),
    gunicorn_workers=int(os.getenv("NILAI_GUNICORN_WORKERS", 10)),
    attestation_host=str(os.getenv("ATTESTATION_HOST", "localhost")),
    attestation_port=int(os.getenv("ATTESTATION_PORT", 8081)),
)

MODEL_SETTINGS: ModelSettings = ModelSettings(
    num_retries=int(os.getenv("MODEL_NUM_RETRIES", 30)),
    timeout=int(os.getenv("MODEL_RETRY_TIMEOUT", 10)),
)
