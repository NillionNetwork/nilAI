import os
from pydantic import BaseModel


class HostSettings(BaseModel):
    host: str = "localhost"
    port: int = 8000
    etcd_host: str = "localhost"
    etcd_port: int = 2379
    tool_support: bool = False
    gunicorn_workers: int = 10
    attestation_host: str = "localhost"
    attestation_port: int = 8081


SETTINGS: HostSettings = HostSettings(
    host=str(os.getenv("SVC_HOST", "localhost")),
    port=int(os.getenv("SVC_PORT", 8000)),
    etcd_host=str(os.getenv("ETCD_HOST", "localhost")),
    etcd_port=int(os.getenv("ETCD_PORT", 2379)),
    tool_support=bool(os.getenv("TOOL_SUPPORT", False)),
    gunicorn_workers=int(os.getenv("NILAI_GUNICORN_WORKERS", 10)),
    attestation_host=str(os.getenv("ATTESTATION_HOST", "localhost")),
    attestation_port=int(os.getenv("ATTESTATION_PORT", 8081)),
)
