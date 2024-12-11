from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# Internal libraries
from nilai_common import ModelEndpoint
from nilai_api.state import state

router = APIRouter()


def is_local_network_request(request: Request) -> bool:
    """
    Check if the request is coming from a local Docker network.

    This implementation checks for common local network ranges:
    - 172.16.0.0/12 (Docker default bridge network)
    - 10.0.0.0/8 (Private network range)
    - 192.168.0.0/16 (Private network range)
    - 127.0.0.0/8 (Loopback addresses)

    Args:
        request: The incoming HTTP request

    Returns:
        bool: True if the request is from a local network, False otherwise
    """
    # Get client host IP
    client_host = request.client.host # type: ignore

    # Local network ranges
    local_networks = [
        # Docker's default bridge network
        ("172.16.0.0", 12),
        # Private network ranges
        ("10.0.0.0", 8),
        ("192.168.0.0", 16),
        # Loopback
        ("127.0.0.0", 8),
    ]

    # Convert IP to integer for easier comparison
    def ip_to_int(ip: str) -> int:
        try:
            return int.from_bytes(bytes(map(int, ip.split("."))), "big")
        except (ValueError, TypeError):
            return 0

    def ip_in_network(ip: str, network: str, prefix: int) -> bool:
        try:
            ip_int = ip_to_int(ip)
            network_int = ip_to_int(network)
            mask = ((1 << 32) - 1) ^ ((1 << (32 - prefix)) - 1)
            return (ip_int & mask) == (network_int & mask)
        except Exception:
            return False

    # Check if IP is in any of the local network ranges
    return any(
        ip_in_network(client_host, network, prefix)
        for network, prefix in local_networks
    )


@router.post("/internal/endpoints", include_in_schema=False)
async def add_endpoint(
    request: Request,
    endpoint: ModelEndpoint
) -> None:
    """Add a new model endpoint."""
    if not is_local_network_request(request):
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )

    await state.add_endpoint(endpoint)
    return None

@router.delete("/internal/endpoints/{model_name}", include_in_schema=False)
async def remove_endpoint(
    request: Request,
    model_name: str
) -> None:
    """Remove a model endpoint."""
    if not is_local_network_request(request):
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )

    if model_name not in state.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found"
        )

    await state.remove_endpoint(model_name)
    return None