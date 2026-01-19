"""E2E tests for pricing API endpoints."""

import pytest
from .config import BASE_URL, api_key_getter
from nilai_api.config import CONFIG
import httpx


@pytest.fixture
def http_client():
    """Create an HTTPX client with user authentication."""
    invocation_token: str = api_key_getter()
    # Use base URL without /v1 since pricing endpoint is at /v1/pricing
    base = BASE_URL.rsplit("/v1", 1)[0]
    return httpx.Client(
        base_url=base,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token}",
        },
        verify=False,
        timeout=30.0,
    )


@pytest.fixture
def admin_http_client():
    """Create an HTTPX client with admin authentication."""
    admin_token = CONFIG.auth.admin_token
    if not admin_token:
        pytest.skip("Admin token not configured")

    base = BASE_URL.rsplit("/v1", 1)[0]
    return httpx.Client(
        base_url=base,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {admin_token}",
        },
        verify=False,
        timeout=30.0,
    )


class TestPricingGet:
    """Tests for GET pricing endpoints."""

    def test_get_all_prices(self, http_client):
        """Test getting all model prices."""
        response = http_client.get("/v1/pricing")
        assert response.status_code == 200

        prices = response.json()
        assert isinstance(prices, dict)
        assert "default" in prices

        # Check default pricing structure
        default = prices["default"]
        assert "prompt_tokens_price" in default
        assert "completion_tokens_price" in default
        assert "web_search_cost" in default

    def test_get_specific_model_price(self, http_client):
        """Test getting price for a specific model."""
        response = http_client.get("/v1/pricing/meta-llama/Llama-3.2-1B-Instruct")
        assert response.status_code == 200

        price = response.json()
        assert "prompt_tokens_price" in price
        assert "completion_tokens_price" in price
        assert "web_search_cost" in price

    def test_get_unknown_model_returns_default(self, http_client):
        """Test that unknown model returns default pricing."""
        response = http_client.get("/v1/pricing/unknown/model-that-does-not-exist")
        assert response.status_code == 200

        price = response.json()
        # Should return default pricing
        assert price["prompt_tokens_price"] == 2.0
        assert price["completion_tokens_price"] == 2.0
        assert price["web_search_cost"] == 0.05

    def test_get_default_price(self, http_client):
        """Test getting the default price directly."""
        response = http_client.get("/v1/pricing/default")
        assert response.status_code == 200

        price = response.json()
        assert price["prompt_tokens_price"] == 2.0
        assert price["completion_tokens_price"] == 2.0


class TestPricingUpdateDelete:
    """Tests for PUT/DELETE pricing endpoints (admin only)."""

    def test_update_price_without_admin_token_fails(self, http_client):
        """Test that updating price without admin token fails."""
        response = http_client.put(
            "/v1/pricing/test-model",
            json={
                "prompt_tokens_price": 10.0,
                "completion_tokens_price": 10.0,
                "web_search_cost": 0.1,
            },
        )
        assert response.status_code == 403

    def test_delete_price_without_admin_token_fails(self, http_client):
        """Test that deleting price without admin token fails."""
        response = http_client.delete("/v1/pricing/test-model")
        assert response.status_code == 403

    @pytest.mark.skipif(
        not CONFIG.auth.admin_token, reason="Admin token not configured"
    )
    def test_update_price_with_admin_token(self, admin_http_client, http_client):
        """Test updating a model price with admin token."""
        model_name = "e2e-test-model"
        new_price = {
            "prompt_tokens_price": 25.0,
            "completion_tokens_price": 30.0,
            "web_search_cost": 0.5,
        }

        # Update the price
        response = admin_http_client.put(f"/v1/pricing/{model_name}", json=new_price)
        assert response.status_code == 200

        result = response.json()
        assert result["prompt_tokens_price"] == 25.0
        assert result["completion_tokens_price"] == 30.0
        assert result["web_search_cost"] == 0.5

        # Verify with a regular GET request
        get_response = http_client.get(f"/v1/pricing/{model_name}")
        assert get_response.status_code == 200

        fetched = get_response.json()
        assert fetched["prompt_tokens_price"] == 25.0
        assert fetched["completion_tokens_price"] == 30.0

        # Clean up
        admin_http_client.delete(f"/v1/pricing/{model_name}")

    @pytest.mark.skipif(
        not CONFIG.auth.admin_token, reason="Admin token not configured"
    )
    def test_delete_price_with_admin_token(self, admin_http_client, http_client):
        """Test deleting a custom price with admin token."""
        model_name = "e2e-delete-test-model"

        # First create a custom price
        new_price = {
            "prompt_tokens_price": 50.0,
            "completion_tokens_price": 50.0,
            "web_search_cost": 1.0,
        }
        admin_http_client.put(f"/v1/pricing/{model_name}", json=new_price)

        # Delete it
        response = admin_http_client.delete(f"/v1/pricing/{model_name}")
        assert response.status_code == 204

        # Verify it now returns default pricing
        get_response = http_client.get(f"/v1/pricing/{model_name}")
        fetched = get_response.json()
        assert fetched["prompt_tokens_price"] == 2.0  # Default

    @pytest.mark.skipif(
        not CONFIG.auth.admin_token, reason="Admin token not configured"
    )
    def test_delete_nonexistent_price_returns_404(self, admin_http_client):
        """Test that deleting a non-existent price returns 404."""
        response = admin_http_client.delete("/v1/pricing/nonexistent-model-xyz-12345")
        assert response.status_code == 404

    @pytest.mark.skipif(
        not CONFIG.auth.admin_token, reason="Admin token not configured"
    )
    def test_delete_default_fails(self, admin_http_client):
        """Test that deleting default pricing fails."""
        response = admin_http_client.delete("/v1/pricing/default")
        assert response.status_code == 400

    @pytest.mark.skipif(
        not CONFIG.auth.admin_token, reason="Admin token not configured"
    )
    def test_update_price_with_invalid_values(self, admin_http_client):
        """Test that updating price with negative values fails."""
        response = admin_http_client.put(
            "/v1/pricing/test-model",
            json={
                "prompt_tokens_price": -1.0,
                "completion_tokens_price": 10.0,
                "web_search_cost": 0.1,
            },
        )
        assert response.status_code == 400


class TestPricingAuth:
    """Tests for pricing authentication requirements."""

    def test_get_prices_requires_auth(self):
        """Test that getting prices requires authentication."""
        base = BASE_URL.rsplit("/v1", 1)[0]
        client = httpx.Client(base_url=base, verify=False, timeout=30.0)

        response = client.get("/v1/pricing")
        # Should return 401 or 403 without auth
        assert response.status_code in [401, 403, 422]

    def test_get_model_price_requires_auth(self):
        """Test that getting a model price requires authentication."""
        base = BASE_URL.rsplit("/v1", 1)[0]
        client = httpx.Client(base_url=base, verify=False, timeout=30.0)

        response = client.get("/v1/pricing/default")
        assert response.status_code in [401, 403, 422]
