from unittest.mock import AsyncMock, patch

import pytest
from nilai_api.state import AppState


@pytest.fixture
def app_state():
    return AppState()


def test_generate_key_pair(app_state):
    assert app_state.private_key is not None
    assert app_state.public_key is not None
    assert app_state.verifying_key is not None


def test_semaphore_initialization(app_state):
    assert app_state.sem._value == 2


def test_uptime(app_state):
    uptime = app_state.uptime
    assert (
        "days" in uptime
        or "hours" in uptime
        or "minutes" in uptime
        or "seconds" in uptime
    )


@patch("nilai_api.state.sev.init")
@patch("nilai_api.state.sev.get_quote", return_value="mocked_quote")
def test_cpu_attestation(mock_get_quote, mock_init, app_state):
    assert app_state.cpu_attestation == "mocked_quote"
    mock_init.assert_called_once()
    mock_get_quote.assert_called_once()


def test_gpu_attestation(app_state):
    assert app_state.gpu_attestation == "<No GPU>"


@pytest.mark.asyncio
async def test_models(app_state):
    with patch.object(
        app_state.discovery_service, "discover_models", new_callable=AsyncMock
    ) as mock_discover_models:
        mock_discover_models.return_value = {"model1": "endpoint1"}
        models = await app_state.models
        assert models == {"model1": "endpoint1"}
        mock_discover_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model(app_state):
    with patch.object(
        app_state.discovery_service, "get_model", new_callable=AsyncMock
    ) as mock_get_model:
        mock_get_model.return_value = "endpoint1"
        model = await app_state.get_model("model1")
        assert model == "endpoint1"
        mock_get_model.assert_awaited_once_with("model1")


@pytest.mark.asyncio
async def test_models_empty(app_state):
    with patch.object(
        app_state.discovery_service, "discover_models", new_callable=AsyncMock
    ) as mock_discover_models:
        mock_discover_models.return_value = {}
        models = await app_state.models
        assert models == {}
        mock_discover_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model_not_found(app_state):
    with patch.object(
        app_state.discovery_service, "get_model", new_callable=AsyncMock
    ) as mock_get_model:
        mock_get_model.return_value = None
        model = await app_state.get_model("non_existent_model")
        assert model is None
        mock_get_model.assert_awaited_once_with("non_existent_model")
