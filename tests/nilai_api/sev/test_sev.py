import base64
import ctypes
import pytest
from nilai_api.sev.sev import SEVGuest


@pytest.fixture
def sev_guest():
    return SEVGuest()


def test_init_success(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.Init.return_value = 0
    assert sev_guest.init() is True


def test_init_failure(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.Init.return_value = -1
    assert sev_guest.init() is False


def test_get_quote_success(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.GetQuote.return_value = ctypes.create_string_buffer(b"quote_data")
    report_data = bytes([0] * 64)
    quote = sev_guest.get_quote(report_data)
    expected_quote = base64.b64encode(b"quote_data").decode("ascii")
    assert quote == expected_quote


def test_get_quote_failure(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.GetQuote.return_value = None
    report_data = bytes([0] * 64)
    with pytest.raises(RuntimeError):
        sev_guest.get_quote(report_data)


def test_get_quote_invalid_report_data(sev_guest):
    with pytest.raises(ValueError):
        sev_guest.get_quote(bytes([0] * 63))


def test_verify_quote_success(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.VerifyQuote.return_value = 0
    quote = base64.b64encode(b"quote_data").decode("ascii")
    assert sev_guest.verify_quote(quote) is True


def test_verify_quote_failure(sev_guest, mocker):
    mocker.patch.object(sev_guest, "_load_library", return_value=None)
    sev_guest.lib = mocker.Mock()
    sev_guest.lib.VerifyQuote.return_value = -1
    quote = base64.b64encode(b"quote_data").decode("ascii")
    assert sev_guest.verify_quote(quote) is False
