import pytest
from base64 import urlsafe_b64encode
import json
from ...nilai_api.auth import (
    keplr_jwt_valid_forever,
    keplr_jwt_expired,
    metamask_jwt_valid_forever,
    metamask_jwt_expired,
    keplr_jwt_invalid_sig,
    metamask_jwt_invalid_sig,
)
from nilai_api.auth.jwt import (
    to_base64_url,
    sorted_object,
    sorted_json_string,
    escape_characters,
    serialize_sign_doc,
    extract_fields,
    keplr_validate,
    metamask_validate,
    validate_jwt,
)


def test_to_base64_url():
    data = {"key": "value"}
    result = to_base64_url(data)
    expected = urlsafe_b64encode(
        json.dumps(data, separators=(",", ":")).encode("ascii")
    ).decode("utf-8")
    assert result == expected


def test_sorted_object():
    obj = {"b": 2, "a": 1, "c": [3, 2, 1]}
    result = sorted_object(obj)
    expected = {"a": 1, "b": 2, "c": [3, 2, 1]}
    assert result == expected, result


def test_sorted_json_string():
    obj = {"b": 2, "a": 1}
    result = sorted_json_string(obj)
    expected = json.dumps({"a": 1, "b": 2}, separators=(",", ":"))
    assert result == expected


def test_escape_characters():
    input_str = "&<>"
    result = escape_characters(input_str)
    expected = "\\u0026\\u003c\\u003e"
    assert result == expected


def test_serialize_sign_doc():
    sign_doc = {"key": "value"}
    result = serialize_sign_doc(sign_doc)
    expected = escape_characters(sorted_json_string(sign_doc)).encode("utf-8")
    assert result == expected


def test_validate_keplr_valid():
    result = validate_jwt(keplr_jwt_valid_forever)
    assert result is not None, result


def test_validate_metamask_valid():
    result = validate_jwt(metamask_jwt_valid_forever)
    assert result is not None, result


def test_validate_keplr_expired():
    with pytest.raises(ValueError):
        _ = validate_jwt(keplr_jwt_expired)


def test_validate_metamask_expired():
    with pytest.raises(ValueError):
        _ = validate_jwt(metamask_jwt_expired)


def test_validate_keplr_invalid():
    with pytest.raises(ValueError):
        _ = validate_jwt(keplr_jwt_invalid_sig)


def test_validate_metamask_invalid():
    with pytest.raises(ValueError):
        _ = validate_jwt(metamask_jwt_invalid_sig)


def test_keplr_validate():
    message, header, payload, signature = extract_fields(keplr_jwt_valid_forever)
    result = keplr_validate(message, header, payload, signature)
    assert result is not None, result


def test_keplr_validate_invalid():
    message, header, payload, signature = extract_fields(keplr_jwt_invalid_sig)
    with pytest.raises(ValueError):
        keplr_validate(message, header, payload, signature)


def test_metamask_validate():
    message, header, payload, signature = extract_fields(metamask_jwt_valid_forever)
    result = metamask_validate(message, header, payload, signature)
    assert result is not None, result


def test_metamask_validate_invalid():
    message, header, payload, signature = extract_fields(metamask_jwt_invalid_sig)
    with pytest.raises(ValueError):
        metamask_validate(message, header, payload, signature)


def test_keplr_validate_expired():
    message, header, payload, signature = extract_fields(keplr_jwt_expired)
    with pytest.raises(ValueError):
        keplr_validate(message, header, payload, signature)


def test_metamask_validate_expired():
    message, header, payload, signature = extract_fields(metamask_jwt_expired)
    with pytest.raises(ValueError):
        metamask_validate(message, header, payload, signature)
