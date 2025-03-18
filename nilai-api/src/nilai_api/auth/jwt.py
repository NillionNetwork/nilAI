import json
import ecdsa
from pydantic import BaseModel
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256

import time
from web3 import Web3
from hexbytes import HexBytes

from eth_account.messages import encode_defunct


class JWTAuthResult(BaseModel):
    pub_key: str
    user_address: str


def to_base64_url(data: object) -> str:
    # Convert the object to a JSON string
    json_string = json.dumps(data, separators=(",", ":"))

    # Encode the string to bytes using ASCII and convert to base64
    base64_bytes = urlsafe_b64encode(json_string.encode("ascii"))

    # Return the base64-encoded string
    return base64_bytes.decode("utf-8")


def sorted_object(obj: dict | list) -> dict | list:
    if not isinstance(obj, (dict, list)):
        return obj
    if isinstance(obj, list):
        return [sorted_object(item) for item in obj]

    sorted_keys = sorted(obj.keys())
    result = {}
    for key in sorted_keys:
        result[key] = sorted_object(obj[key])

    return result


def sorted_json_string(obj: dict | list) -> str:
    return json.dumps(sorted_object(obj), separators=(",", ":"))


def escape_characters(input: str) -> str:
    amp = "&"
    lt = "<"
    gt = ">"
    return input.replace(amp, "\\u0026").replace(lt, "\\u003c").replace(gt, "\\u003e")


def serialize_sign_doc(sign_doc: dict) -> bytes:
    serialized = escape_characters(sorted_json_string(sign_doc))
    return serialized.encode("utf-8")


def keplr_validate(
    message: str, header: dict, payload: dict, signature: bytes
) -> JWTAuthResult:
    # Validate the algorithm
    if header["alg"] != "ES256":
        raise ValueError("Unsupported algorithm")

    # Check expiration
    if payload.get("exp") and payload["exp"] < int(time.time()):
        raise ValueError("Token has expired")

    signature_payload = to_base64_url({"message": message})

    sign_doc = {
        "chain_id": "",
        "account_number": "0",
        "sequence": "0",
        "fee": {"gas": "0", "amount": []},
        "msgs": [
            {
                "type": "sign/MsgSignData",
                "value": {"signer": payload["user_address"], "data": signature_payload},
            }
        ],
        "memo": "",
    }

    serialized_sign_doc = serialize_sign_doc(sign_doc)

    public_key = ecdsa.VerifyingKey.from_string(
        bytes.fromhex(payload["pub_key"]), curve=ecdsa.SECP256k1, hashfunc=sha256
    )

    public_key.verify(
        signature,
        serialized_sign_doc,
    )

    pub_key = payload.get("pub_key")
    user_address = payload.get("user_address")
    if not pub_key or not user_address:
        raise ValueError("Invalid payload, missing pub_key or user_address")
    return JWTAuthResult(pub_key=pub_key, user_address=user_address)


def metamask_validate(
    message: str, header: dict, payload: dict, signature: bytes
) -> JWTAuthResult:
    # Validate the algorithm
    if header["alg"] != "ES256K":
        raise ValueError("Unsupported algorithm")
    # Check expiration
    if payload.get("exp") and payload["exp"] < int(time.time()):
        raise ValueError("Token has expired")
    w3 = Web3(Web3.HTTPProvider(""))
    signable_message = encode_defunct(text=message)
    address = w3.eth.account.recover_message(
        signable_message, signature=HexBytes("0x" + signature.hex())
    )

    if address.lower() != payload.get("user_address"):
        raise ValueError("Invalid signature")

    pub_key = payload.get("pub_key")
    user_address = payload.get("user_address")
    if not pub_key or not user_address:
        raise ValueError("Invalid payload, missing pub_key or user_address")

    return JWTAuthResult(pub_key=pub_key, user_address=user_address)


def extract_fields(jwt: str) -> tuple[str, dict, dict, bytes]:
    # Split and decode JWT components
    header_b64, payload_b64, signature_b64 = jwt.split(".")
    if not all([header_b64, payload_b64, signature_b64]):
        raise ValueError("Invalid JWT format")

    # header = json.loads(urlsafe_b64decode(header_b64 + '=' * (-len(header_b64) % 4)))
    # payload = json.loads(urlsafe_b64decode(payload_b64 + '=' * (-len(payload_b64) % 4)))
    # signature = urlsafe_b64decode(signature_b64 + '=' * (-len(signature_b64) % 4))
    header = json.loads(urlsafe_b64decode(header_b64))
    payload = json.loads(urlsafe_b64decode(payload_b64))
    signature = urlsafe_b64decode(signature_b64)

    return f"{header_b64}.{payload_b64}", header, payload, signature


def validate_jwt(jwt: str) -> JWTAuthResult:
    message, header, payload, signature = extract_fields(jwt)

    match header.get("wallet"):
        case "Keplr":
            return keplr_validate(message, header, payload, signature)
        case "Metamask":
            return metamask_validate(message, header, payload, signature)
        case _:
            raise ValueError("Unsupported wallet")
