from pydantic import BaseModel
from enum import StrEnum


## Pydantic models
class TokenType(StrEnum):
    ROOT = "root"
    DELEGATION = "delegation"
    INVOCATION = "invocation"
    PRIVATE_KEY = "private_key"


class ChainId(StrEnum):
    NILLION_CHAIN_MAINNET = "nillion-1"
    NILLION_CHAIN_TESTNET = "nillion-chain-testnet-1"
    NILLION_CHAIN_DEVNET = "nillion-chain-devnet"


class PrivateKey(BaseModel):
    type: TokenType = TokenType.PRIVATE_KEY
    token: str


class RootToken(BaseModel):
    type: TokenType = TokenType.ROOT
    token: str


class DelegationToken(BaseModel):
    type: TokenType = TokenType.DELEGATION
    token: str


class InvocationToken(BaseModel):
    type: TokenType = TokenType.INVOCATION
    token: str
