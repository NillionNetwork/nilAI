"""
Common Pydantic models for nildb_wrapper package.

This module provides base models and common types used across all modules.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, Union
from enum import Enum
from datetime import datetime

from secretvaults import SecretVaultUserClient
from secretvaults.common.keypair import Keypair


class BaseResult(BaseModel):
    """Base result model for all operations"""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    success: bool
    error: Optional[Union[str, Exception]] = None
    message: Optional[str] = None


class PromptDelegationToken(BaseModel):
    """Delegation token model"""

    model_config = ConfigDict(validate_assignment=True)

    token: str
    did: str


class TimestampedModel(BaseModel):
    """Base model with timestamp fields"""

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, populate_by_name=True
    )

    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class KeyData(TimestampedModel):
    """Model for key data in JSON files"""

    type: str
    key: str
    name: Optional[str] = None

    # For public keys
    did: Optional[str] = None
    private_key_file: Optional[str] = None

    # For private keys
    public_key_file: Optional[str] = None


class KeypairInfo(BaseModel):
    """Information about stored keypairs"""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    private_key_file: str
    public_key_file: Optional[str] = None
    created_at: Optional[str] = None
    name: str = "unnamed"
    did: str = "unknown"


# User module models
class UserSetupResult(BaseResult):
    """Result of user setup operation"""

    user_client: Optional[SecretVaultUserClient] = None
    keypair: Optional[Keypair] = None
    keys_saved_to: Optional[Dict[str, str]] = None


# Collection module models
class CollectionResult(BaseResult):
    """Result of collection operations"""

    data: Optional[Any] = None


class CollectionCreationResult(BaseResult):
    """Result of collection creation"""

    collection_id: Optional[str] = None
    collection_name: Optional[str] = None
    collection_type: Optional[str] = None


# Document module models
class OperationResult(BaseResult):
    """Result of document operations"""

    data: Optional[Any] = None


class DocumentReference(BaseModel):
    """Reference to a document"""

    model_config = ConfigDict(validate_assignment=True)

    builder: str
    collection: str
    document: str


# Builder module models
class RegistrationStatus(str, Enum):
    """Builder registration status"""

    SUCCESS = "success"
    ALREADY_REGISTERED = "already_registered"
    ERROR = "error"


class DelegationToken(BaseModel):
    """Delegation token model"""

    model_config = ConfigDict(validate_assignment=True)

    token: str
    did: str


class RegistrationResult(BaseResult):
    """Result of builder registration"""

    status: RegistrationStatus
    response: Optional[Any] = None


class TokenData(TimestampedModel):
    """Delegation token data for JSON serialization"""

    type: str = "delegation_token"
    expires_at: datetime
    user_did: str
    builder_did: str
    token: str
    usage: str = "Use this token for data creation operations"
    valid_for_seconds: int = 60
