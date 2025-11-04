import logging
import uuid
from pydantic import BaseModel, ConfigDict, Field

from typing import Optional

import sqlalchemy
from sqlalchemy import String, JSON
from sqlalchemy.exc import SQLAlchemyError

from nilai_api.db import Base, Column, get_db_session
from nilai_api.config import CONFIG

logger = logging.getLogger(__name__)


class RateLimits(BaseModel):
    """Rate limit configuration for a user."""

    # General rate limits
    user_rate_limit_day: Optional[int] = None
    user_rate_limit_hour: Optional[int] = None
    user_rate_limit_minute: Optional[int] = None

    # Web search rate limits
    web_search_rate_limit_day: Optional[int] = None
    web_search_rate_limit_hour: Optional[int] = None
    web_search_rate_limit_minute: Optional[int] = None

    # For-good rate limits
    user_rate_limit: Optional[int] = None
    web_search_rate_limit: Optional[int] = None

    def get_effective_limits(self) -> "RateLimits":
        """Return rate limits with defaults applied from config."""
        return RateLimits(
            user_rate_limit_day=self.user_rate_limit_day
            or CONFIG.rate_limiting.user_rate_limit_day,
            user_rate_limit_hour=self.user_rate_limit_hour
            or CONFIG.rate_limiting.user_rate_limit_hour,
            user_rate_limit_minute=self.user_rate_limit_minute
            or CONFIG.rate_limiting.user_rate_limit_minute,
            web_search_rate_limit_day=self.web_search_rate_limit_day
            or CONFIG.rate_limiting.web_search_rate_limit_day,
            web_search_rate_limit_hour=self.web_search_rate_limit_hour
            or CONFIG.rate_limiting.web_search_rate_limit_hour,
            web_search_rate_limit_minute=self.web_search_rate_limit_minute
            or CONFIG.rate_limiting.web_search_rate_limit_minute,
            user_rate_limit=self.user_rate_limit
            or CONFIG.rate_limiting.user_rate_limit,
            web_search_rate_limit=self.web_search_rate_limit
            or CONFIG.rate_limiting.user_rate_limit,
        )


# Enhanced User Model with additional constraints and validation
class UserModel(Base):
    __tablename__ = "users"
    user_id: str = Column(String(75), primary_key=True, index=True)  # type: ignore
    rate_limits: dict = Column(JSON, nullable=True)  # type: ignore

    def __repr__(self):
        return f"<User(user_id={self.user_id})>"

    @property
    def rate_limits_obj(self) -> RateLimits:
        """Get rate limits as a RateLimits object with defaults applied."""
        if self.rate_limits is None:
            return RateLimits().get_effective_limits()
        return RateLimits(**self.rate_limits).get_effective_limits()

    def to_pydantic(self) -> "UserData":
        return UserData.from_sqlalchemy(self)


class UserData(BaseModel):
    user_id: str  # apikey or subscription holder public key
    rate_limits: RateLimits = Field(default_factory=RateLimits().get_effective_limits)

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_sqlalchemy(cls, user: UserModel) -> "UserData":
        return cls(
            user_id=user.user_id,
            rate_limits=user.rate_limits_obj,
        )


class UserManager:
    @staticmethod
    def generate_user_id() -> str:
        """Generate a unique user ID."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_api_key() -> str:
        """Generate a unique API key."""
        return str(uuid.uuid4())

    @staticmethod
    async def insert_user(
        user_id: str | None = None,
        rate_limits: RateLimits | None = None,
    ) -> UserModel:
        """
        Insert a new user into the database.

        Args:
            name (str): Name of the user
            apikey (str): API key for the user
            user_id (str): Unique ID for the user
            rate_limits (RateLimits): Rate limit configuration

        Returns:
            UserModel: The created user model
        """
        user_id = user_id if user_id else UserManager.generate_user_id()

        user = UserModel(
            user_id=user_id,
            rate_limits=rate_limits.model_dump() if rate_limits else None,
        )
        return await UserManager.insert_user_model(user)

    @staticmethod
    async def insert_user_model(user: UserModel) -> UserModel:
        """
        Insert a new user model into the database.

        Args:
            user (UserModel): User model to insert
        """
        try:
            async with get_db_session() as session:
                session.add(user)
                await session.commit()
                logger.info(f"User {user.user_id} added successfully.")
                return user
        except SQLAlchemyError as e:
            logger.error(f"Error inserting user: {e}")
            raise

    @staticmethod
    async def check_user(user_id: str) -> Optional[UserModel]:
        """
        Validate an API key.

        Args:
            api_key (str): API key to validate

        Returns:
            User's rate limits if user id is valid, None otherwise
        """
        try:
            async with get_db_session() as session:
                query = sqlalchemy.select(UserModel).filter(
                    UserModel.user_id == user_id  # type: ignore
                )
                user = await session.execute(query)
                user = user.scalar_one_or_none()
                return user
        except SQLAlchemyError as e:
            logger.error(f"Rate limit checking user id: {e}")
            return None

    @staticmethod
    async def update_rate_limits(user_id: str, rate_limits: RateLimits) -> bool:
        """
        Update rate limits for a specific user.

        Args:
            user_id (str): User's unique ID
            rate_limits (RateLimits): New rate limit configuration

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, user_id)
                if user:
                    user.rate_limits = rate_limits.model_dump()
                    await session.commit()
                    logger.info(f"Updated rate limits for user {user_id}")
                    return True
                else:
                    logger.warning(f"User {user_id} not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating rate limits: {e}")
            return False


__all__ = ["UserManager", "UserData", "UserModel"]
