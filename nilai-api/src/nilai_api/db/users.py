import logging
import uuid
from pydantic import BaseModel, ConfigDict, Field

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import sqlalchemy
from sqlalchemy import Integer, String, DateTime, JSON
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

    userid: str = Column(String(75), primary_key=True, index=True)  # type: ignore
    name: str = Column(String(100), nullable=False)  # type: ignore
    apikey: str = Column(String(75), unique=False, nullable=False, index=True)  # type: ignore
    prompt_tokens: int = Column(Integer, default=0, nullable=False)  # type: ignore
    completion_tokens: int = Column(Integer, default=0, nullable=False)  # type: ignore
    queries: int = Column(Integer, default=0, nullable=False)  # type: ignore
    signup_date: datetime = Column(
        DateTime(timezone=True), server_default=sqlalchemy.func.now(), nullable=False
    )  # type: ignore
    last_activity: datetime = Column(DateTime(timezone=True), nullable=True)  # type: ignore
    rate_limits: dict = Column(JSON, nullable=True)  # type: ignore

    def __repr__(self):
        return f"<User(userid={self.userid}, name={self.name})>"

    @property
    def rate_limits_obj(self) -> RateLimits:
        """Get rate limits as a RateLimits object with defaults applied."""
        if self.rate_limits is None:
            return RateLimits().get_effective_limits()
        return RateLimits(**self.rate_limits).get_effective_limits()

    def to_pydantic(self) -> "UserData":
        return UserData.from_sqlalchemy(self)


class UserData(BaseModel):
    userid: str
    name: str
    apikey: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queries: int = 0
    signup_date: datetime
    last_activity: Optional[datetime] = None
    rate_limits: RateLimits = Field(default_factory=RateLimits().get_effective_limits)

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_sqlalchemy(cls, user: UserModel) -> "UserData":
        return cls(
            userid=user.userid,
            name=user.name,
            apikey=user.apikey,
            prompt_tokens=user.prompt_tokens or 0,
            completion_tokens=user.completion_tokens or 0,
            queries=user.queries or 0,
            signup_date=user.signup_date or datetime.now(timezone.utc),
            last_activity=user.last_activity,
            rate_limits=user.rate_limits_obj,
        )

    @property
    def is_subscription_owner(self):
        return self.userid == self.apikey


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
    async def update_last_activity(userid: str):
        """
        Update the last activity timestamp for a user.

        Args:
            userid (str): User's unique ID
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, userid)
                if user:
                    user.last_activity = datetime.now(timezone.utc)
                    await session.commit()
                    logger.info(f"Updated last activity for user {userid}")
                else:
                    logger.warning(f"User {userid} not found")
        except SQLAlchemyError as e:
            logger.error(f"Error updating last activity: {e}")

    @staticmethod
    async def insert_user(
        name: str,
        apikey: str | None = None,
        userid: str | None = None,
        rate_limits: RateLimits | None = None,
    ) -> UserModel:
        """
        Insert a new user into the database.

        Args:
            name (str): Name of the user
            apikey (str): API key for the user
            userid (str): Unique ID for the user
            rate_limits (RateLimits): Rate limit configuration

        Returns:
            UserModel: The created user model
        """
        userid = userid if userid else UserManager.generate_user_id()
        apikey = apikey if apikey else UserManager.generate_api_key()

        user = UserModel(
            userid=userid,
            name=name,
            apikey=apikey,
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
                logger.info(f"User {user.name} added successfully.")
                return user
        except SQLAlchemyError as e:
            logger.error(f"Error inserting user: {e}")
            raise

    @staticmethod
    async def check_user(userid: str) -> Optional[UserModel]:
        """
        Validate a user.

        Args:
            userid (str): User ID to validate

        Returns:
            User's name if user is valid, None otherwise
        """
        try:
            async with get_db_session() as session:
                query = sqlalchemy.select(UserModel).filter(UserModel.userid == userid)  # type: ignore
                user = await session.execute(query)
                user = user.scalar_one_or_none()
                return user
        except SQLAlchemyError as e:
            logger.error(f"Error checking API key: {e}")
            return None

    @staticmethod
    async def check_api_key(api_key: str) -> Optional[UserModel]:
        """
        Validate an API key.

        Args:
            api_key (str): API key to validate

        Returns:
            User's name if API key is valid, None otherwise
        """
        try:
            async with get_db_session() as session:
                query = sqlalchemy.select(UserModel).filter(UserModel.apikey == api_key)  # type: ignore
                user = await session.execute(query)
                user = user.scalar_one_or_none()
                return user
        except SQLAlchemyError as e:
            logger.error(f"Error checking API key: {e}")
            return None

    @staticmethod
    async def update_token_usage(
        userid: str, prompt_tokens: int, completion_tokens: int
    ):
        """
        Update token usage for a specific user.

        Args:
            userid (str): User's unique ID
            prompt_tokens (int): Number of input tokens
            completion_tokens (int): Number of generated tokens
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, userid)
                if user:
                    user.prompt_tokens += prompt_tokens
                    user.completion_tokens += completion_tokens
                    user.queries += 1
                    await session.commit()
                    logger.info(f"Updated token usage for user {userid}")
                else:
                    logger.warning(f"User {userid} not found")
        except SQLAlchemyError as e:
            logger.error(f"Error updating token usage: {e}")

    @staticmethod
    async def get_token_usage(userid: str) -> Optional[Dict[str, Any]]:
        """
        Get token usage for a specific user.

        Args:
            userid (str): User's unique ID
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, userid)
                if user:
                    return {
                        "prompt_tokens": user.prompt_tokens,
                        "completion_tokens": user.completion_tokens,
                        "total_tokens": user.prompt_tokens + user.completion_tokens,
                        "queries": user.queries,
                    }
                else:
                    logger.warning(f"User {userid} not found")
                    return None
        except SQLAlchemyError as e:
            logger.error(f"Error updating token usage: {e}")
            return None

    @staticmethod
    async def get_all_users() -> Optional[List[UserData]]:
        """
        Retrieve all users from the database.

        Returns:
            List of UserData or None if no users found
        """
        try:
            async with get_db_session() as session:
                users = await session.execute(sqlalchemy.select(UserModel))
                users = users.scalars().all()
                return [UserData.from_sqlalchemy(user) for user in users]
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving all users: {e}")
            return None

    @staticmethod
    async def get_user_token_usage(userid: str) -> Optional[Dict[str, int]]:
        """
        Retrieve total token usage for a user.

        Args:
            userid (str): User's unique ID

        Returns:
            Dict of token usage or None if user not found
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, userid)
                if user:
                    return {
                        "prompt_tokens": user.prompt_tokens,
                        "completion_tokens": user.completion_tokens,
                        "queries": user.queries,
                    }
                return None
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving token usage: {e}")
            return None

    @staticmethod
    async def update_rate_limits(userid: str, rate_limits: RateLimits) -> bool:
        """
        Update rate limits for a specific user.

        Args:
            userid (str): User's unique ID
            rate_limits (RateLimits): New rate limit configuration

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            async with get_db_session() as session:
                user = await session.get(UserModel, userid)
                if user:
                    user.rate_limits = rate_limits.model_dump()
                    await session.commit()
                    logger.info(f"Updated rate limits for user {userid}")
                    return True
                else:
                    logger.warning(f"User {userid} not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating rate limits: {e}")
            return False


__all__ = ["UserManager", "UserData", "UserModel"]
