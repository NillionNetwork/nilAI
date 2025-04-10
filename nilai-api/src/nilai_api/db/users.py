import logging
import uuid

from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sqlalchemy
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.exc import SQLAlchemyError

from nilai_api.db import Base, Column, get_db_session
from nilai_api.config import (
    USER_RATE_LIMIT_MINUTE,
    USER_RATE_LIMIT_HOUR,
    USER_RATE_LIMIT_DAY,
)

logger = logging.getLogger(__name__)


# Enhanced User Model with additional constraints and validation
class UserModel(Base):
    __tablename__ = "users"

    userid: str = Column(String(50), primary_key=True, index=True)  # type: ignore
    name: str = Column(String(100), nullable=False)  # type: ignore
    email: str = Column(String(255), unique=True, nullable=False, index=True)  # type: ignore
    apikey: str = Column(String(50), unique=True, nullable=False, index=True)  # type: ignore
    prompt_tokens: int = Column(Integer, default=0, nullable=False)  # type: ignore
    completion_tokens: int = Column(Integer, default=0, nullable=False)  # type: ignore
    queries: int = Column(Integer, default=0, nullable=False)  # type: ignore
    signup_date: datetime = Column(
        DateTime, server_default=sqlalchemy.func.now(), nullable=False
    )  # type: ignore
    last_activity: datetime = Column(DateTime, nullable=True)  # type: ignore
    ratelimit_day: int = Column(Integer, default=1000, nullable=True)  # type: ignore
    ratelimit_hour: int = Column(Integer, default=100, nullable=True)  # type: ignore
    ratelimit_minute: int = Column(Integer, default=10, nullable=True)  # type: ignore

    def __repr__(self):
        return f"<User(userid={self.userid}, name={self.name}, email={self.email})>"


@dataclass
class UserData:
    userid: str
    name: str
    apikey: str
    input_tokens: int
    generated_tokens: int
    queries: int


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
                    user.last_activity = datetime.now()
                    await session.commit()
                    logger.info(f"Updated last activity for user {userid}")
                else:
                    logger.warning(f"User {userid} not found")
        except SQLAlchemyError as e:
            logger.error(f"Error updating last activity: {e}")

    @staticmethod
    async def insert_user(
        name: str,
        email: str,
        apikey: str | None = None,
        userid: str | None = None,
        ratelimit_day: int | None = USER_RATE_LIMIT_DAY,
        ratelimit_hour: int | None = USER_RATE_LIMIT_HOUR,
        ratelimit_minute: int | None = USER_RATE_LIMIT_MINUTE,
    ) -> UserModel:
        """
        Insert a new user into the database.

        Args:
            name (str): Name of the user
            email (str): Email of the user
            apikey (str): API key for the user
            userid (str): Unique ID for the user
            ratelimit_day (int): Daily rate limit
            ratelimit_hour (int): Hourly rate limit
            ratelimit_minute (int): Minute rate limit

        Returns:
            Dict containing userid and apikey
        """
        userid = userid if userid else UserManager.generate_user_id()
        apikey = apikey if apikey else UserManager.generate_api_key()
        ratelimit_day = ratelimit_day if ratelimit_day else USER_RATE_LIMIT_DAY
        ratelimit_hour = ratelimit_hour if ratelimit_hour else USER_RATE_LIMIT_HOUR
        ratelimit_minute = (
            ratelimit_minute if ratelimit_minute else USER_RATE_LIMIT_MINUTE
        )
        user = UserModel(
            userid=userid,
            name=name,
            email=email,
            apikey=apikey,
            ratelimit_day=ratelimit_day,
            ratelimit_hour=ratelimit_hour,
            ratelimit_minute=ratelimit_minute,
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
                return [
                    UserData(
                        userid=user.userid,
                        name=user.name,
                        apikey=user.apikey,
                        input_tokens=user.prompt_tokens,
                        generated_tokens=user.completion_tokens,
                        queries=user.queries,
                    )
                    for user in users
                ]
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


__all__ = ["UserManager", "UserData", "UserModel"]
