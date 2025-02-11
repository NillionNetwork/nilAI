import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import sqlalchemy
from datetime import datetime
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from nilai_api import config

# Configure logging
logger = logging.getLogger(__name__)

DATABASE_URL = sqlalchemy.engine.url.URL.create(
    drivername="postgresql+asyncpg",  # Use asyncpg driver
    username=config.DB_USER,
    password=config.DB_PASS,
    host=config.DB_HOST,
    port=config.DB_PORT,
    database=config.DB_NAME,
)


class DatabaseConfig:
    DATABASE_URL = DATABASE_URL
    POOL_SIZE = 5
    MAX_OVERFLOW = 10
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600  # Reconnect after 1 hour


# Create base and engine with improved configuration
Base = sqlalchemy.orm.declarative_base()

_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> sqlalchemy.ext.asyncio.AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DatabaseConfig.DATABASE_URL,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=DatabaseConfig.POOL_SIZE,
            max_overflow=DatabaseConfig.MAX_OVERFLOW,
            pool_timeout=DatabaseConfig.POOL_TIMEOUT,
            pool_recycle=DatabaseConfig.POOL_RECYCLE,
            echo=False,  # Set to True for SQL logging during development
        )
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


# Enhanced User Model with additional constraints and validation
class UserModel(Base):
    __tablename__ = "users"

    userid = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    apikey = Column(String(50), unique=True, nullable=False, index=True)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    queries = Column(Integer, default=0, nullable=False)
    signup_date = Column(DateTime, default=datetime.now(), nullable=False)
    last_activity = Column(DateTime, nullable=True)
    ratelimit_day = Column(Integer, default=1000, nullable=True)
    ratelimit_hour = Column(Integer, default=100, nullable=True)
    ratelimit_minute = Column(Integer, default=10, nullable=True)

    def __repr__(self):
        return f"<User(userid={self.userid}, name={self.name}, email={self.email})>"


# New QueryLog Model for tracking individual queries
class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    userid = Column(String(50), ForeignKey("users.userid"), nullable=False, index=True)
    query_timestamp = Column(DateTime, default=datetime.now(), nullable=False)
    model = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<QueryLog(userid={self.userid}, query_timestamp={self.query_timestamp}, total_tokens={self.total_tokens})>"


@dataclass
class UserData:
    userid: str
    name: str
    apikey: str
    input_tokens: int
    generated_tokens: int
    queries: int


# Async context manager for database sessions
@asynccontextmanager
async def get_db_session() -> "Generator[AsyncSession, Any, Any]":
    """Provide a transactional scope for database operations."""
    session = get_sessionmaker()()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        await session.close()


class UserManager:
    @staticmethod
    async def initialize_db() -> bool:
        """
        Create database tables only if they do not already exist.

        Returns:
            bool: True if tables were created, False if tables already existed
        """
        try:
            async with get_engine().begin() as conn:
                # Create an inspector to check existing tables
                inspector = await conn.run_sync(
                    lambda sync_conn: sqlalchemy.inspect(sync_conn)
                )

                # Check if the 'users' table already exists
                if not await conn.run_sync(
                    lambda sync_conn: inspector.has_table("users")
                ) or not await conn.run_sync(
                    lambda sync_conn: inspector.has_table("query_logs")
                ):
                    # Create all tables that do not exist
                    await conn.run_sync(Base.metadata.create_all)
                    logger.info("Database tables created successfully.")
                    return True
                else:
                    logger.info("Database tables already exist. Skipping creation.")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error checking or creating database tables: {e}")
            raise

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
    async def insert_user(name: str, email: str) -> Dict[str, str]:
        """
        Insert a new user into the database.

        Args:
            name (str): Name of the user
            email (str): Email of the user

        Returns:
            Dict containing userid and apikey
        """
        userid = UserManager.generate_user_id()
        apikey = UserManager.generate_api_key()
        user = UserModel(userid=userid, name=name, email=email, apikey=apikey)
        UserManager.insert_user_model(user)

    @staticmethod
    async def insert_user_model(user: UserModel):
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
        except SQLAlchemyError as e:
            logger.error(f"Error inserting user: {e}")
            raise

    @staticmethod
    async def log_query(
        userid: str, model: str, prompt_tokens: int, completion_tokens: int
    ):
        """
        Log a user's query.

        Args:
            userid (str): User's unique ID
            model (str): The model that generated the response
            prompt_tokens (int): Number of input tokens used
            completion_tokens (int): Number of tokens in the generated response
        """
        total_tokens = prompt_tokens + completion_tokens

        try:
            async with get_db_session() as session:
                query_log = QueryLog(
                    userid=userid,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    query_timestamp=datetime.now(),
                )
                session.add(query_log)
                await session.commit()
                logger.info(
                    f"Query logged for user {userid} with total tokens {total_tokens}."
                )
        except SQLAlchemyError as e:
            logger.error(f"Error logging query: {e}")
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
                user = await session.execute(
                    sqlalchemy.select(UserModel).filter(UserModel.apikey == api_key)
                )
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


# Example Usage
async def main():
    # Initialize the database
    await UserManager.initialize_db()

    # Add some users
    bob = await UserManager.insert_user("Bob", "bob@example.com")
    alice = await UserManager.insert_user("Alice", "alice@example.com")

    print(f"Bob's details: {bob}")
    print(f"Alice's details: {alice}")

    # Check API key
    user_name = await UserManager.check_api_key(bob["apikey"])
    print(f"API key validation: {user_name}")

    # Update and retrieve token usage
    await UserManager.update_token_usage(
        bob["userid"], prompt_tokens=50, completion_tokens=20
    )
    usage = await UserManager.get_user_token_usage(bob["userid"])
    print(f"Bob's token usage: {usage}")

    # Log a query
    await UserManager.log_query(
        userid=bob["userid"],
        model="gpt-3.5-turbo",
        prompt_tokens=8,
        completion_tokens=7,
    )


if __name__ == "__main__":
    # Run the example
    import asyncio

    asyncio.run(main())
