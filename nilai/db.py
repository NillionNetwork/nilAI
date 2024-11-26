import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import sqlalchemy
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Configure logging

logger = logging.getLogger(__name__)


# Database configuration with better defaults and connection pooling
class DatabaseConfig:
    # Use environment variables in a real-world scenario
    DATABASE_URL = "sqlite:///db/users.sqlite"
    POOL_SIZE = 5
    MAX_OVERFLOW = 10
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600  # Reconnect after 1 hour


# Create base and engine with improved configuration
Base = sqlalchemy.orm.declarative_base()
engine = create_engine(
    DatabaseConfig.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=DatabaseConfig.POOL_SIZE,
    max_overflow=DatabaseConfig.MAX_OVERFLOW,
    pool_timeout=DatabaseConfig.POOL_TIMEOUT,
    pool_recycle=DatabaseConfig.POOL_RECYCLE,
    echo=False,  # Set to True for SQL logging during development
)

# Create session factory with improved settings
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,  # Changed to False for more explicit transaction control
    autoflush=False,  # More control over when to flush
    expire_on_commit=False,  # Keep objects usable after session closes
)


# Enhanced User Model with additional constraints and validation
class User(Base):
    __tablename__ = "users"

    userid = Column(String(36), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    apikey = Column(String(36), unique=True, nullable=False, index=True)
    input_tokens = Column(Integer, default=0, nullable=False)
    generated_tokens = Column(Integer, default=0, nullable=False)

    def __repr__(self):
        return f"<User(userid={self.userid}, name={self.name})>"


@dataclass
class UserData:
    userid: str
    name: str
    apikey: str
    input_tokens: int
    generated_tokens: int


# Context manager for database sessions
@contextmanager
def get_db_session() -> "Generator[Session, Any, Any]":
    """Provide a transactional scope for database operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


class UserManager:
    @staticmethod
    def initialize_db() -> bool:
        """
        Create database tables only if they do not already exist.

        Returns:
            bool: True if tables were created, False if tables already existed
        """
        try:
            # Create an inspector to check existing tables
            inspector = sqlalchemy.inspect(engine)

            # Check if the 'users' table already exists
            if not inspector.has_table("users"):
                # Create all tables that do not exist
                Base.metadata.create_all(bind=engine)
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
    def insert_user(name: str) -> Dict[str, str]:
        """
        Insert a new user into the database.

        Args:
            name (str): Name of the user

        Returns:
            Dict containing userid and apikey
        """
        userid = UserManager.generate_user_id()
        apikey = UserManager.generate_api_key()

        try:
            with get_db_session() as session:
                user = User(userid=userid, name=name, apikey=apikey)
                session.add(user)
                logger.info(f"User {name} added successfully.")
                return {"userid": userid, "apikey": apikey}
        except SQLAlchemyError as e:
            logger.error(f"Error inserting user: {e}")
            raise

    @staticmethod
    def check_api_key(api_key: str) -> Optional[str]:
        """
        Validate an API key.

        Args:
            api_key (str): API key to validate

        Returns:
            User's name if API key is valid, None otherwise
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.apikey == api_key).first()
                return user.name if user else None  # type: ignore
        except SQLAlchemyError as e:
            logger.error(f"Error checking API key: {e}")
            return None

    @staticmethod
    def update_token_usage(userid: str, input_tokens: int, generated_tokens: int):
        """
        Update token usage for a specific user.

        Args:
            userid (str): User's unique ID
            input_tokens (int): Number of input tokens
            generated_tokens (int): Number of generated tokens
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.userid == userid).first()
                if user:
                    user.input_tokens += input_tokens  # type: ignore
                    user.generated_tokens += generated_tokens  # type: ignore
                    logger.info(f"Updated token usage for user {userid}")
                else:
                    logger.warning(f"User {userid} not found")
        except SQLAlchemyError as e:
            logger.error(f"Error updating token usage: {e}")

    @staticmethod
    def get_all_users() -> Optional[List[UserData]]:
        """
        Retrieve all users from the database.

        Returns:
            Dict of users or None if no users found
        """
        try:
            with get_db_session() as session:
                users = session.query(User).all()
                return [
                    UserData(
                        userid=user.userid,  # type: ignore
                        name=user.name,  # type: ignore
                        apikey=user.apikey,  # type: ignore
                        input_tokens=user.input_tokens,  # type: ignore
                        generated_tokens=user.generated_tokens,  # type: ignore
                    )
                    for user in users
                ]
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving all users: {e}")
            return None

    @staticmethod
    def get_user_token_usage(userid: str) -> Optional[Dict[str, int]]:
        """
        Retrieve total token usage for a user.

        Args:
            userid (str): User's unique ID

        Returns:
            Dict of token usage or None if user not found
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.userid == userid).first()
                if user:
                    return {
                        "input_tokens": user.input_tokens,
                        "generated_tokens": user.generated_tokens,
                    }  # type: ignore
                return None
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving token usage: {e}")
            return None


# Example Usage
if __name__ == "__main__":
    # Initialize the database
    UserManager.initialize_db()

    print(UserManager.get_all_users())

    # Add some users
    bob = UserManager.insert_user("Bob")
    alice = UserManager.insert_user("Alice")

    print(f"Bob's details: {bob}")
    print(f"Alice's details: {alice}")

    # Check API key
    user_name = UserManager.check_api_key(bob["apikey"])
    print(f"API key validation: {user_name}")

    # Update and retrieve token usage
    UserManager.update_token_usage(bob["userid"], input_tokens=50, generated_tokens=20)
    usage = UserManager.get_user_token_usage(bob["userid"])
    print(f"Bob's token usage: {usage}")
