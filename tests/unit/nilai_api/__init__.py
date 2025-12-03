import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class MockUserDatabase:
    def __init__(self):
        """Initialize a mock database for testing UserManager functionality."""
        self.users = {}
        self.query_logs = {}
        self._next_query_log_id = 1

    def generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return str(uuid.uuid4())

    def generate_api_key(self) -> str:
        """Generate a unique API key."""
        return str(uuid.uuid4())

    async def insert_user(self, name: str, email: str) -> Dict[str, str]:
        """Insert a new user into the mock database."""
        user_id = self.generate_user_id()
        apikey = self.generate_api_key()

        user_data = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "apikey": apikey,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "queries": 0,
            "signup_date": datetime.now(timezone.utc),
            "last_activity": None,
        }

        self.users[user_id] = user_data
        return {"user_id": user_id, "apikey": apikey}

    async def check_api_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key in the mock database."""
        for user in self.users.values():
            if user["apikey"] == api_key:
                return {"name": user["name"], "user_id": user["user_id"]}
        return None

    async def update_token_usage(
        self, user_id: str, prompt_tokens: int, completion_tokens: int
    ):
        """Update token usage for a specific user."""
        if user_id in self.users:
            user = self.users[user_id]
            user["prompt_tokens"] += prompt_tokens
            user["completion_tokens"] += completion_tokens
            user["queries"] += 1
            user["last_activity"] = datetime.now(timezone.utc)

    async def log_query(
        self, user_id: str, model: str, prompt_tokens: int, completion_tokens: int
    ):
        """Log a user's query in the mock database."""
        query_log = {
            "id": self._next_query_log_id,
            "user_id": user_id,
            "query_timestamp": datetime.now(timezone.utc),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        self.query_logs[self._next_query_log_id] = query_log
        self._next_query_log_id += 1

    async def get_token_usage(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get token usage for a specific user."""
        user = self.users.get(user_id)
        if user:
            return {
                "prompt_tokens": user["prompt_tokens"],
                "completion_tokens": user["completion_tokens"],
                "total_tokens": user["prompt_tokens"] + user["completion_tokens"],
                "queries": user["queries"],
            }
        return None

    async def get_all_users(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve all users from the mock database."""
        return list(self.users.values()) if self.users else None

    async def get_user_token_usage(self, user_id: str) -> Optional[Dict[str, int]]:
        """Retrieve total token usage for a user."""
        user = self.users.get(user_id)
        if user:
            return {
                "prompt_tokens": user["prompt_tokens"],
                "completion_tokens": user["completion_tokens"],
                "queries": user["queries"],
            }
        return None


# Pytest fixture for MockUserDatabase
@pytest.fixture
def mock_db():
    """Fixture to create a fresh MockUserDatabase for each test."""
    return MockUserDatabase()
