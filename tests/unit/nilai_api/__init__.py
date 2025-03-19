import pytest
import uuid
from datetime import datetime
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
        userid = self.generate_user_id()
        apikey = self.generate_api_key()

        user_data = {
            "userid": userid,
            "name": name,
            "email": email,
            "apikey": apikey,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "queries": 0,
            "signup_date": datetime.now(),
            "last_activity": None,
        }

        self.users[userid] = user_data
        return {"userid": userid, "apikey": apikey}

    async def check_api_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key in the mock database."""
        for user in self.users.values():
            if user["apikey"] == api_key:
                return {"name": user["name"], "userid": user["userid"]}
        return None

    async def update_token_usage(
        self, userid: str, prompt_tokens: int, completion_tokens: int
    ):
        """Update token usage for a specific user."""
        if userid in self.users:
            user = self.users[userid]
            user["prompt_tokens"] += prompt_tokens
            user["completion_tokens"] += completion_tokens
            user["queries"] += 1
            user["last_activity"] = datetime.now()

    async def log_query(
        self, userid: str, model: str, prompt_tokens: int, completion_tokens: int
    ):
        """Log a user's query in the mock database."""
        query_log = {
            "id": self._next_query_log_id,
            "userid": userid,
            "query_timestamp": datetime.now(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        self.query_logs[self._next_query_log_id] = query_log
        self._next_query_log_id += 1

    async def get_token_usage(self, userid: str) -> Optional[Dict[str, Any]]:
        """Get token usage for a specific user."""
        user = self.users.get(userid)
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

    async def get_user_token_usage(self, userid: str) -> Optional[Dict[str, int]]:
        """Retrieve total token usage for a user."""
        user = self.users.get(userid)
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
