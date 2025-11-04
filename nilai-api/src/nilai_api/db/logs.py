import logging
import time
from datetime import datetime, timezone
from typing import Optional

from nilai_common import Usage
import sqlalchemy

from sqlalchemy import Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.exc import SQLAlchemyError
from nilai_api.db import Base, Column, get_db_session

logger = logging.getLogger(__name__)


# New QueryLog Model for tracking individual queries
class QueryLog(Base):
    __tablename__ = "query_logs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)  # type: ignore
    user_id: str = Column(String(75), nullable=False, index=True)  # type: ignore
    lockid: str = Column(String(75), nullable=False, index=True)  # type: ignore
    query_timestamp: datetime = Column(
        DateTime(timezone=True), server_default=sqlalchemy.func.now(), nullable=False
    )  # type: ignore
    model: str = Column(Text, nullable=False)  # type: ignore
    prompt_tokens: int = Column(Integer, nullable=False)  # type: ignore
    completion_tokens: int = Column(Integer, nullable=False)  # type: ignore
    total_tokens: int = Column(Integer, nullable=False)  # type: ignore
    tool_calls: int = Column(Integer, nullable=False)  # type: ignore
    web_search_calls: int = Column(Integer, nullable=False)  # type: ignore
    temperature: Optional[float] = Column(Float, nullable=True)  # type: ignore
    max_tokens: Optional[int] = Column(Integer, nullable=True)  # type: ignore

    response_time_ms: int = Column(Integer, nullable=False)  # type: ignore
    model_response_time_ms: int = Column(Integer, nullable=False)  # type: ignore
    tool_response_time_ms: int = Column(Integer, nullable=False)  # type: ignore

    was_streamed: bool = Column(Boolean, nullable=False)  # type: ignore
    was_multimodal: bool = Column(Boolean, nullable=False)  # type: ignore
    was_nildb: bool = Column(Boolean, nullable=False)  # type: ignore
    was_nilrag: bool = Column(Boolean, nullable=False)  # type: ignore

    error_code: int = Column(Integer, nullable=False)  # type: ignore
    error_message: str = Column(Text, nullable=False)  # type: ignore

    def __repr__(self):
        return f"<QueryLog(user_id={self.user_id}, query_timestamp={self.query_timestamp}, total_tokens={self.total_tokens})>"


class QueryLogContext:
    """
    Context manager for logging query metrics during a request.
    Used as a FastAPI dependency to track request metrics.
    """

    def __init__(self):
        self.user_id: Optional[str] = None
        self.lockid: Optional[str] = None
        self.model: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.tool_calls: int = 0
        self.web_search_calls: int = 0
        self.temperature: Optional[float] = None
        self.max_tokens: Optional[int] = None
        self.was_streamed: bool = False
        self.was_multimodal: bool = False
        self.was_nildb: bool = False
        self.was_nilrag: bool = False
        self.error_code: int = 0
        self.error_message: str = ""

        # Timing tracking
        self.start_time: float = time.monotonic()
        self.model_start_time: Optional[float] = None
        self.model_end_time: Optional[float] = None
        self.tool_start_time: Optional[float] = None
        self.tool_end_time: Optional[float] = None

    def set_user(self, user_id: str) -> None:
        """Set the user ID for this query."""
        self.user_id = user_id

    def set_lockid(self, lockid: str) -> None:
        """Set the lock ID for this query."""
        self.lockid = lockid

    def set_model(self, model: str) -> None:
        """Set the model name for this query."""
        self.model = model

    def set_request_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        was_streamed: bool = False,
        was_multimodal: bool = False,
        was_nildb: bool = False,
        was_nilrag: bool = False,
    ) -> None:
        """Set request parameters."""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.was_streamed = was_streamed
        self.was_multimodal = was_multimodal
        self.was_nildb = was_nildb
        self.was_nilrag = was_nilrag

    def set_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_calls: int = 0,
        web_search_calls: int = 0,
    ) -> None:
        """Set token usage and feature usage."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.tool_calls = tool_calls
        self.web_search_calls = web_search_calls

    def set_error(self, error_code: int, error_message: str) -> None:
        """Set error information."""
        self.error_code = error_code
        self.error_message = error_message

    def start_model_timing(self) -> None:
        """Mark the start of model inference."""
        self.model_start_time = time.monotonic()

    def end_model_timing(self) -> None:
        """Mark the end of model inference."""
        self.model_end_time = time.monotonic()

    def start_tool_timing(self) -> None:
        """Mark the start of tool execution."""
        self.tool_start_time = time.monotonic()

    def end_tool_timing(self) -> None:
        """Mark the end of tool execution."""
        self.tool_end_time = time.monotonic()

    def _calculate_timings(self) -> tuple[int, int, int]:
        """Calculate response times in milliseconds."""
        total_ms = int((time.monotonic() - self.start_time) * 1000)

        model_ms = 0
        if self.model_start_time and self.model_end_time:
            model_ms = int((self.model_end_time - self.model_start_time) * 1000)

        tool_ms = 0
        if self.tool_start_time and self.tool_end_time:
            tool_ms = int((self.tool_end_time - self.tool_start_time) * 1000)

        return total_ms, model_ms, tool_ms

    async def commit(self) -> None:
        """
        Commit the query log to the database.
        Should be called at the end of the request lifecycle.
        """
        if not self.user_id or not self.model:
            logger.warning(
                "Skipping query log: user_id or model not set "
                f"(user_id={self.user_id}, model={self.model})"
            )
            return

        total_ms, model_ms, tool_ms = self._calculate_timings()
        total_tokens = self.prompt_tokens + self.completion_tokens

        try:
            async with get_db_session() as session:
                query_log = QueryLog(
                    user_id=self.user_id,
                    lockid=self.lockid,
                    model=self.model,
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    total_tokens=total_tokens,
                    tool_calls=self.tool_calls,
                    web_search_calls=self.web_search_calls,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    query_timestamp=datetime.now(timezone.utc),
                    response_time_ms=total_ms,
                    model_response_time_ms=model_ms,
                    tool_response_time_ms=tool_ms,
                    was_streamed=self.was_streamed,
                    was_multimodal=self.was_multimodal,
                    was_nilrag=self.was_nilrag,
                    was_nildb=self.was_nildb,
                    error_code=self.error_code,
                    error_message=self.error_message,
                )
                session.add(query_log)
                await session.commit()
                logger.info(
                    f"Query logged for user {self.user_id}: model={self.model}, "
                    f"tokens={total_tokens}, total_ms={total_ms}"
                )
        except SQLAlchemyError as e:
            logger.error(f"Error logging query: {e}")
            # Don't raise - logging failure shouldn't break the request


class QueryLogManager:
    """Static methods for direct query logging (legacy support)."""

    @staticmethod
    async def log_query(
        user_id: str,
        lockid: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: int,
        web_search_calls: int,
        was_streamed: bool,
        was_multimodal: bool,
        was_nilrag: bool,
        was_nildb: bool,
        tool_calls: int = 0,
        temperature: float = 1.0,
        max_tokens: int = 0,
        model_response_time_ms: int = 0,
        tool_response_time_ms: int = 0,
        error_code: int = 0,
        error_message: str = "",
    ):
        """
        Log a user's query (legacy method).
        Consider using QueryLogContext as a dependency instead.
        """
        total_tokens = prompt_tokens + completion_tokens

        try:
            async with get_db_session() as session:
                query_log = QueryLog(
                    user_id=user_id,
                    lockid=lockid,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls,
                    web_search_calls=web_search_calls,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    query_timestamp=datetime.now(timezone.utc),
                    response_time_ms=response_time_ms,
                    model_response_time_ms=model_response_time_ms,
                    tool_response_time_ms=tool_response_time_ms,
                    was_streamed=was_streamed,
                    was_multimodal=was_multimodal,
                    was_nilrag=was_nilrag,
                    was_nildb=was_nildb,
                    error_code=error_code,
                    error_message=error_message,
                )
                session.add(query_log)
                await session.commit()
                logger.info(
                    f"Query logged for user {user_id} with total tokens {total_tokens}."
                )
        except SQLAlchemyError as e:
            logger.error(f"Error logging query: {e}")
            raise

    @staticmethod
    async def get_user_token_usage(user_id: str) -> Optional[Usage]:
        """
        Get aggregated token usage for a specific user using server-side SQL aggregation.
        This is more efficient than fetching all records and calculating in Python.
        """
        try:
            async with get_db_session() as session:
                # Use SQL aggregation functions to calculate on the database server
                query = (
                    sqlalchemy.select(
                        sqlalchemy.func.coalesce(
                            sqlalchemy.func.sum(QueryLog.prompt_tokens), 0
                        ).label("prompt_tokens"),
                        sqlalchemy.func.coalesce(
                            sqlalchemy.func.sum(QueryLog.completion_tokens), 0
                        ).label("completion_tokens"),
                        sqlalchemy.func.coalesce(
                            sqlalchemy.func.sum(QueryLog.total_tokens), 0
                        ).label("total_tokens"),
                        sqlalchemy.func.count().label("queries"),
                    ).where(QueryLog.user_id == user_id)  # type: ignore[arg-type]
                )

                result = await session.execute(query)
                row = result.one_or_none()

                if row is None:
                    return None

                return Usage(
                    prompt_tokens=int(row.prompt_tokens),
                    completion_tokens=int(row.completion_tokens),
                    total_tokens=int(row.total_tokens),
                )
        except SQLAlchemyError as e:
            logger.error(f"Error getting token usage: {e}")
            return None


__all__ = ["QueryLogManager", "QueryLog", "QueryLogContext"]
