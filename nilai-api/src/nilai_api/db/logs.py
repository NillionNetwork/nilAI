import logging
from datetime import datetime, timezone

import sqlalchemy

from sqlalchemy import ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.exc import SQLAlchemyError
from nilai_api.db import Base, Column, get_db_session
from nilai_api.db.users import UserModel

logger = logging.getLogger(__name__)


# New QueryLog Model for tracking individual queries
class QueryLog(Base):
    __tablename__ = "query_logs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)  # type: ignore
    userid: str = Column(
        String(75), ForeignKey(UserModel.userid), nullable=False, index=True
    )  # type: ignore
    query_timestamp: datetime = Column(
        DateTime(timezone=True), server_default=sqlalchemy.func.now(), nullable=False
    )  # type: ignore
    model: str = Column(Text, nullable=False)  # type: ignore
    prompt_tokens: int = Column(Integer, nullable=False)  # type: ignore
    completion_tokens: int = Column(Integer, nullable=False)  # type: ignore
    total_tokens: int = Column(Integer, nullable=False)  # type: ignore
    web_search_calls: int = Column(Integer, nullable=False)  # type: ignore

    def __repr__(self):
        return f"<QueryLog(userid={self.userid}, query_timestamp={self.query_timestamp}, total_tokens={self.total_tokens})>"


class QueryLogManager:
    @staticmethod
    async def log_query(
        userid: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        web_search_calls: int,
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
                    query_timestamp=datetime.now(timezone.utc),
                    web_search_calls=web_search_calls,
                )
                session.add(query_log)
                await session.commit()
                logger.info(
                    f"Query logged for user {userid} with total tokens {total_tokens}."
                )
        except SQLAlchemyError as e:
            logger.error(f"Error logging query: {e}")
            raise


__all__ = ["QueryLogManager", "QueryLog"]
