"""create initial set of tables

Revision ID: da89d3230653
Revises:
Create Date: 2025-02-06 10:57:42.966226

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from nilai_api.config import CONFIG


# revision identifiers, used by Alembic.
revision: str = "da89d3230653"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("userid", sa.String(50), primary_key=True, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("apikey", sa.String(50), unique=True, nullable=False, index=True),
        sa.Column("prompt_tokens", sa.Integer, default=0, nullable=False),
        sa.Column("completion_tokens", sa.Integer, default=0, nullable=False),
        sa.Column("queries", sa.Integer, default=0, nullable=False),
        sa.Column(
            "signup_date", sa.DateTime, server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_activity", sa.DateTime, nullable=True),
        sa.Column(
            "ratelimit_day",
            sa.Integer,
            default=CONFIG.rate_limiting.user_rate_limit_day,
            nullable=True,
        ),
        sa.Column(
            "ratelimit_hour",
            sa.Integer,
            default=CONFIG.rate_limiting.user_rate_limit_hour,
            nullable=True,
        ),
        sa.Column(
            "ratelimit_minute",
            sa.Integer,
            default=CONFIG.rate_limiting.user_rate_limit_minute,
            nullable=True,
        ),
    )
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "userid",
            sa.String(50),
            sa.ForeignKey("users.userid"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "query_timestamp", sa.DateTime, server_default=sa.func.now(), nullable=False
        ),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False),
        sa.Column("completion_tokens", sa.Integer, nullable=False),
        sa.Column("total_tokens", sa.Integer, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("query_logs")
