"""chore: merged database schema updates

Revision ID: 0ba073468afc
Revises: ea942d6c7a00
Create Date: 2025-10-31 09:43:12.022675

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0ba073468afc"
down_revision: Union[str, None] = "9ddf28cf6b6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### merged commands from ea942d6c7a00 and 0ba073468afc ###
    # query_logs: new telemetry columns (with defaults to backfill existing rows)
    op.add_column(
        "query_logs",
        sa.Column(
            "tool_calls", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "temperature", sa.Float(), server_default=sa.text("0.9"), nullable=True
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "max_tokens", sa.Integer(), server_default=sa.text("4096"), nullable=True
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "response_time_ms",
            sa.Integer(),
            server_default=sa.text("-1"),
            nullable=False,
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "model_response_time_ms",
            sa.Integer(),
            server_default=sa.text("-1"),
            nullable=False,
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "tool_response_time_ms",
            sa.Integer(),
            server_default=sa.text("-1"),
            nullable=False,
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "was_streamed",
            sa.Boolean(),
            server_default=sa.text("False"),
            nullable=False,
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "was_multimodal",
            sa.Boolean(),
            server_default=sa.text("False"),
            nullable=False,
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "was_nildb", sa.Boolean(), server_default=sa.text("False"), nullable=False
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "was_nilrag", sa.Boolean(), server_default=sa.text("False"), nullable=False
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "error_code", sa.Integer(), server_default=sa.text("200"), nullable=False
        ),
    )
    op.add_column(
        "query_logs",
        sa.Column(
            "error_message", sa.Text(), server_default=sa.text("'OK'"), nullable=False
        ),
    )

    # query_logs: remove FK to users.userid before dropping the column later
    op.drop_constraint("query_logs_userid_fkey", "query_logs", type_="foreignkey")

    # query_logs: add lockid and index, drop legacy userid and its index
    op.add_column(
        "query_logs", sa.Column("lockid", sa.String(length=75), nullable=False)
    )
    op.drop_index("ix_query_logs_userid", table_name="query_logs")
    op.create_index(
        op.f("ix_query_logs_lockid"), "query_logs", ["lockid"], unique=False
    )
    op.drop_column("query_logs", "userid")

    # users: drop legacy token counters
    op.drop_column("users", "prompt_tokens")
    op.drop_column("users", "completion_tokens")

    # users: reshape identity columns and indexes
    op.add_column("users", sa.Column("user_id", sa.String(length=75), nullable=False))
    op.drop_index("ix_users_apikey", table_name="users")
    op.drop_index("ix_users_userid", table_name="users")
    op.create_index(op.f("ix_users_user_id"), "users", ["user_id"], unique=False)
    op.drop_column("users", "last_activity")
    op.drop_column("users", "userid")
    op.drop_column("users", "apikey")
    op.drop_column("users", "signup_date")
    op.drop_column("users", "queries")
    op.drop_column("users", "name")
    # ### end merged commands ###


def downgrade() -> None:
    # ### revert merged commands back to 9ddf28cf6b6f ###
    # users: restore legacy columns and indexes
    op.add_column("users", sa.Column("name", sa.VARCHAR(length=100), nullable=False))
    op.add_column("users", sa.Column("queries", sa.INTEGER(), nullable=False))
    op.add_column(
        "users",
        sa.Column(
            "signup_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column("users", sa.Column("apikey", sa.VARCHAR(length=75), nullable=False))
    op.add_column("users", sa.Column("userid", sa.VARCHAR(length=75), nullable=False))
    op.add_column(
        "users",
        sa.Column("last_activity", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.drop_index(op.f("ix_users_user_id"), table_name="users")
    op.create_index("ix_users_userid", "users", ["userid"], unique=False)
    op.create_index("ix_users_apikey", "users", ["apikey"], unique=False)
    op.drop_column("users", "user_id")
    op.add_column(
        "users",
        sa.Column(
            "completion_tokens",
            sa.INTEGER(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "prompt_tokens", sa.INTEGER(), server_default=sa.text("0"), nullable=False
        ),
    )

    # query_logs: restore userid, index and FK; drop new columns
    op.add_column(
        "query_logs", sa.Column("userid", sa.VARCHAR(length=75), nullable=False)
    )
    op.drop_index(op.f("ix_query_logs_lockid"), table_name="query_logs")
    op.create_index("ix_query_logs_userid", "query_logs", ["userid"], unique=False)
    op.create_foreign_key(
        "query_logs_userid_fkey", "query_logs", "users", ["userid"], ["userid"]
    )
    op.drop_column("query_logs", "lockid")
    op.drop_column("query_logs", "error_message")
    op.drop_column("query_logs", "error_code")
    op.drop_column("query_logs", "was_nilrag")
    op.drop_column("query_logs", "was_nildb")
    op.drop_column("query_logs", "was_multimodal")
    op.drop_column("query_logs", "was_streamed")
    op.drop_column("query_logs", "tool_response_time_ms")
    op.drop_column("query_logs", "model_response_time_ms")
    op.drop_column("query_logs", "response_time_ms")
    op.drop_column("query_logs", "max_tokens")
    op.drop_column("query_logs", "temperature")
    op.drop_column("query_logs", "tool_calls")
    # ### end revert ###
