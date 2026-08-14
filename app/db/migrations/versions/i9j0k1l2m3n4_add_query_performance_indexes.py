"""add_query_performance_indexes

Revision ID: i9j0k1l2m3n4
Revises: b1c2d3e4f5a6
Create Date: 2026-07-31

Indexes on frequently filtered FK / ORDER BY columns used by
dashboard, chat history, quiz history, progress, and notifications.
Postgres does not auto-index foreign keys.
"""
from alembic import op

revision = "i9j0k1l2m3n4"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # learning_sessions — list by user, ownership checks
    op.create_index(
        "ix_learning_sessions_user_created",
        "learning_sessions",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_learning_sessions_user_level",
        "learning_sessions",
        ["user_id", "level"],
        unique=False,
    )

    # curricula — latest version per session
    op.create_index(
        "ix_curricula_session_version",
        "curricula",
        ["session_id", "version"],
        unique=False,
    )

    # topics — curriculum page, week/day order
    op.create_index(
        "ix_topics_session_week_day",
        "topics",
        ["session_id", "week_number", "day_number"],
        unique=False,
    )

    # learning_modules — get by topic / list by session
    op.create_index(
        "ix_learning_modules_topic_id",
        "learning_modules",
        ["topic_id"],
        unique=False,
    )
    op.create_index(
        "ix_learning_modules_session_id",
        "learning_modules",
        ["session_id"],
        unique=False,
    )

    # resource_links
    op.create_index(
        "ix_resource_links_session_id",
        "resource_links",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_resource_links_topic_id",
        "resource_links",
        ["topic_id"],
        unique=False,
    )

    # chat_messages — history, RAGAS aggregates, heatmap join
    op.create_index(
        "ix_chat_messages_session_topic_created",
        "chat_messages",
        ["session_id", "topic_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_chat_messages_session_role",
        "chat_messages",
        ["session_id", "role"],
        unique=False,
    )
    op.create_index(
        "ix_chat_messages_session_created",
        "chat_messages",
        ["session_id", "created_at"],
        unique=False,
    )

    # quiz_results — cooldown, history, daily study, heatmap
    op.create_index(
        "ix_quiz_results_session_topic_created",
        "quiz_results",
        ["session_id", "topic_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_quiz_results_session_created",
        "quiz_results",
        ["session_id", "created_at"],
        unique=False,
    )

    # progress_signals — evaluate, daily study time
    op.create_index(
        "ix_progress_signals_session_topic_created",
        "progress_signals",
        ["session_id", "topic_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_progress_signals_session_type_created",
        "progress_signals",
        ["session_id", "signal_type", "created_at"],
        unique=False,
    )

    # agent_logs — WebSocket history restore
    op.create_index(
        "ix_agent_logs_session_created",
        "agent_logs",
        ["session_id", "created_at"],
        unique=False,
    )

    # notifications — inbox
    op.create_index(
        "ix_notifications_user_created",
        "notifications",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_notifications_user_unread",
        "notifications",
        ["user_id", "is_read"],
        unique=False,
    )

    # users — daily reminder beat filter
    op.create_index(
        "ix_users_last_login_date",
        "users",
        ["last_login_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_users_last_login_date", table_name="users")
    op.drop_index("ix_notifications_user_unread", table_name="notifications")
    op.drop_index("ix_notifications_user_created", table_name="notifications")
    op.drop_index("ix_agent_logs_session_created", table_name="agent_logs")
    op.drop_index("ix_progress_signals_session_type_created", table_name="progress_signals")
    op.drop_index("ix_progress_signals_session_topic_created", table_name="progress_signals")
    op.drop_index("ix_quiz_results_session_created", table_name="quiz_results")
    op.drop_index("ix_quiz_results_session_topic_created", table_name="quiz_results")
    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_role", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_topic_created", table_name="chat_messages")
    op.drop_index("ix_resource_links_topic_id", table_name="resource_links")
    op.drop_index("ix_resource_links_session_id", table_name="resource_links")
    op.drop_index("ix_learning_modules_session_id", table_name="learning_modules")
    op.drop_index("ix_learning_modules_topic_id", table_name="learning_modules")
    op.drop_index("ix_topics_session_week_day", table_name="topics")
    op.drop_index("ix_curricula_session_version", table_name="curricula")
    op.drop_index("ix_learning_sessions_user_level", table_name="learning_sessions")
    op.drop_index("ix_learning_sessions_user_created", table_name="learning_sessions")
