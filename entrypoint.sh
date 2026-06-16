#!/bin/sh
# Container entrypoint.
#
# Applies the latest Alembic migrations before exec'ing the CMD. This
# keeps the schema in sync with the ORM models across deploys — no more
# "column does not exist" 500s after pulling new changes.
#
# ``alembic upgrade head`` is idempotent: if the DB is already at head,
# it's a no-op. The only cost is one extra DB round-trip on container
# start.
#
# The entrypoint is intentionally a thin shell wrapper (not Python) so
# it works identically in uvicorn and the celery worker — both services
# use the same image and therefore the same entrypoint.
set -e

echo "[entrypoint] Applying Alembic migrations..."
alembic upgrade head
echo "[entrypoint] Migrations applied."

# Hand off to whatever CMD was set in the Dockerfile / docker-compose.
exec "$@"
