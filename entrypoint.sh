#!/bin/bash
set -e

echo "Applying database migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"
