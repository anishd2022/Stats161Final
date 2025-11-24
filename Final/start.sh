#!/bin/bash
# Startup script for Render deployment
# Ensures PORT is set and gunicorn binds correctly

# Get port from environment variable (Render sets this)
PORT=${PORT:-10000}

echo "Starting application on port $PORT..."

# Start gunicorn with explicit port binding
exec gunicorn \
  --bind 0.0.0.0:$PORT \
  --workers 2 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  app:app

