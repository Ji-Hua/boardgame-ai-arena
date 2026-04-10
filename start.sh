#!/bin/sh
# start.sh — local development launcher
# Rebuilds and starts all services via Docker Compose.
set -e

cd "$(dirname "$0")"

echo "Stopping existing containers..."
docker compose down

echo "Building and starting services..."
docker compose up --build
