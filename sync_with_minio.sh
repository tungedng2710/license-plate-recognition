#!/bin/bash
set -e
DATASET="$1"
TARGET_DIR="datasets/$DATASET"
CONFIG_FILE="minio_config.json"

ENDPOINT=$(jq -r '.endpoint' "$CONFIG_FILE")
ACCESS_KEY=$(jq -r '.access_key' "$CONFIG_FILE")
SECRET_KEY=$(jq -r '.secret_key' "$CONFIG_FILE")
BUCKET=$(jq -r '.bucket' "$CONFIG_FILE")

mkdir -p "$TARGET_DIR"
if command -v mc >/dev/null 2>&1; then
  mc alias set local "$ENDPOINT" "$ACCESS_KEY" "$SECRET_KEY" >/dev/null 2>&1 || true
  mc cp --recursive "local/$BUCKET/$DATASET" "$TARGET_DIR" >/dev/null 2>&1 || true
fi
