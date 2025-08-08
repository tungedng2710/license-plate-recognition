#!/bin/bash
set -e

DATASET_DIR="$1"
CONFIG_FILE="minio_config.json"

ENDPOINT=$(jq -r '.endpoint' "$CONFIG_FILE")
ACCESS_KEY=$(jq -r '.access_key' "$CONFIG_FILE")
SECRET_KEY=$(jq -r '.secret_key' "$CONFIG_FILE")
BUCKET=$(jq -r '.bucket' "$CONFIG_FILE")

# Configure MinIO alias
if command -v mc >/dev/null 2>&1; then
  mc alias set local "$ENDPOINT" "$ACCESS_KEY" "$SECRET_KEY"

  TARGET_DIR="datasets/$DATASET_DIR"
  mkdir -p "$TARGET_DIR"

  # Mirror dataset from MinIO to local directory
  mc mirror --overwrite "local/$BUCKET/$DATASET_DIR" "$TARGET_DIR/"

  # Update data.yaml files with absolute paths
  cd "$TARGET_DIR"
  for yaml_file in $(find . -name "data.yaml"); do
    sed -i "s|train: |train: $(pwd)/|" "$yaml_file"
    sed -i "s|val: |val: $(pwd)/|" "$yaml_file"
    sed -i "s|test: |test: $(pwd)/|" "$yaml_file"
  done
fi
