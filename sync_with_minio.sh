#!/bin/bash
set -e
DATASET="$1"
TARGET_DIR="datasets/$DATASET"
mkdir -p "$TARGET_DIR"
if command -v mc >/dev/null 2>&1; then
  mc alias set local http://localhost:9000 minioadmin minioadmin >/dev/null 2>&1 || true
  mc cp --recursive "local/ivadatasets/$DATASET" "$TARGET_DIR" >/dev/null 2>&1 || true
fi
