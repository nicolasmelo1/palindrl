#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${HF_SPACE_REPO_ID:-nicolasmelo/palindromon-0.116M-space}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING_DIR="${TMPDIR:-/tmp}/palindromon-space-upload"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: hf CLI not found. Install or activate huggingface_hub first." >&2
  exit 1
fi

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

rsync -a --delete \
  --exclude "__pycache__/" \
  "$ROOT_DIR/space/" \
  "$STAGING_DIR/"

rsync -a --delete \
  --exclude "__pycache__/" \
  "$ROOT_DIR/palindrl/" \
  "$STAGING_DIR/palindrl/"

rsync -a --delete \
  "$ROOT_DIR/checkpoints/" \
  "$STAGING_DIR/checkpoints/"

echo "Uploading Space repo: $REPO_ID"
echo "Staged files in: $STAGING_DIR"

hf upload "$REPO_ID" "$STAGING_DIR" . \
  --repo-type space \
  --exclude "__pycache__/*"
