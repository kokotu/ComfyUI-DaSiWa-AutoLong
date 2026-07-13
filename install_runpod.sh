#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
TARGET="$COMFY_DIR/custom_nodes/ComfyUI-DaSiWa-AutoLong"
REPO="https://github.com/kokotu/ComfyUI-DaSiWa-AutoLong.git"

if [[ -d "$TARGET/.git" ]]; then
  git -C "$TARGET" pull --ff-only
else
  git clone "$REPO" "$TARGET"
fi

echo "Installed at $TARGET"
echo "Restart ComfyUI to load the nodes."
