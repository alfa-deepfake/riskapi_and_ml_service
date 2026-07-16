#!/usr/bin/env bash
# Download model weights into ./models for the Compose runtime mount.
# Run from the riskapi_and_ml_service repository root:
#   bash scripts/download_runtime_models.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASR_DIR="$ROOT/models/asr/faster-whisper-medium"
INSIGHTFACE_ROOT="$ROOT/models/insightface"
ASR_REVISION="08e178d48790749d25932bbc082711ddcfdfbc4f"
ASR_BASE="https://huggingface.co/Systran/faster-whisper-medium/resolve/$ASR_REVISION"
INSIGHTFACE_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

download() {
  local url=$1
  local output=$2
  mkdir -p "$(dirname "$output")"
  curl --fail --location --retry 3 --continue-at - --output "$output" "$url"
}

mkdir -p "$ASR_DIR"
for file in model.bin config.json tokenizer.json vocabulary.txt; do
  download "$ASR_BASE/$file?download=true" "$ASR_DIR/$file"
done

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT
download "$INSIGHTFACE_URL" "$archive"
mkdir -p "$INSIGHTFACE_ROOT/models"
unzip -oq "$archive" -d "$INSIGHTFACE_ROOT/models"

test -f "$ASR_DIR/model.bin"
test -f "$INSIGHTFACE_ROOT/models/buffalo_l/det_10g.onnx"
echo "Runtime models are ready under $ROOT/models"
