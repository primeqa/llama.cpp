#!/usr/bin/env bash
# Run llama-embedding on each test input from the HF reference dir.
#
# Usage:
#   scripts/test-granite-llama-cpp.sh <gguf_path> <out_dir> <ref_dir>
#
# Writes:
#   <out_dir>/embeddings.tsv  — one comma-separated embedding per line (last line of llama-embedding output)
#   <out_dir>/stderr.log
set -euo pipefail

MODEL="${1:?missing gguf path}"
OUT="${2:?missing out dir}"
REF="${3:?missing ref dir}"

mkdir -p "$OUT"
: > "$OUT/embeddings.tsv"
: > "$OUT/stderr.log"

mapfile -t INPUTS < <(python3 -c '
import json,sys
for s in json.load(open(sys.argv[1])): print(s)
' "$REF/inputs.json")

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/bin/llama-embedding"
[[ -x "$BIN" ]] || { echo "missing $BIN"; exit 1; }

for s in "${INPUTS[@]}"; do
    "$BIN" \
        -m "$MODEL" \
        --pooling cls \
        --embd-normalize 2 \
        --embd-output-format array \
        -p "$s" \
        --no-warmup \
        -ngl 0 \
        2>>"$OUT/stderr.log" \
        | grep -E '^\[' | head -n 1 >> "$OUT/embeddings.tsv"
done

echo "[llama-cpp] wrote $(wc -l < "$OUT/embeddings.tsv") embeddings to $OUT/embeddings.tsv"
