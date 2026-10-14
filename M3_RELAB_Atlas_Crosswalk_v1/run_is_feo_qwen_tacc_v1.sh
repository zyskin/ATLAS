#!/usr/bin/env bash
set -euo pipefail

# Run only on a TACC compute node (interactive allocation or sbatch job).
# QWEN_BACKEND=cpu is the default; set QWEN_BACKEND=gpu on an A100 node.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATLAS_PROJECT=${ATLAS_PROJECT:-$SCRIPT_DIR}
QWEN_BACKEND=${QWEN_BACKEND:-cpu}
QWEN_LIMIT=${QWEN_LIMIT:-0}
QWEN_PORT=${QWEN_PORT:-8080}
QWEN_CONDA_ROOT=${QWEN_CONDA_ROOT:-$SCRATCH/qwen-local/miniforge3}
QWEN_MODEL=${QWEN_MODEL:-$SCRATCH/qwen-local/models/qwen3.8-27b/Qwen3.8-27B-Q6_K.gguf}

unset PYTHONPATH
source "$QWEN_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qwen-local
hash -r

QWEN_LLAMA_ROOT=${QWEN_LLAMA_ROOT:-$CONDA_PREFIX/opt/llama.cpp-v0.5.0}
PYTHON_BIN="$CONDA_PREFIX/bin/python"

case "$QWEN_BACKEND" in
  cpu)
    LLAMA_SERVER="$QWEN_LLAMA_ROOT/build-cpu/bin/llama-server"
    THREADS=${QWEN_THREADS:-120}
    backend_args=(--n-gpu-layers 0)
    ;;
  gpu)
    LLAMA_SERVER="$QWEN_LLAMA_ROOT/build-cuda/bin/llama-server"
    THREADS=${QWEN_THREADS:-12}
    backend_args=(--n-gpu-layers 99)
    ;;
  *)
    echo "ERROR: QWEN_BACKEND must be cpu or gpu" >&2
    exit 2
    ;;
esac

for required in "$PYTHON_BIN" "$LLAMA_SERVER" "$QWEN_MODEL" \
  "$ATLAS_PROJECT/data/is_feo_v1/atlas_is_feo_review_chunks.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required path missing: $required" >&2
    if [[ "$required" == "$LLAMA_SERVER" ]]; then
      echo "Build it with: cmake --build $QWEN_LLAMA_ROOT/build-$QWEN_BACKEND --target llama-server -j 8" >&2
    fi
    exit 2
  fi
done

LOG_DIR="$ATLAS_PROJECT/data/is_feo_v1/local_ai"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/llama_server_${QWEN_BACKEND}_${SLURM_JOB_ID:-interactive}.log"

"$LLAMA_SERVER" \
  --model "$QWEN_MODEL" \
  --host 127.0.0.1 \
  --port "$QWEN_PORT" \
  --ctx-size 8192 \
  --parallel 1 \
  --threads "$THREADS" \
  "${backend_args[@]}" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

export QWEN_HEALTH_URL="http://127.0.0.1:$QWEN_PORT/health"
"$PYTHON_BIN" - <<'PY'
import os
import time
import urllib.request

url = os.environ["QWEN_HEALTH_URL"]
last_error = None
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print("llama-server is ready")
                break
    except Exception as exc:
        last_error = exc
    time.sleep(1)
else:
    raise SystemExit(f"llama-server did not become ready: {last_error}")
PY

limit_args=()
if [[ "$QWEN_LIMIT" != 0 ]]; then
  limit_args=(--limit "$QWEN_LIMIT")
fi

set +e
"$PYTHON_BIN" "$ATLAS_PROJECT/adjudicate_atlas_is_feo_local_ai_v1.py" \
  --project-root "$ATLAS_PROJECT" \
  --api-url "http://127.0.0.1:$QWEN_PORT/v1/chat/completions" \
  --model "Qwen3.8-27B-Q6_K" \
  --no-stream \
  "${limit_args[@]}"
adjudicator_status=$?
set -e

"$PYTHON_BIN" "$ATLAS_PROJECT/summarize_feo_is_feo_status_v1.py" \
  --project-root "$ATLAS_PROJECT"
exit "$adjudicator_status"

