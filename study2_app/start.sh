#!/bin/bash
# Start backend and frontend for PrefQuest Study 2
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve Python/uvicorn. Priority:
#   1. $PREFQUEST_PYTHON_BIN  (explicit override: dir containing `uvicorn`)
#   2. Activated conda env    ($CONDA_PREFIX/bin)
#   3. `uvicorn` on PATH
if [ -n "$PREFQUEST_PYTHON_BIN" ]; then
  UVICORN="$PREFQUEST_PYTHON_BIN/uvicorn"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/uvicorn" ]; then
  UVICORN="$CONDA_PREFIX/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN="$(command -v uvicorn)"
else
  echo "ERROR: cannot find 'uvicorn'. Activate your conda env (e.g. 'conda activate behavior'),"
  echo "       or set PREFQUEST_PYTHON_BIN=/path/to/env/bin"
  exit 1
fi

# LLM backend — override via environment if needed.
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export LLM_MODEL="${LLM_MODEL:-qwen3}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://110.42.252.68:8080}"

# STT via Aliyun Dashscope (paraformer-realtime-v2). Export in your shell.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

BACKEND_PORT="${BACKEND_PORT:-8000}"

echo "Using uvicorn: $UVICORN"
echo "LLM backend:   $LLM_BACKEND @ $LLM_BASE_URL (model=$LLM_MODEL)"
if [ -z "$DASHSCOPE_API_KEY" ]; then
  echo "WARNING: DASHSCOPE_API_KEY is not set — voice STT will fail."
fi

echo "Starting backend on http://127.0.0.1:$BACKEND_PORT ..."
cd "$ROOT"
"$UVICORN" study2_app.backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
cd "$ROOT/study2_app/frontend"
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "Backend:  http://127.0.0.1:$BACKEND_PORT/docs"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."

wait
