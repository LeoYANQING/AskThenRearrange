#!/bin/bash
# Start backend and frontend for PrefQuest Study 2
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="/Users/yuanda/miniconda3/envs/behavior/bin"

export LLM_BACKEND=ollama
export LLM_MODEL=qwen3
export LLM_BASE_URL=http://110.42.252.68:8080

# STT via Aliyun Dashscope (paraformer-realtime-v2). Fill in your key below,
# or export DASHSCOPE_API_KEY in your shell before running this script.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

echo "Starting backend on http://localhost:8000 ..."
cd "$ROOT"
"$CONDA/uvicorn" study2_app.backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
cd "$ROOT/study2_app/frontend"
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "Backend:  http://127.0.0.1:8000/docs"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."

wait
