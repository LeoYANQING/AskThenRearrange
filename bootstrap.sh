#!/bin/bash
# PrefQuest / AskThenRearrange — one-shot environment setup.
# Usage: bash bootstrap.sh
#
# Installs Python + frontend dependencies. Safe to re-run.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY_ENV_NAME="${PREFQUEST_ENV:-behavior}"
PY_VERSION="${PREFQUEST_PY_VERSION:-3.10}"

echo "=============================================="
echo " PrefQuest bootstrap"
echo " repo: $ROOT"
echo "=============================================="

# ---------- Python env ----------
PIP=""
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$PY_ENV_NAME"; then
    echo "[1/3] conda env '$PY_ENV_NAME' not found — creating with Python $PY_VERSION ..."
    conda create -y -n "$PY_ENV_NAME" "python=$PY_VERSION"
  else
    echo "[1/3] conda env '$PY_ENV_NAME' already exists."
  fi
  conda activate "$PY_ENV_NAME"
  PIP="$CONDA_PREFIX/bin/pip"
elif [ -n "$VIRTUAL_ENV" ]; then
  echo "[1/3] Using active venv at $VIRTUAL_ENV"
  PIP="$VIRTUAL_ENV/bin/pip"
elif command -v python3 >/dev/null 2>&1; then
  echo "[1/3] No conda found — creating local .venv with $(python3 --version)"
  python3 -m venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PIP="$ROOT/.venv/bin/pip"
else
  echo "ERROR: neither conda nor python3 found. Install one and re-run." >&2
  exit 1
fi

echo "[2/3] Installing Python deps from requirements.txt ..."
"$PIP" install --upgrade pip
"$PIP" install -r "$ROOT/requirements.txt"

# ---------- Frontend ----------
echo "[3/3] Installing frontend deps (npm install) ..."
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js (https://nodejs.org) and re-run." >&2
  exit 1
fi
cd "$ROOT/study2_app/frontend"
npm install
cd "$ROOT"

# ---------- Summary ----------
echo ""
echo "=============================================="
echo " ✔ Bootstrap complete."
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Set STT key:  export DASHSCOPE_API_KEY=sk-..."
if command -v conda >/dev/null 2>&1; then
  echo "  2. Activate env: conda activate $PY_ENV_NAME"
else
  echo "  2. Activate env: source $ROOT/.venv/bin/activate"
fi
echo "  3. Start app:    bash study2_app/start.sh"
echo ""
echo "LLM backend defaults to ollama @ http://110.42.252.68:8080 (model=qwen3)."
echo "Override with LLM_BACKEND / LLM_MODEL / LLM_BASE_URL / LLM_API_KEY env vars."
