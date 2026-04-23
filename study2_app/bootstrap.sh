#!/bin/bash
# PrefQuest Study 2 app — one-shot setup (backend + frontend).
# Usage: bash study2_app/bootstrap.sh
#
# Installs only what the app needs to run. For the full repo (plots,
# ablation scripts, paper tooling), use ../bootstrap.sh at repo root.
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$APP_DIR/.." && pwd)"

PY_ENV_NAME="${PREFQUEST_ENV:-behavior}"
PY_VERSION="${PREFQUEST_PY_VERSION:-3.10}"

echo "=============================================="
echo " PrefQuest Study 2 bootstrap"
echo " app:  $APP_DIR"
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

echo "[2/3] Installing backend deps from study2_app/backend/requirements.txt ..."
"$PIP" install --upgrade pip
"$PIP" install -r "$APP_DIR/backend/requirements.txt"

# ---------- Frontend ----------
echo "[3/3] Installing frontend deps (npm install) ..."
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js (https://nodejs.org) and re-run." >&2
  exit 1
fi
cd "$APP_DIR/frontend"
npm install
cd "$APP_DIR"

# ---------- Summary ----------
echo ""
echo "=============================================="
echo " ✔ Study 2 bootstrap complete."
echo "=============================================="
echo ""
echo "Start the app:"
if command -v conda >/dev/null 2>&1; then
  echo "  conda activate $PY_ENV_NAME"
else
  echo "  source $ROOT/.venv/bin/activate"
fi
echo "  bash study2_app/start.sh"
echo ""
echo "Defaults: backend :8000, frontend :5173, ollama @ 110.42.252.68:8080,"
echo "Dashscope STT key baked into start.sh / voice/stt.py."
