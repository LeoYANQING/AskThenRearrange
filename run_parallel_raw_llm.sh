#!/usr/bin/env bash
# Parallel raw_llm baseline: 4 Ollama instances (2 GPUs each) + 4 workers.
# Usage: bash run_parallel_raw_llm.sh [model] [budget-list]
set -euo pipefail

MODEL="${1:-qwen3}"
BUDGET_LIST="${2:-1,2,3,4,5,6}"
DATA="v2/data/scenarios_three_rooms_102.json"
LOG_DIR="v2/logs"
mkdir -p "$LOG_DIR" v2/plots

PORTS=(11434 11435 11436 11437)
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
# 102 episodes split into 4 roughly equal shards
INDEX_RANGES=("0:26" "26:52" "52:78" "78:102")

# ── 1. Start Ollama instances ────────────────────────────────────────────────
echo "[$(date +%T)] Starting 4 Ollama instances..."
OLLAMA_PIDS=()
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpus="${GPU_PAIRS[$i]}"
    CUDA_VISIBLE_DEVICES="$gpus" OLLAMA_HOST="127.0.0.1:$port" ollama serve \
        > "$LOG_DIR/ollama_$port.log" 2>&1 &
    OLLAMA_PIDS+=($!)
    echo "  Instance $i: port=$port gpus=$gpus pid=${OLLAMA_PIDS[$i]}"
done

# ── 2. Wait for all instances to be ready ───────────────────────────────────
echo "[$(date +%T)] Waiting for Ollama instances to be ready..."
for port in "${PORTS[@]}"; do
    for attempt in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$port/api/tags" > /dev/null 2>&1; then
            echo "  Port $port ready."
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo "ERROR: port $port not ready after 30s" >&2
            exit 1
        fi
        sleep 1
    done
done

# ── 3. Pull model on each instance (skip if already present) ─────────────────
echo "[$(date +%T)] Pulling model '$MODEL' on all instances..."
for port in "${PORTS[@]}"; do
    OLLAMA_HOST="127.0.0.1:$port" ollama pull "$MODEL" > "$LOG_DIR/pull_$port.log" 2>&1 &
done
wait
echo "  Model pull done."

# ── 4. Launch 4 parallel workers ─────────────────────────────────────────────
echo "[$(date +%T)] Launching 4 experiment workers..."
WORKER_PIDS=()
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    range="${INDEX_RANGES[$i]}"
    shard="$LOG_DIR/shard_$i.jsonl"
    python -m v2.test_raw_llm \
        --data "$DATA" \
        --index-range "$range" \
        --budget-list "$BUDGET_LIST" \
        --model "$MODEL" \
        --base-url "http://127.0.0.1:$port" \
        --output-jsonl "$shard" \
        > "$LOG_DIR/worker_$i.log" 2>&1 &
    WORKER_PIDS+=($!)
    echo "  Worker $i: range=$range port=$port shard=$shard pid=${WORKER_PIDS[$i]}"
done

# ── 5. Wait for all workers ───────────────────────────────────────────────────
echo "[$(date +%T)] Waiting for workers to finish..."
for i in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$i]}"; then
        echo "  Worker $i finished OK."
    else
        echo "  WARNING: Worker $i exited with error — check $LOG_DIR/worker_$i.log"
    fi
done

# ── 6. Kill Ollama instances ──────────────────────────────────────────────────
echo "[$(date +%T)] Stopping Ollama instances..."
for pid in "${OLLAMA_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done

# ── 7. Merge shards ───────────────────────────────────────────────────────────
echo "[$(date +%T)] Merging shards..."
SHARDS=()
for i in 0 1 2 3; do
    SHARDS+=("$LOG_DIR/shard_$i.jsonl")
done
python -m v2.merge_raw_llm \
    --shards "${SHARDS[@]}" \
    --output-plot "v2/plots/raw_llm_accuracy_curve.png"

echo "[$(date +%T)] Done. Plot saved to v2/plots/raw_llm_accuracy_curve.png"
