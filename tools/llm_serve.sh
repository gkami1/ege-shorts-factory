#!/usr/bin/env bash
# Start the scriptwriter LLM (vLLM, OpenAI-compatible) on the pod, in its own venv.
#   bash tools/llm_serve.sh                       # Qwen/Qwen3.6-35B-A3B, FP8 on the fly, port 8011 (RunPod's nginx holds 8001)
#   MODEL=Qwen/Qwen3.6-27B bash tools/llm_serve.sh
#   bash tools/llm_serve.sh stop
# The served model name equals $MODEL, so set llm.model accordingly (or pass --config).
# The LLM takes most of a 48 GB GPU: write scripts first, stop the server, then build videos (TTS/whisper).
set -euo pipefail
WS=${WORKSPACE:-/workspace}
VENV=${VLLM_VENV:-/opt/vllm-venv}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
PORT=${PORT:-8011}
LOG=$WS/vllm.log
export HF_HOME=${HF_HOME:-$WS/cache/hf} PATH=$HOME/.local/bin:$PATH

if [ "${1:-}" = stop ]; then
    pkill -f "vllm serve" && echo "stopped" || echo "not running"
    exit 0
fi
if [ ! -x "$VENV/bin/vllm" ]; then
    echo "== installing vLLM into $VENV (~5 min)"
    uv venv -q -p 3.12 "$VENV"
    VIRTUAL_ENV=$VENV uv pip install -q "vllm>=0.19.0" --torch-backend=auto
fi
pkill -f "vllm serve" 2>/dev/null || true
while pgrep -f "vllm serve" >/dev/null; do sleep 2; done   # the old server must free the GPU first
echo "== serving $MODEL on :$PORT (log: $LOG)"
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
    --quantization fp8 --max-model-len "${MAX_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.88}" \
    --max-num-seqs "${MAX_SEQS:-32}" \
    --reasoning-parser qwen3 --language-model-only ${VLLM_EXTRA:-} > "$LOG" 2>&1 &
PID=$!
# max-num-seqs: Qwen3.6 is hybrid (linear attention): each running sequence needs a Mamba cache block, and
# the default 256 doesn't fit next to the FP8 weights on 48 GB. `factory write/llm-eval` use 4 jobs × 5 samples.
for i in $(seq 1 360); do   # loading weights from the network volume alone takes ~15 min
    if curl -sf "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""; then echo "ready after $((i * 5))s"; exit 0; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM died:"; tail -30 "$LOG"; exit 1; fi
    sleep 5
done
echo "not ready after 30 min; see $LOG"; exit 1
