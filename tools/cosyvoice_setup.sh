#!/usr/bin/env bash
# CosyVoice 3 (Apache-2.0) in its OWN venv: it pins torch 2.3.1 / numpy 1.26 / transformers 4.51,
# which clash with the factory venv. The factory talks to it through a worker process
# (src/factory/speech/tts/cosyvoice_worker.py). Run after tools/pod_setup.sh:
#   bash tools/cosyvoice_setup.sh
set -euo pipefail
WS=${WORKSPACE:-/workspace}
CV=${COSYVOICE_DIR:-/opt/CosyVoice}
CVENV=${COSYVOICE_VENV:-/opt/cosyvoice-venv}
MODEL=${COSYVOICE_MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
MODEL_DIR=$WS/cache/models/$(basename "$MODEL")
export PATH=$HOME/.local/bin:$PATH

[ -d "$CV" ] || git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git "$CV"
[ -x "$CVENV/bin/python" ] || uv venv -q -p 3.10 "$CVENV"
# Inference-only subset of requirements.txt (no deepspeed/tensorrt/gradio/servers/training).
grep -v -E '^(deepspeed|tensorrt|gradio|fastapi|uvicorn|grpcio|tensorboard|lightning|openai-whisper|--extra-index-url)' \
    "$CV/requirements.txt" > /tmp/cv_req.txt
# openai-whisper 20231117's setup.py needs pkg_resources (gone from new setuptools): build it without isolation
VIRTUAL_ENV=$CVENV uv pip install -q "setuptools<70" wheel
VIRTUAL_ENV=$CVENV uv pip install -q --no-build-isolation openai-whisper==20231117
VIRTUAL_ENV=$CVENV uv pip install -q --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/cv_req.txt huggingface_hub
VIRTUAL_ENV=$CVENV uv pip install -q lightning==2.2.4 || true   # imported by some modules

if [ ! -f "$MODEL_DIR/.complete" ]; then
    "$CVENV/bin/python" -c "from huggingface_hub import snapshot_download as d; d('$MODEL', local_dir='$MODEL_DIR')"
    touch "$MODEL_DIR/.complete"
fi
grep -q COSYVOICE_DIR "$WS/factory.env" || cat >> "$WS/factory.env" <<EOF
export COSYVOICE_DIR=$CV COSYVOICE_PYTHON=$CVENV/bin/python COSYVOICE_MODEL_DIR=$MODEL_DIR
EOF
echo "CosyVoice ready: $MODEL_DIR  (re-source $WS/factory.env)"
