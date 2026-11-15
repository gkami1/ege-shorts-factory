#!/usr/bin/env bash
# End-to-end check of the narrator fine-tuning chain on synthetic sessions (mechanics, not quality).
#   source /workspace/factory.env && bash tools/finetune_selftest.sh
set -euo pipefail
T=work/ft_selftest
rm -rf "$T"; mkdir -p "$T"
python tools/make_fake_sessions.py "$T/sessions" 60
factory voice-dataset "$T"/sessions/session_*.wav --out "$T/data" --config config/pod.yaml
bash tools/cosyvoice_finetune.sh "$T/data" ft_selftest 2
COSYVOICE_MODEL_DIR=${WORKSPACE:-/workspace}/cache/models/ft_selftest \
    factory say "Десять умножить на восемь — восемьдесят. Ответ: восемьдесят!" \
    --config config/pod.yaml --out "$T/finetuned_say.wav"
echo "OK: fine-tuned model synthesizes → $T/finetuned_say.wav"
