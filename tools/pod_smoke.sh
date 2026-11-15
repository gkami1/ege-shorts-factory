#!/usr/bin/env bash
# End-to-end check on the pod before designing the voice (uses a preset Qwen speaker).
#   source /workspace/factory.env && bash tools/pod_smoke.sh
set -euo pipefail
pytest -q
factory validate
factory say "Дано: два в степени икс равно десяти. Ответ: восемьдесят!" \
    --config config/pod_preset.yaml --out work/say_ryan.wav
factory build --task-id m_profile_07_0001 --config config/pod_preset.yaml --out out/smoke
cat out/smoke/m_profile_07_0001.meta.json
cat work/m_profile_07_0001/tts_qa.json
echo "OK: out/smoke/m_profile_07_0001.mp4"
