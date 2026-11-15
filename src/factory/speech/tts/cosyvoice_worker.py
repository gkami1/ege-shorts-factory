"""CosyVoice 3 worker. Runs under the CosyVoice venv (NOT the factory venv), stdlib + CosyVoice only.

    python cosyvoice_worker.py --repo /opt/CosyVoice --model-dir … --prompt-wav ref.wav --prompt-text-file ref.txt

Protocol: one JSON request per stdin line {"text", "out", "seed", "speed"} → one JSON reply per
stdout line {"ok": true, "sr": 24000} or {"ok": false, "error": "…"}. Logs go to stderr.
"""

import argparse
import json
import os
import random
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text-file", required=True)
    ap.add_argument("--text-frontend", type=int, default=0)
    args = ap.parse_args()

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol channel
    sys.path[:0] = [args.repo, os.path.join(args.repo, "third_party", "Matcha-TTS")]
    import numpy as np
    import soundfile as sf
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=args.model_dir)
    with open(args.prompt_text_file, encoding="utf-8") as f:
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + f.read().strip()

    def reply(obj):
        real_stdout.write(json.dumps(obj) + "\n")
        real_stdout.flush()

    reply({"ok": True, "ready": True, "sr": model.sample_rate})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            seed = int(req.get("seed", 0))
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            chunks = [
                out["tts_speech"].squeeze(0).cpu().numpy()
                for out in model.inference_zero_shot(
                    req["text"], prompt_text, args.prompt_wav, stream=False,
                    speed=float(req.get("speed", 1.0)), text_frontend=bool(args.text_frontend),
                )
            ]
            sf.write(req["out"], np.concatenate(chunks).astype("float32"), model.sample_rate)
            reply({"ok": True, "sr": model.sample_rate})
        except Exception as e:  # report and keep serving
            reply({"ok": False, "error": repr(e)})


if __name__ == "__main__":
    main()
