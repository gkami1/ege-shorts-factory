# Running the factory on a RunPod GPU pod

The GPU runs the narrator voice (Qwen3-TTS) and whisper large-v3. Manim renders on the CPU.
M2 needs **one 24 GB GPU**. M3 (the local LLM) will want 48 GB.

## 1. Create the pod

| Setting | Value |
|---|---|
| GPU | RTX 4090 24 GB (A5000, 3090 or L4 also work; more vCPUs = faster Manim render) |
| Template | **RunPod PyTorch** (any recent Ubuntu 22.04/24.04 image; the setup script brings its own Python 3.12) |
| CUDA filter | **12.8 or newer** (PyPI torch wheels need a recent driver) |
| Container disk | 40 GB (TeX Live + apt) |
| Volume disk | 60 GB at `/workspace`, pod volume or network volume (models, outputs; the venv lives on the container disk) |
| Exposed HTTP ports | `8888` (JupyterLab), `8000` (listening pages; or tunnel it: `ssh -L 8000:localhost:8000 root@<ip> -p <port>`) |
| SSH | add your public key in RunPod → Settings → SSH keys |

## 2. Get the code onto the pod

```bash
cd /workspace
git clone https://github.com/gkami1/ege-shorts-factory.git
cd ege-shorts-factory && git checkout m2-voice
```

(If the repo is made private later, clone with a read-only fine-grained token:
`https://<TOKEN>@github.com/gkami1/ege-shorts-factory.git`.)

Without GitHub, copy a bundle: `git bundle create ../factory.bundle --all` locally, then
`scp -P <port> ../factory.bundle root@<ip>:/workspace/` and `git clone factory.bundle ege-shorts-factory` on the pod.

## 3. Setup and smoke test (~15 min the first time)

```bash
bash tools/pod_setup.sh            # apt + Python 3.12 venv + qwen-tts + fonts; prints GPU check
source /workspace/factory.env      # in every new shell
bash tools/pod_smoke.sh            # tests, one spoken line, one full video with a preset voice
```

The first run downloads Qwen3-TTS (~4 GB per model) and whisper large-v3 (~3 GB) into `/workspace/cache`.

## 4. Design the narrator voice (once)

```bash
factory voice-design --config config/pod.yaml --n 9
cd work && python -m http.server 8000      # open https://<pod-id>-8000.proxy.runpod.net/voices/
```

Each candidate has its designed reference clip plus three test lines read through the production
path (a voice clone of that reference). Pick the number you like, then:

```bash
factory voice-pick <N> --config config/pod.yaml    # → assets/voices/narrator_ru.{wav,txt,json}
```

The voice is synthetic (made by VoiceDesign from a text description), so no real person's
voice rights are involved. The reference clip is the channel's voice, so commit it to the repo.
To steer the timbre, pass your own descriptions: `--instruct "Male, 30, …" --instruct "…"`.

## 5. Bake-off and full build

```bash
factory tts-bakeoff --config config/pod.yaml       # work/bakeoff/report.md + blind page /bakeoff/
factory build --task-id m_profile_07_0001 --config config/pod.yaml
factory batch video_scripts/*.json --config config/pod.yaml
```

Every clip is re-heard by whisper. A clip whose character error rate against the script is
above `tts.qa.max_cer` is re-synthesized with another seed (`work/<id>/tts_qa.json` shows what
whisper heard). `out/<id>.meta.json` records the voice and `commercial_ok`.

## 6. Bring results home, stop the pod

```bash
# on your PC (Git Bash):
scp -P <port> -r root@<ip>:/workspace/ege-shorts-factory/out ./out-pod
scp -P <port> root@<ip>:/workspace/ege-shorts-factory/assets/voices/narrator_ru.* assets/voices/
```

Alternatively, use JupyterLab's file browser → right-click → Download.
**Stop** the pod when done (billing stops for the GPU; the volume is billed at storage rates).
After a restart, run `bash tools/pod_setup.sh` again: apt packages are on the container disk.
