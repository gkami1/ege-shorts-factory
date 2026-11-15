# ege-shorts-factory

A pipeline that automatically produces vertical short videos (TikTok / YouTube Shorts / Reels, 1080x1920, 25–45 s) explaining ЕГЭ part-1 problems in Mathematics (профиль + база) and Physics. All on-screen text and voiceover are in Russian.

Everything runs locally: open-weight models only, and all visuals are rendered by code (Manim + ffmpeg).
Design: [docs/M0_design.md](docs/M0_design.md).

## Quick start (Docker; CPU is enough for rendering)

```bash
docker build -t ege-factory .                      # CPU image (local look-dev)
# GPU pod:
# docker build -t ege-factory:gpu --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .

docker run --rm -v "$PWD:/app" -v ege-cache:/cache ege-factory factory validate
docker run --rm -v "$PWD:/app" -v ege-cache:/cache ege-factory factory build --task-id m_profile_07_0001
#   → out/m_profile_07_0001.mp4 + out/m_profile_07_0001.meta.json
docker run --rm -v "$PWD:/app" -v ege-cache:/cache ege-factory factory build --task-id m_profile_07_0001 --draft  # 540p/15fps preview
docker run --rm -v "$PWD:/app" -v ege-cache:/cache ege-factory pytest -q
```

On Windows PowerShell use `${PWD}` instead of `$PWD`. Models (Silero, whisper) download once into the `ege-cache` volume.

**Voice licensing:** the channel is monetized. Local builds use Silero (non-commercial license) for previews only. Publishable videos are built on a GPU pod with Qwen3-TTS (Apache-2.0); `out/<id>.meta.json` records `commercial_ok`. See [docs/runpod.md](docs/runpod.md).

## Writing narration

- Captions show digits and symbols, and the voice reads words: «Ответ: восемьдесят!» shows as «Ответ: 80!».
- Write a number as words when its case matters («равно десяти»). Digits are read in the nominative («$t = 2$ с» → «две секунды»).
- Formulas go in `$…$` (powers, fractions, roots, logs, units). Try one line with `factory say "…"`.
- When the voice stresses a word wrongly, add it to `config/lexicon_ru.yaml`.

## Layout

| Path | What |
|---|---|
| `tasks/` | Task bank (YAML, one per task) |
| `video_scripts/` | Video scripts (JSON). Hand-written for now; LLM-generated from M3 |
| `config/factory.yaml` | All settings |
| `src/factory/` | `bank`, `speech`, `captions`, `render`, `assemble`, `llm` (M3), `verify` (M3) |
| `out/`, `work/` | Results, plus per-task intermediates (clips, timeline, ASS, silent render) |

## Milestones

- [x] M0 design · [x] M1 end-to-end from a hand-written script
- [x] M2 LaTeX→speech normalizer + digit captions, Qwen3-TTS narrator (voice design → clone), TTS QA re-takes, bake-off, RunPod setup
  - [ ] pick the narrator on the pod, run the bake-off, publishable build
- [ ] M3 local LLM scriptwriter + verification gate + review queue
  - [x] verification gate (`factory verify`): SymPy step chain, final value, LaTeX ↔ machine form, sheet arithmetic, spoken answer, units, bank cross-check; review queue (`review/<id>/`, `factory review-accept`); self-consistency helper
  - [ ] LLM scriptwriter (pod)
- [ ] M4 batch, variation knobs, physics templates
- [ ] M5 batch quality report
