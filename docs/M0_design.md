# M0 — Design: ЕГЭ shorts factory

## 1. Pipeline (refined)

```
tasks/*.yaml ──► [llm] solve (free reasoning) ──► [llm] fill VideoScript (JSON-schema constrained) ×N
                                                        │
                                   [verify] answer == official ─┬─► fail ─► review/<id>/ (script + reasons)
                                            SymPy step chain    │
                                            units (pint)        │
                                            N-way agreement ────┘ pass
                                                        ▼
  video_scripts/<id>.json ─► [speech] normalize → TTS one clip per beat ─► timeline (beat windows)
                                                        │
                     ┌──────────────────────────────────┼─────────────────────────────┐
                     ▼                                  ▼                             ▼
          [render] Manim scene (CPU)        [captions] whisper word times      voice.wav (mix)
                     │                         → ASS (rounded boxes)                 │
                     └───────────────► [assemble] ffmpeg: burn ASS + mux + loudnorm ◄┘
                                                        ▼
                                     out/<id>.mp4 + out/<id>.meta.json
```

Changes vs. the original brief, and why:

| Change | Reason |
|---|---|
| **One TTS clip per *beat*** (hook, given, target, each step, substitute, answer, outro), not only per step | Every visual event gets an exact time window, so highlight/circle/morph land on the words without a forced aligner. Pacing "new event every 1–3 s" falls out naturally. |
| **Two-pass LLM**: free-form solve with thinking → separate constrained call that fills the JSON | Grammar-constrained decoding measurably hurts reasoning. Solve freely, then format. |
| **Hand-written scripts are first-class** (`video_scripts/<id>.json`) | M1 runs without an LLM; later, LLM output and human edits use the same format. A reviewer fixes a script in `review/`, then moves it to `video_scripts/`. |
| **Spans are exact substrings of the condition** that never split a `$…$` | The renderer typesets the condition once and isolates spans as Tex parts. Highlight, circle and flash then sit on real glyphs, not guessed coordinates. Checked in `models.check_span`. |
| **The task bank carries an optional `check` (SymPy) expression** | Checks the *bank itself* (a typo in an official answer would otherwise put a wrong answer in a video). Tests enforce it. |
| **Captions are aligned to the known text** (whisper timestamps mapped via difflib onto our words, gaps interpolated by syllables) | We never show whisper's transcription. Captions are always spelled as written, with no ASR mistakes. |
| **Cross-model self-consistency** (M3): N samples from the main model + 1 from a different model family | Two different models agreeing catches more errors than N samples from one model. |

## 2. Repo layout

```
config/factory.yaml        single config (paths, video, timing, tts, captions, render, assemble)
tasks/{math,physics}/*.yaml   task bank, one file per task, file name == id
video_scripts/*.json       VideoScript per task (hand-written now, LLM-written from M3)
schemas/*.schema.json      exported JSON Schemas (`factory schema`)
src/factory/
  models.py                pydantic: Task, VideoScript + cross-checks (spans, answer)
  bank.py                  bank loader
  config.py                YAML loader (deep-merge over defaults)
  timeline.py              beats → windows on a frame grid; voice mix
  pipeline.py              build(): tts → timeline → captions → render → assemble
  cli.py                   `factory build|batch|validate|schema`
  speech/normalize.py      text/LaTeX → spoken Russian (M1 minimal; M2 full + tests)
  speech/tts/              pluggable engines (base.py registry; silero.py)
  captions/align.py        word timestamps (faster-whisper, proportional fallback)
  captions/ass.py          ASS writer: 2–4 words, rounded box, active-word tint
  render/                  tex.py (XeLaTeX templates), sheet.py (exam sheet), fx.py (highlighter,
                           marker circle/arrow, confetti, paper texture), scene.py, run.py, style.py
  assemble/ffmpeg.py       burn + mux + loudnorm (+ ducked music)
  llm/  verify/            M3
tests/                     pytest (bank, spans, timeline, captions; normalizer + verifier in M2/M3)
tools/install_fonts.sh     OFL fonts with Cyrillic, instanced to static weights
Dockerfile                 one image: CPU locally, CUDA on the pod (build args)
```

## 3. Tech choices

| Area | Choice | Why |
|---|---|---|
| Rendering | **Manim CE 0.19, Cairo renderer** | Runs on CPU, deterministic, has `TransformMatchingShapes`/`TransformMatchingTex`. The GPU stays free. |
| TeX | **XeLaTeX + fontspec + mathastext** | Cyrillic without T2A tricks. With `mathastext`, math digits and letters use the text font, so the sheet looks like a КИМ (Liberation Serif ≈ Times) and the big formulas are hand-written (**Caveat Bold**) while TeX still lays out powers and fractions. |
| Fonts | Caveat (handwriting), Montserrat ExtraBold (captions), Liberation Serif (sheet) | All OFL, all with Cyrillic. Variable fonts are instanced to static files so Pango, XeTeX and libass all pick the same face. |
| TTS (M1) | **Silero v4_ru** (CPU) | Easiest to run, good Russian, `+` stress marks, runs faster than real time on CPU. **License: CC BY-NC-SA, so dev/preview only; the channel is monetized (§5.1).** |
| TTS (M2 candidates) | F5-TTS Russian fine-tunes · CosyVoice 3 · XTTS-v2 · Piper | Compared by blind listening plus WER from whisper. F5 Russian fine-tunes (inherit NC from the Emilia data) and XTTS-v2 (Coqui CPML, NC) are non-commercial. CosyVoice (Apache-2.0, Russian listed for v3) and Piper (MIT engine; licenses vary per voice) are the commercial-safe options. |
| Stress | `+` marks in the script, plus an automatic accentizer (RUAccent) in M2 | Math vocabulary (сте́пени, показа́тель, логари́фм) is where TTS stress usually breaks. |
| ASR / captions | **faster-whisper `small`** (int8 on CPU, fp16 on GPU) | Used only for word timestamps on text we already know. |
| LLM (M3) | **Qwen3-30B-A3B-Thinking-2507** (Apache-2.0): AWQ-4bit on 24 GB, FP8 on 48 GB, served with **vLLM** structured outputs (xgrammar). Second opinion: **gpt-oss-20b** (Apache-2.0, ~16 GB) | The MoE has only 3B active parameters, so N-sample self-consistency is cheap. Strong at maths and good Russian. I'll re-check the newest open models when we start M3 and pick by accuracy on our own bank. |
| Verification | SymPy (`parse_latex` + our own tex→sympy fallback), pint for units | Exact checks, no model involved. |
| CLI | Typer | `factory build --task-id …`, `factory batch video_scripts/*.json` |
| Packaging | Docker (Ubuntu 24.04, Python 3.12) | The same image runs on Windows/macOS for look-dev and on the pod. |

## 4. Schemas

Source of truth: `src/factory/models.py`. `factory schema` exports it to `schemas/`.

**Task** (`tasks/**/<id>.yaml`):
```yaml
id: m_profile_07_0001            # <m_profile|m_base|p>_<KIM ##>_<serial ####>
subject: math_profile            # math_profile | math_base | physics
exam_part: 1
kim_number: 7
topic: Вычисления и преобразования — свойства степеней
condition: 'Найдите значение выражения $2^{x+3}$, если $2^x = 10$.'
image: null                      # optional figure path
answer: "80"                     # as written in the blank: "80", "-0,25", "13"
answer_unit: null                # physics: "м/с"
check: "10 * 2**3"               # optional SymPy cross-check of the official answer
source: original (FIPI-style)
tags: [степени]
```

**VideoScript** (what the LLM fills; example `video_scripts/m_profile_07_0001.json`):
```jsonc
{
  "schema_version": 1, "task_id": "m_profile_07_0001",
  "hook":  {"caption": "Эту задачу из ЕГЭ решают за 5 секунд", "narration": "…"},
  "read":  {"given": ["$2^x = 10$"], "given_narration": "…",
            "target": "$2^{x+3}$",   "target_narration": "…"},
  "steps": [{"tex": "2^{x+3}", "narration": "…"},              // 2–6 morph steps
            {"tex": "2^{x}\\cdot 2^{3}", "narration": "…"},
            {"tex": "2^{x}\\cdot 8", "narration": "…"}],
  "substitute": {"flash": "$2^x = 10$", "tex": "10\\cdot 8 = 80", "narration": "…"},
  "answer": {"value": "80", "narration": "Ответ: восемьдесят!"},
  "outro":  {"cta": "Больше разборов — в профиле", "narration": "…"},
  "has_short_trick": 0.95,
  "meta":   {"title": "…", "description": "…", "hashtags": ["#егэ", "…"]}
}
```
M3 will add `steps[i].sympy` (a machine form of each step) and `substitute.values` (e.g. `{"2**x": 10}`) so the verifier can check every step without parsing LaTeX.

## 5. Decisions (answered 2026-11-15)

1. **Monetized: yes.** Silero, F5 Russian fine-tunes, XTTS-v2 and Fish-Speech weights are all non-commercial, so none of them may voice a published video. Silero stays only as a dev/preview voice in M1. M2 standardizes on a commercial-safe voice: CosyVoice 3 (Apache-2.0) or a Piper voice whose own license allows commercial use. Check the license of every voice and model before adoption.
2. **Captions: digits.** The voice says «восемьдесят»; captions show «80». Needs the M2 normalizer's token map.
3. **Hook claims: neutral templates for non-trick tasks.** «решают за 5 секунд» only when `has_short_trick`; otherwise neutral templates («90% ошибаются здесь», «Задача №7 за 30 секунд»). Template set in M4.
4. **Voice: male.** Pick the best male voice from the commercial-safe M2 candidates (`aidar` is Silero, so non-commercial). Whether to rotate several voices is still open; default is one voice.
5. **Branding: yes.** The outro card shows the channel name/logo. Assets pending from the owner; placeholder until then.
6. **Physics answers: show the unit** after the answer field («Ответ: ____ м/с»), like the real form. M4.
