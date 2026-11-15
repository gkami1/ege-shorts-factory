# Recording the narrator dataset (fine-tuning)

Cloning from a 13-second clip copies your timbre and manner, but pronunciation and stress
still come from the base model, which wasn't trained mainly on Russian. Fine-tuning on ~20–30 minutes of your
reading teaches CosyVoice how *you* pronounce the words the videos use most (numbers in every
case, «степени», «показатель», «логарифм», units).

## What to read

`assets/voices/script/reading_script.txt`: 466 numbered lines (~16 min of speech, ~25–30 min at
the microphone). Read the text only, not the numbers in front of each line.

## How

- **Same room and microphone** as your 13-second clip. Quiet, soft furnishings, phone or mic 20–30 cm away, slightly to the side.
- **Same style as the videos:** a lively, clear tutor. The model learns your intonation too.
- Read in order, **one line, then a pause of about 1 second**, then the next line.
- **Mistake?** Stop, pause, and read the whole line again. The last good take wins automatically, and false starts are dropped.
- Split it into **sessions of about 10 minutes**, one file per session (any format). Take breaks: a tired voice ends up in the model.
- Name the files in reading order, e.g. `session_01.m4a`, `session_02.m4a`, … and put them in
  `assets/voices/raw/sessions/`. That folder is gitignored: it never goes to GitHub.

## What happens next (on the pod)

```bash
factory voice-dataset assets/voices/raw/sessions/* --config config/pod.yaml   # → work/voice_dataset
#   prints how many lines were captured, and the numbers of any missing lines to re-record
bash tools/cosyvoice_finetune.sh work/voice_dataset narrator_v1 10            # ~1–2 h on one GPU
COSYVOICE_MODEL_DIR=/workspace/cache/models/narrator_v1 factory build --task-id m_profile_07_0001 --config config/pod.yaml
```

Compare the result against the current clone. If it's better, set `tts.cosyvoice.model_dir` in
`config/pod.yaml`. Before recording, check the training chain end to end on synthetic data with
`tools/finetune_selftest.sh`.
