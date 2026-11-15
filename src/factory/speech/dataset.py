"""Long reading sessions → one clip per script line (fine-tuning data for the narrator).

The reader reads assets/voices/script/reading_script.txt in order, pausing ~1 s between lines.
Mistakes are fine: pause and read the line again; the LAST good take of a line wins, and false
starts / chatter that match no line are dropped.

Algorithm: whisper (word timestamps) → split the transcript at pauses → greedily match 1–3
consecutive pieces to a script line near the expected position (CER on canonical tokens, so
«80» and «восемьдесят» compare equal) → cut the audio with padding into the silence around it.
Output (kaldi-style, what CosyVoice's recipe expects):
    <out>/wavs/<utt>.wav, wav.scp, text, utt2spk, spk2utt, instruct, report.json
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import qa

SR = 24000
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


@dataclass
class Piece:
    words: list[str]
    start: float
    end: float


def load_script(path: Path) -> list[tuple[str, str]]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            lid, text = ln.split("\t", 1)
            out.append((lid.strip(), text.strip()))
    return out


def _to_wav(src: Path, dst: Path) -> np.ndarray:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
                    "-af", "highpass=f=70", str(dst)], check=True)
    a, _ = sf.read(dst, dtype="float32")
    return a


def _pieces(hyp: list[tuple[str, float, float]], gap: float) -> list[Piece]:
    out: list[Piece] = []
    for w, s, e in hyp:
        if out and s - out[-1].end < gap:
            out[-1].words.append(w)
            out[-1].end = e
        else:
            out.append(Piece([w], s, e))
    return out


def segment_session(audio: np.ndarray, hyp, script, line_canon, expected: int, max_cer: float,
                    gap: float = 0.55, window: tuple[int, int] = (5, 25)):
    """Yield (line_index, start_s, end_s, cer) for each matched take, in order."""
    pcs = _pieces(hyp, gap)
    i, matches = 0, []
    while i < len(pcs):
        best = None
        for k in (1, 2, 3):
            if i + k > len(pcs):
                break
            words = [w for p in pcs[i:i + k] for w in p.words]
            hc = qa.canonical_hyp(words)
            lo, hi = max(0, expected - window[0]), min(len(script), expected + window[1])
            for li in range(lo, hi):
                c = qa.cer(line_canon[li], hc)
                if best is None or c < best[0] - 1e-9:
                    best = (c, li, k)
        if best and best[0] <= max_cer:
            c, li, k = best
            matches.append((li, pcs[i].start, pcs[i + k - 1].end, c, i, i + k))
            expected = li + 1
            i += k
        else:
            i += 1
    # pad each cut into the surrounding silence (up to half the gap to the neighbouring speech)
    total = len(audio) / SR
    res = []
    for li, s, e, c, a, b in matches:
        prev_end = pcs[a - 1].end if a > 0 else 0.0
        next_start = pcs[b].start if b < len(pcs) else total
        s2 = max(s - 0.25, (prev_end + s) / 2)
        e2 = min(e + 0.35, (e + next_start) / 2)
        res.append((li, max(0.0, s2), min(total, e2), c))
    return res, expected


def build_dataset(sessions: list[Path], script_path: Path, out: Path, cfg: dict, speaker: str = "narrator",
                  max_cer: float = 0.12) -> dict:
    from ..captions.align import Aligner

    script = load_script(script_path)
    line_canon = [qa.canonical(t) for _, t in script]
    aligner = Aligner({**cfg["captions"], "aligner": "whisper"})
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    takes: dict[int, tuple[np.ndarray, float, str]] = {}  # line → (audio, cer, session); last take wins
    expected = 0
    for sess in sessions:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "s.wav"
            audio = _to_wav(sess, wav)
            hyp = aligner.transcribe(str(wav))
        if not hyp:
            raise RuntimeError(f"whisper heard nothing in {sess}")
        matched, expected = segment_session(audio, hyp, script, line_canon, expected, max_cer)
        for li, s, e, c in matched:
            takes[li] = (audio[int(s * SR): int(e * SR)], c, sess.name)
        print(f"[dataset] {sess.name}: {len(matched)} takes, now at line {expected}/{len(script)}", flush=True)

    utts = []
    for li in sorted(takes):
        a, c, sess = takes[li]
        lid, text = script[li]
        utt = f"{speaker}_{lid}"
        p = (out / "wavs" / f"{utt}.wav").resolve()
        sf.write(p, a, SR, subtype="PCM_16")
        utts.append((utt, str(p), text, round(c, 3), sess, len(a) / SR))
    w = lambda name, rows: (out / name).write_text("".join(r + "\n" for r in rows), encoding="utf-8")  # noqa: E731
    w("wav.scp", [f"{u} {p}" for u, p, *_ in utts])
    w("text", [f"{u} {t}" for u, _, t, *_ in utts])
    w("utt2spk", [f"{u} {speaker}" for u, *_ in utts])
    w("spk2utt", [f"{speaker} " + " ".join(u for u, *_ in utts)])
    w("instruct", [f"{u} {INSTRUCT}" for u, *_ in utts])
    missing = [script[i][0] for i in range(len(script)) if i not in takes]
    report = {
        "lines": len(script), "clips": len(utts), "minutes": round(sum(u[5] for u in utts) / 60, 1),
        "missing_lines": missing,
        "clips_detail": [{"utt": u, "cer": c, "session": s, "dur": round(d, 2)} for u, _, _, c, s, d in utts],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
