import numpy as np
import soundfile as sf

from factory.captions.align import Aligner, _fill_gaps
from factory.captions.ass import _ts, build_ass, chunk_words
from factory.timeline import Segment, layout

TIMING = {"pad_before": 0.1, "pad_after": 0.2, "min_segment": 1.2, "outro_hold": 1.0}


def test_layout_is_contiguous_and_frame_aligned():
    segs = [Segment("hook", "a", audio_dur=2.03), Segment("given", "b", audio_dur=0.4), Segment("outro", "c", audio_dur=1.0)]
    layout(segs, TIMING, fps=30)
    assert segs[0].start == 0
    for a, b in zip(segs, segs[1:]):
        assert abs(a.end - b.start) < 1e-6
    for s in segs:
        assert abs(round(s.dur * 30) - s.dur * 30) < 1e-6
    assert segs[1].dur >= 1.2  # min_segment
    assert segs[2].dur >= 1.0 + 0.3 + 1.0  # outro hold


def test_chunking_respects_limits_and_punctuation():
    words = [{"w": w, "start": i, "end": i + 0.5} for i, w in enumerate("Дано: два в степени икс равно десяти.".split())]
    chunks = chunk_words(words, max_words=3, max_chars=20)
    assert [len(c) for c in chunks][0] == 1  # "Дано:" ends a chunk
    assert all(len(c) <= 3 for c in chunks)
    assert sum(len(c) for c in chunks) == len(words)


def test_ass_timestamp_format():
    assert _ts(0) == "0:00:00.00"
    assert _ts(61.257) == "0:01:01.26"


def test_fill_gaps_monotonic():
    out = _fill_gaps(["а", "бэ", "вэ", "гэ"], [(0.1, 0.3), None, None, (1.0, 1.2)], 0.0, 1.5)
    starts = [a for a, _ in out]
    assert starts == sorted(starts)
    assert 0.3 <= out[1][0] < out[2][0] < 1.0


def test_proportional_alignment_on_synthetic_clip(tmp_path):
    sr = 16000
    a = np.zeros(sr * 2, np.float32)
    a[int(0.5 * sr) : int(1.5 * sr)] = 0.3 * np.sin(np.arange(sr) / 5)
    p = tmp_path / "x.wav"
    sf.write(p, a, sr)
    words = Aligner({"aligner": "proportional"}).align(str(p), "раз два три")
    assert [w["w"] for w in words] == ["раз", "два", "три"]
    assert 0.45 <= words[0]["start"] <= 0.55
    assert 1.45 <= words[-1]["end"] <= 1.55


def test_build_ass_has_box_and_text(tmp_path):
    from factory.render.style import palette

    seg = Segment("hook", "Эту задачу решают", start=0, dur=2)
    seg.words = [{"w": w, "start": 0.2 + i * 0.4, "end": 0.5 + i * 0.4} for i, w in enumerate(seg.text.split())]
    cfg = {
        "font": "DejaVu Sans", "font_file": _any_font(), "size": 70, "y": 1390,
        "pad_x": 30, "pad_y": 16, "radius": 22, "max_words": 3, "max_chars": 20,
    }
    ass = build_ass([seg], cfg, palette("classic"), 1080, 1920)
    assert "\\p1" in ass and ass.count("Dialogue: 1,") == 3


def _any_font() -> str:
    from pathlib import Path

    for p in ["/usr/share/fonts/truetype/factory/Montserrat-ExtraBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(p).exists():
            return p
    import pytest

    pytest.skip("no font available")
