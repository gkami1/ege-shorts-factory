import numpy as np

from factory.speech import qa
from factory.speech.dataset import SR, segment_session

SCRIPT = [("001", "Ответ: восемьдесят."), ("002", "Икс мы не ищем. Смотрим на показатель."), ("003", "Разделим всё на двенадцать.")]


def _hyp(groups):
    """groups: [(start, [words])] → whisper-like (word, start, end) list, 0.3 s per word."""
    out = []
    for t, words in groups:
        for k, w in enumerate(words):
            out.append((w, t + 0.3 * k, t + 0.3 * k + 0.25))
    return out


def test_segment_session_handles_false_starts_and_rereads():
    hyp = _hyp([
        (0.0, ["Ответ:", "80."]),
        (2.0, ["Икс", "мы", "не"]),                      # false start
        (4.0, ["ой,", "сейчас", "заново"]),               # chatter
        (6.0, ["Икс", "мы", "не", "ищем."]),              # reread, with a pause inside the line
        (8.0, ["Смотрим", "на", "показатель."]),
        (10.0, ["Разделим", "всё", "на", "12."]),
    ])
    audio = np.zeros(int(12 * SR), np.float32)
    canon = [qa.canonical(t) for _, t in SCRIPT]
    res, expected = segment_session(audio, hyp, SCRIPT, canon, 0, max_cer=0.12)
    lines = [li for li, *_ in res]
    assert lines == [0, 1, 2] and expected == 3
    _, s, e, c = res[1]
    assert 5.5 <= s <= 6.0 and 8.85 <= e <= 9.3   # the reread spans both pieces, not the false start
    assert c == 0
