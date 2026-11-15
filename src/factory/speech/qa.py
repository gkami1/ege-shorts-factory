"""Speech QA: does the clip say what the script says?

Both sides are canonicalized through the normalizer, so «восемьдесят» (our text) and «80»
(whisper's transcript) compare equal, and «икс» matches whisper's «x».
"""

from __future__ import annotations

import re

from . import numbers as nb
from .normalize import LETTERS, tokenize


def canonical(text: str) -> list[str]:
    out = []
    for t in tokenize(text):
        for piece in t.display.split():
            w = re.sub(r"[^\w,/]", "", piece.lower().replace("ё", "е")).strip(",")
            if not w:
                continue
            out.append(LETTERS.get(w, w) if re.fullmatch(r"[a-z]", w) else "икс" if w == "х" else w)
    return out


def canonical_hyp(words: list[str]) -> list[str]:
    """Whisper words (may mix «80» and «восемьдесят») → canonical tokens."""
    return canonical(" ".join(words))


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def merge_number_words(hyp: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Group whisper's number words into one item with digits («двадцать пять» → «25»)."""
    words = [h[0].strip() for h in hyp]
    out, i = [], 0
    while i < len(hyp):
        got = nb.parse_words(words, i, words[i - 1] if i else "")
        if got:
            disp, n = got
            out.append((disp, hyp[i][1], hyp[i + n - 1][2]))
            i += n
        else:
            out.append(hyp[i])
            i += 1
    return out


def cer(ref: list[str], hyp: list[str]) -> float:
    """Character error rate on the canonical tokens joined without spaces.

    Robust to whisper splitting/merging words («Да но» vs «дано»); still catches mumbles,
    skipped words and wrong numbers. This is the QA metric; wer() is kept for reports.
    """
    return wer(list("".join(ref)), list("".join(hyp)))


def score(spoken: str, hyp_words: list[str]) -> float:
    return cer(canonical(spoken), canonical_hyp(hyp_words))
