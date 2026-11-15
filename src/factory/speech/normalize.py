"""Narration → (caption tokens, spoken Russian).

Every narration is turned into a list of Tokens. A Token pairs what the caption shows with
what the voice says:

    «Ответ: восемьдесят!»     → [Ответ:|Ответ:] [80!|восемьдесят!]
    «Дано: $2^x = 10$.»       → [Дано:|Дано:] [2|два] [в степени|в степени] [x|икс] [=|равно] [10.|десять.]
    «$t = 2$ с»               → … [2 с|две секунды]

Captions show digits and symbols; TTS gets words (with optional '+' stress marks from the
lexicon). The aligner times each spoken word and a caption token spans its spoken words.

Rules of thumb for script writers (and the M3 LLM prompt):
  * Numbers whose case matters («равно десяти», «из восьми») — write them as words.
    Digits are read in the nominative.
  * Formulas go inside $…$ and are read by `MathReader`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from . import numbers as nb

LEXICON_PATH = Path(__file__).resolve().parents[3] / "config" / "lexicon_ru.yaml"


@dataclass
class Token:
    display: str        # caption text ("" = not shown)
    spoken: str         # words sent to TTS ("" = silent)
    num: str | None = None  # digit string if this is a bare number (units may attach)

    @property
    def words(self) -> list[str]:
        """Spoken words that carry sound (used for alignment)."""
        return [w for w in self.spoken.split() if re.search(r"\w", w)]


# ── lexicon (stress marks, spoken replacements) ───────────────────────────────────


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> dict:
    p = Path(path) if path else LEXICON_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    stress = {k.lower().replace("ё", "е"): v for k, v in (data.get("stress") or {}).items()}
    spoken = dict(data.get("spoken") or {})
    return {"stress": stress, "spoken": spoken}


def _apply_stress(word: str, stress: dict) -> str:
    m = re.match(r"^(\W*)([\w+\u0301-]+)(\W*)$", word)
    if not m or "+" in word:
        return word
    lead, core, trail = m.groups()
    rep = stress.get(core.lower().replace("ё", "е"))
    if not rep:
        return word
    if core[0].isupper():
        rep = rep[0].upper() + rep[1:] if rep[0] != "+" else "+" + rep[1].upper() + rep[2:]
    return lead + rep + trail


# ── LaTeX reader ──────────────────────────────────────────────────────────────────

LETTERS = {
    "a": "а", "b": "бэ", "c": "цэ", "d": "дэ", "e": "е", "f": "эф", "g": "же", "h": "аш", "i": "и",
    "j": "йот", "k": "ка", "l": "эль", "m": "эм", "n": "эн", "o": "о", "p": "пэ", "q": "ку", "r": "эр",
    "s": "эс", "t": "тэ", "u": "у", "v": "вэ", "w": "дубль-вэ", "x": "икс", "y": "игрек", "z": "зет",
}
GREEK = {
    "alpha": "альфа", "beta": "бета", "gamma": "гамма", "delta": "дельта", "Delta": "дельта",
    "epsilon": "эпсилон", "varepsilon": "эпсилон", "eta": "эта", "theta": "тэта", "lambda": "лямбда",
    "mu": "мю", "nu": "ню", "pi": "пи", "rho": "ро", "sigma": "сигма", "tau": "тау", "varphi": "фи",
    "phi": "фи", "omega": "омега", "Omega": "омега",
}
FUNCS = {"sin": "синус", "cos": "косинус", "tg": "тангенс", "tan": "тангенс", "ctg": "котангенс",
         "cot": "котангенс", "arcsin": "арксинус", "arccos": "арккосинус", "arctg": "арктангенс"}
OPS = {
    "+": ("+", "плюс"), "-": ("−", "минус"), "−": ("−", "минус"), "=": ("=", "равно"),
    "<": ("<", "меньше"), ">": (">", "больше"), "cdot": ("·", "умножить на"), "times": ("·", "умножить на"),
    "div": (":", "разделить на"), ":": (":", "разделить на"), "le": ("≤", "меньше или равно"),
    "leq": ("≤", "меньше или равно"), "ge": ("≥", "больше или равно"), "geq": ("≥", "больше или равно"),
    "ne": ("≠", "не равно"), "neq": ("≠", "не равно"), "approx": ("≈", "примерно равно"),
    "pm": ("±", "плюс-минус"), "infty": ("∞", "бесконечность"), "Rightarrow": ("⇒", "следовательно"),
}
IGNORE = {"left", "right", ",", ";", "!", "quad", "qquad", "displaystyle", " ", "big", "Big", "limits"}
_LTX = re.compile(
    r"\\[A-Za-z]+|\\.|\d+(?:(?:\{,\}|[.,])\d+)?|[A-Za-z]|[А-Яа-яЁё]+|\S"
)


class MathReader:
    def __init__(self, src: str):
        self.t = _LTX.findall(src)
        self.i = 0

    def peek(self, k: int = 0) -> str | None:
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def next(self) -> str:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def group(self) -> list[Token]:
        """One argument: {…} or a single atom."""
        if self.peek() == "{":
            self.next()
            return self.seq("}")
        return self.atom() or []

    def raw_group(self) -> str:
        if self.peek() != "{":
            return self.next() if self.peek() else ""
        depth, out = 0, []
        while self.peek() is not None:
            c = self.next()
            if c == "{":
                depth += 1
                if depth == 1:
                    continue
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c)
        return "".join(out)

    def seq(self, stop: str | None = None) -> list[Token]:
        out: list[Token] = []
        while (c := self.peek()) is not None:
            if c == stop:
                self.next()
                break
            if c in ("^", "_"):
                self.next()
                self.script(out, c)
                continue
            out += self.atom()
        return out

    def script(self, out: list[Token], kind: str) -> None:
        if kind == "^" and self.peek() == "\\circ":
            self.next()
            out.append(Token("°", "", None))
            return
        arg = self.group()
        if kind == "_":  # v_0 → «вэ ноль», shown as v0
            if out:
                out[-1] = Token(out[-1].display + "".join(t.display for t in arg),
                                " ".join([out[-1].spoken] + [t.spoken for t in arg]))
            return
        disp = "".join(t.display for t in arg)
        if disp == "2":
            out.append(Token("в квадрате", "в квадрате"))
        elif disp == "3":
            out.append(Token("в кубе", "в кубе"))
        else:
            out.append(Token("в степени", "в степени"))
            out += arg

    def atom(self) -> list[Token]:
        c = self.next()
        if c in ("{",):
            return self.seq("}")
        if re.fullmatch(r"\d+(?:(?:\{,\}|[.,])\d+)?", c):
            num = c.replace("{,}", ",").replace(".", ",")
            return [Token(num, nb.spell_number(num)[0], num)]
        if re.fullmatch(r"[A-Za-z]", c):
            name = LETTERS.get(c.lower(), c)
            if self.peek() == "(":  # f(x) → «эф от икс»
                save = self.i
                self.next()
                inner = self.seq(")")
                if len(inner) == 1:
                    return [Token(f"{c}({inner[0].display})", f"{name} от {inner[0].spoken}")]
                self.i = save
            return [Token(c, name)]
        if c[0] == "\\":
            cmd = c[1:]
            if cmd in IGNORE:
                return []
            if cmd in ("text", "mathrm", "textrm", "mbox", "operatorname"):
                return tokenize_text(self.raw_group(), math=False)
            if cmd in ("frac", "dfrac", "tfrac"):
                a, b = self.raw_group(), self.raw_group()
                if re.fullmatch(r"-?\d+", a) and re.fullmatch(r"\d+", b) and (w := nb.spell_fraction(int(a), int(b))):
                    return [Token(f"{a}/{b}", w)]
                ta, tb = MathReader(a).seq(), MathReader(b).seq()
                return _paren(ta) + [Token("/", "делённое на")] + _paren(tb)
            if cmd == "sqrt":
                deg = ""
                if self.peek() == "[":
                    self.next()
                    while self.peek() not in (None, "]"):
                        deg += self.next()
                    self.next()
                arg = MathReader(self.raw_group()).seq()
                head = {"": "корень из", "3": "кубический корень из"}.get(deg, f"корень степени {deg} из")
                return [Token(head, head)] + _genitive(arg)
            if cmd in ("log", "lg", "ln"):
                base = ""
                if cmd == "log" and self.peek() == "_":
                    self.next()
                    base = self.raw_group()
                if self.peek() == "(":
                    self.next()
                    arg = self.seq(")")
                else:
                    arg = self.group()
                name = {"log": "логарифм", "lg": "десятичный логарифм", "ln": "натуральный логарифм"}[cmd]
                out = [Token(name, name)] + _genitive(arg)
                if base:
                    out += [Token("по основанию", "по основанию")] + MathReader(base).seq()
                return out
            if cmd in FUNCS:
                return [Token(cmd, FUNCS[cmd])]
            if cmd in GREEK:
                return [Token(GREEK[cmd], GREEK[cmd])]
            if cmd in OPS:
                return [Token(*OPS[cmd])]
            if cmd == "%":
                return [Token("%", "")]
            return []  # unknown command: silent
        if c in OPS:
            return [Token(*OPS[c])]
        if c in "()[]|":
            return [Token(c, "")]
        if c == ",":
            return [Token(",", ",")]
        if c == "'":
            return [Token("'", "штрих")]
        if c == "%":
            return [Token("%", "")]
        if re.fullmatch(r"[А-Яа-яЁё]+", c):
            return [Token(c, c)]
        return []


def _paren(toks: list[Token]) -> list[Token]:
    if len(toks) > 1:
        return [Token("(", "")] + toks + [Token(")", "")]
    return toks


def _genitive(toks: list[Token]) -> list[Token]:
    """«корень из шестнадцати», «логарифм восьми»: a bare integer argument goes to the genitive."""
    toks = [t for t in toks if t.display not in "()"] if len(toks) == 3 else toks
    if len(toks) == 1 and toks[0].num and re.fullmatch(r"\d+", toks[0].num):
        n = int(toks[0].num)
        return [Token(toks[0].display, nb._n2w(n, case="genitive"))]
    return _paren(toks)


# ── plain text ────────────────────────────────────────────────────────────────────

_DIGITS = re.compile(r"^([(«\"„]*)(№\s?)?(-?\d+(?:[.,]\d+)?(?:/\d+)?)(.*)$")
_TRAIL = re.compile(r"[.,!?:;…»\")]+$")
_UNIT_EXACT = re.compile(rf"^({nb.UNIT_RE})$")


def _split_trail(w: str) -> tuple[str, str]:
    m = _TRAIL.search(w)
    return (w[: m.start()], m.group(0)) if m else (w, "")


def tokenize_text(text: str, math: bool = False) -> list[Token]:
    raw = text.split()
    out: list[Token] = []
    i = 0
    while i < len(raw):
        w = raw[i]
        if w in ("—", "–", "-"):
            out.append(Token("—", ","))
            i += 1
            continue
        m = _DIGITS.match(w)
        if m:
            lead, no, num, rest = m.groups()
            core_rest, trail = _split_trail(rest)
            if "/" in num:
                a, b = num.split("/")
                sp = nb.spell_fraction(int(a), int(b)) or f"{nb.spell_int(int(a))} делённое на {nb.spell_int(int(b))}"
                out.append(Token(w, lead + sp + trail))
            elif core_rest and _UNIT_EXACT.match(core_rest):
                out.append(Token(w, lead + nb.spell_with_unit(num, core_rest) + trail))
            elif not core_rest:
                sp = nb.spell_number(num)[0]
                if no:
                    sp = "номер " + sp
                out.append(Token(w, lead + sp + trail, None if (no or trail or lead) else num))
            else:
                out.append(Token(w, w))
            i += 1
            continue
        prev = raw[i - 1] if i else ""
        got = nb.parse_words(raw, i, prev)
        if got:
            disp, n = got
            first, last = raw[i], raw[i + n - 1]
            lead = re.match(r"^[(«\"„]*", first).group(0)
            trail = _split_trail(last)[1]
            out.append(Token(lead + disp + trail, " ".join(raw[i : i + n])))
            i += n
            continue
        core, trail = _split_trail(w)
        if re.fullmatch(r"[A-Za-z]{1,3}", core):  # \u00ab\u043e\u0441\u044c Ox\u00bb \u2192 \u00ab\u043e\u0441\u044c \u043e \u0438\u043a\u0441\u00bb
            out.append(Token(w, " ".join(LETTERS.get(ch.lower(), ch) for ch in core) + trail))
        else:
            out.append(Token(w.replace("+", "").replace("\u0301", ""), w))
        i += 1
    return out


def _attach_units(toks: list[Token]) -> list[Token]:
    out: list[Token] = []
    for t in toks:
        prev = out[-1] if out else None
        core, trail = _split_trail(t.display)
        if prev is not None and prev.num and t.spoken in ("", t.display) and _UNIT_EXACT.match(core):
            out[-1] = Token(f"{prev.display} {t.display}" if core not in ("%", "°") else prev.display + t.display,
                            nb.spell_with_unit(prev.num, core) + trail)
            continue
        out.append(t)
    return out


def _merge_silent(toks: list[Token]) -> list[Token]:
    """Brackets and other display-only tokens glue onto their neighbours."""
    out: list[Token] = []
    pending = ""
    for t in toks:
        if not re.search(r"\w", t.spoken):
            if t.display in ("(", "[") or not out:
                pending += t.display
            else:
                out[-1].display += (" " + t.display) if t.display == "—" else t.display
                out[-1].spoken += t.spoken
                out[-1].num = None
            continue
        if pending:
            t = Token(pending + t.display, t.spoken, t.num)
            pending = ""
        out.append(t)
    if pending and out:
        out[-1].display += pending
    return out


def tokenize(text: str, lexicon: dict | None = None) -> list[Token]:
    lex = lexicon or load_lexicon()
    for k, v in lex["spoken"].items():
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", lambda _m, v=v, k=k: f"\x00{k}\x01{v}\x02", text)
    toks: list[Token] = []
    for part in re.split(r"(\$[^$]*\$)", text):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            toks += MathReader(part[1:-1]).seq()
            continue
        # punctuation glued after a formula («$x$.») belongs to the previous token
        lead = re.match(r"^[.,!?:;…]+", part)
        if lead and toks:
            toks[-1].display += lead.group(0)
            toks[-1].spoken += lead.group(0)
            toks[-1].num = None
            part = part[lead.end():]
        for chunk in re.split(r"(\x00[^\x01]*\x01[^\x02]*\x02)", part):
            m = re.fullmatch(r"\x00([^\x01]*)\x01([^\x02]*)\x02", chunk)
            if m:
                toks.append(Token(m.group(1), m.group(2)))
            else:
                toks += tokenize_text(chunk)
    toks = _merge_silent(_attach_units(toks))
    for t in toks:
        t.spoken = " ".join(_apply_stress(w, lex["stress"]) for w in t.spoken.split())
    return [t for t in toks if t.display or t.spoken]


def to_speech(text: str) -> str:
    """Text for the TTS engine (may contain '+' stress marks; engines convert or strip them)."""
    s = " ".join(t.spoken for t in tokenize(text) if t.spoken)
    s = re.sub(r"\s+([,.!?:;])", r"\1", s)
    s = re.sub(r",+", ",", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def display_words(text: str) -> list[str]:
    """Caption tokens (digits, symbols) in order."""
    return [t.display for t in tokenize(text) if t.display]


def stress_format(text: str, mode: str) -> str:
    """'+' marks → engine format: plus (Silero) | acute (U+0301 after the vowel) | strip."""
    if mode == "plus":
        return text
    if mode == "acute":
        return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1\u0301", text).replace("+", "")
    return text.replace("+", "")
