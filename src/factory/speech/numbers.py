"""Russian numbers in both directions.

* spell(): digits → words (nominative, gender-aware), decimals, simple fractions, units with
  plural agreement («2 с» → «две секунды»).
* parse_words(): number words in any case → digits («десяти» → 10, «ноль целых двадцать пять
  сотых» → «0,25», «три четвёртых» → «3/4»). Used for captions (digits on screen while the voice
  says words) and for comparing TTS output with whisper transcripts.
"""

from __future__ import annotations

import re
from functools import lru_cache

from num2words import num2words

CASES = ("nominative", "genitive", "dative", "accusative", "instrumental", "prepositional")
_BASES = list(range(0, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100))


def norm(w: str) -> str:
    """Lowercase, ё→е, strip stress marks and punctuation."""
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е").replace("+", "").replace("́", ""))


def _n2w(n: int, **kw) -> str:
    try:
        return num2words(n, lang="ru", **kw)
    except (TypeError, NotImplementedError, ValueError):
        return num2words(n, lang="ru")


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, tuple[str, int]]:
    """word form → (kind, value). kind: unit | teen | ten | hundred | thousand | million."""
    lex: dict[str, tuple[str, int]] = {}
    for n in _BASES:
        kind = "unit" if n < 10 else "teen" if n < 20 else "ten" if n < 100 else "hundred"
        for c in CASES:
            for g in ("m", "f", "n"):
                form = _n2w(n, case=c, gender=g)
                if " " not in form:
                    lex.setdefault(norm(form), (kind, n))
    for stem_forms, kind, val in (
        ("тысяча тысячи тысяче тысячу тысячей тысячею тысяч тысячам тысячами тысячах", "thousand", 1000),
        ("миллион миллиона миллиону миллионом миллионе миллионы миллионов миллионам миллионами миллионах", "million", 10**6),
    ):
        for f in stem_forms.split():
            lex[f] = (kind, val)
    for f in _ONE_FORMS:
        lex.setdefault(f, ("unit", 1))
    return lex


_ONE_FORMS = {"один", "одна", "одно", "одну", "одного", "одной", "одному", "одним", "одном", "одною", "одними", "одних", "одни"}

_INT_PART = re.compile(r"^цел(ая|ой|ую|ых|ым|ыми|ые)$")
_ADJ = r"(ая|ой|ую|ых|ым|ыми|ые)$"
_DEC_DEN = {r"^десят" + _ADJ: 1, r"^сот" + _ADJ: 2, r"^тысячн" + _ADJ: 3}
_ORD_DEN = {
    2: r"^втор(ая|ой|ую|ых|ым|ыми|ые)$", 3: r"^трет(ья|ьей|ью|ьих|ьим|ьими|ьи)$",
    4: r"^четверт(ая|ой|ую|ых|ым|ыми|ые)$", 5: r"^пят(ая|ой|ую|ых|ым|ыми|ые)$",
    6: r"^шест(ая|ой|ую|ых|ым|ыми|ые)$", 7: r"^седьм(ая|ой|ую|ых|ым|ыми|ые)$",
    8: r"^восьм(ая|ой|ую|ых|ым|ыми|ые)$", 9: r"^девят(ая|ой|ую|ых|ым|ыми|ые)$",
}


def _parse_int(words: list[str], i: int) -> tuple[int, int] | None:
    """Greedy cardinal parse starting at words[i] (normalized). Returns (value, n_words)."""
    lex = _lexicon()
    total, group, j = 0, 0, i
    last_rank = 10**9  # rank order inside a group: hundred(3) > ten(2) > unit/teen(1)
    rank = {"hundred": 3, "ten": 2, "teen": 1, "unit": 1}
    seen = False
    while j < len(words):
        e = lex.get(words[j])
        if e is None:
            break
        kind, v = e
        if kind in ("thousand", "million"):
            if not seen and group == 0:
                group = 1  # «тысяча» alone
            total += max(group, 1) * v
            group, last_rank, seen = 0, 10**9, True
            j += 1
            continue
        r = rank[kind]
        if r >= last_rank or (v == 0 and seen):
            break
        if kind == "teen" and last_rank == 2:  # «двадцать двенадцать» is two numbers
            break
        group += v
        last_rank = 1 if kind == "teen" else r
        seen = True
        j += 1
        if v == 0:
            break
    if not seen:
        return None
    return total + group, j - i


def parse_words(words: list[str], i: int, prev: str = "") -> tuple[str, int] | None:
    """Parse a number expressed in words at words[i] (raw tokens). Returns (display, n_tokens).

    Punctuation on a token ends the number (the parse never crosses «пять, два»).
    """
    nw: list[str] = []
    for w in words[i:]:
        nw.append(norm(w))
        if re.search(r"[^\w+́]$", w):  # trailing punctuation stops lookahead
            break
    got = _parse_int(nw, 0)
    if not got:
        return None
    val, n = got
    # decimal: «две целых пять десятых»
    if n < len(nw) and _INT_PART.match(nw[n]):
        frac = _parse_int(nw, n + 1)
        if frac and n + 1 + frac[1] < len(nw):
            den = nw[n + 1 + frac[1]]
            for pat, k in _DEC_DEN.items():
                if re.match(pat, den) and frac[0] < 10**k:
                    return f"{val},{frac[0]:0{k}d}", n + 2 + frac[1]
    # simple fraction: «три четвёртых», «одна вторая»
    if n < len(nw) and val < 100:
        for d, pat in _ORD_DEN.items():
            if re.match(pat, nw[n]):
                return f"{val}/{d}", n + 1
        for pat, k in _DEC_DEN.items():
            if re.match(pat, nw[n]):
                return f"{val}/{10**k}", n + 1
    # «один» alone is usually not a number to show as a digit («один из способов»)
    if n == 1 and nw[0] in _ONE_FORMS and not prev.endswith(":"):
        return None
    return str(val), n


# ── digits → words ────────────────────────────────────────────────────────────────

# abbreviation → (gender, one, few, many, tail). few = genitive singular (also used after decimals).
UNITS: dict[str, tuple[str, str, str, str, str]] = {
    "м/с²": ("m", "метр", "метра", "метров", "в секунду в квадрате"),
    "м/с^2": ("m", "метр", "метра", "метров", "в секунду в квадрате"),
    "м/с": ("m", "метр", "метра", "метров", "в секунду"),
    "км/ч": ("m", "километр", "километра", "километров", "в час"),
    "км": ("m", "километр", "километра", "километров", ""),
    "см": ("m", "сантиметр", "сантиметра", "сантиметров", ""),
    "мм": ("m", "миллиметр", "миллиметра", "миллиметров", ""),
    "м²": ("m", "квадратный метр", "квадратных метра", "квадратных метров", ""),
    "м³": ("m", "кубический метр", "кубических метра", "кубических метров", ""),
    "м": ("m", "метр", "метра", "метров", ""),
    "с": ("f", "секунда", "секунды", "секунд", ""),
    "мин": ("f", "минута", "минуты", "минут", ""),
    "ч": ("m", "час", "часа", "часов", ""),
    "кг": ("m", "килограмм", "килограмма", "килограммов", ""),
    "г": ("m", "грамм", "грамма", "граммов", ""),
    "т": ("f", "тонна", "тонны", "тонн", ""),
    "Н": ("m", "ньютон", "ньютона", "ньютонов", ""),
    "кДж": ("m", "килоджоуль", "килоджоуля", "килоджоулей", ""),
    "Дж": ("m", "джоуль", "джоуля", "джоулей", ""),
    "кВт": ("m", "киловатт", "киловатта", "киловатт", ""),
    "Вт": ("m", "ватт", "ватта", "ватт", ""),
    "кПа": ("m", "килопаскаль", "килопаскаля", "килопаскалей", ""),
    "Па": ("m", "паскаль", "паскаля", "паскалей", ""),
    "мА": ("m", "миллиампер", "миллиампера", "миллиампер", ""),
    "А": ("m", "ампер", "ампера", "ампер", ""),
    "кВ": ("m", "киловольт", "киловольта", "киловольт", ""),
    "В": ("m", "вольт", "вольта", "вольт", ""),
    "кОм": ("m", "килоом", "килоома", "килоом", ""),
    "Ом": ("m", "ом", "ома", "ом", ""),
    "мкКл": ("m", "микрокулон", "микрокулона", "микрокулонов", ""),
    "нКл": ("m", "нанокулон", "нанокулона", "нанокулонов", ""),
    "Кл": ("m", "кулон", "кулона", "кулонов", ""),
    "Гц": ("m", "герц", "герца", "герц", ""),
    "Тл": ("f", "тесла", "теслы", "тесла", ""),
    "мкФ": ("f", "микрофарада", "микрофарады", "микрофарад", ""),
    "Ф": ("f", "фарада", "фарады", "фарад", ""),
    "Гн": ("m", "генри", "генри", "генри", ""),
    "эВ": ("m", "электронвольт", "электронвольта", "электронвольт", ""),
    "дптр": ("f", "диоптрия", "диоптрии", "диоптрий", ""),
    "моль": ("m", "моль", "моля", "моль", ""),
    "л": ("m", "литр", "литра", "литров", ""),
    "К": ("m", "кельвин", "кельвина", "кельвинов", ""),
    "°C": ("m", "градус Цельсия", "градуса Цельсия", "градусов Цельсия", ""),
    "°С": ("m", "градус Цельсия", "градуса Цельсия", "градусов Цельсия", ""),
    "°": ("m", "градус", "градуса", "градусов", ""),
    "%": ("m", "процент", "процента", "процентов", ""),
    "руб.": ("m", "рубль", "рубля", "рублей", ""),
    "руб": ("m", "рубль", "рубля", "рублей", ""),
}
UNIT_RE = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))


def plural_form(n: int) -> int:
    """0 → one, 1 → few, 2 → many (Russian plural rule for the noun after a cardinal)."""
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


def spell_int(n: int, gender: str = "m") -> str:
    s = "минус " if n < 0 else ""
    words = _n2w(abs(n), gender=gender)
    if words.startswith("одна тысяча"):  # spoken Russian: «тысяча двести», not «одна тысяча двести»
        words = words[len("одна "):]
    return s + words


_ORD_FEM = {2: ("вторая", "вторых"), 3: ("третья", "третьих"), 4: ("четвёртая", "четвёртых"), 5: ("пятая", "пятых"),
            6: ("шестая", "шестых"), 7: ("седьмая", "седьмых"), 8: ("восьмая", "восьмых"), 9: ("девятая", "девятых"),
            10: ("десятая", "десятых"), 100: ("сотая", "сотых"), 1000: ("тысячная", "тысячных")}


def spell_fraction(num: int, den: int) -> str | None:
    """«3/4» → «три четвёртых». None if the denominator has no common ordinal name."""
    if den not in _ORD_FEM:
        return None
    one = num % 10 == 1 and num % 100 != 11
    return f"{spell_int(num, 'f')} {_ORD_FEM[den][0 if one else 1]}"


def spell_number(s: str, gender: str = "m") -> tuple[str, bool]:
    """Spell «80», «-0,25», «2.5». Returns (words, is_fractional)."""
    s = s.replace("−", "-").replace(".", ",")
    neg = s.startswith("-")
    s = s.lstrip("-+")
    if "," in s:
        whole, frac = s.split(",", 1)
        k = len(frac)
        den = {1: 10, 2: 100, 3: 1000}.get(k)
        w = int(whole or 0)
        f = int(frac)
        if den:
            words = f"{spell_int(w, 'f')} {'целая' if plural_form(w) == 0 else 'целых'} {spell_fraction(f, den)}"
        else:
            words = f"{spell_int(w, 'f')} целых и {' '.join(spell_int(int(c)) for c in frac)}"
        return ("минус " if neg else "") + words, True
    return ("минус " if neg else "") + spell_int(int(s), gender), False


def spell_with_unit(num: str, unit: str) -> str:
    g, one, few, many, tail = UNITS[unit]
    words, frac = spell_number(num, g)
    if frac:
        noun = few
    else:
        noun = (one, few, many)[plural_form(abs(int(num.replace("−", "-"))))]
    return " ".join(x for x in (words, noun, tail) if x)
