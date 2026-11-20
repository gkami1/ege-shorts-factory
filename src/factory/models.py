"""Data contracts: the task bank entry and the LLM-produced video script.

Both are pydantic models; `factory schema` exports them as JSON Schema
(`schemas/*.schema.json`). The VideoScript schema is what the local LLM is
constrained to in M3 (vLLM structured outputs / llama.cpp json-schema grammar).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Subject = Literal["math_profile", "math_base", "physics"]

_ID_PREFIX = {"math_profile": "m_profile", "math_base": "m_base", "physics": "p"}
# ЕГЭ part-1 answer blank: an integer or a finite decimal with a comma, optional minus.
# Physics "choose two statements" answers are digit strings ("13") and also match.
ANSWER_RE = r"^-?\d+(,\d+)?$"

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

# --------------------------------------------------------------------------- bank

class Task(_Strict):
    """One exam task in the bank (tasks/**/*.yaml)."""

    id: str = Field(
        pattern=r"^(m_profile|m_base|p)_\d{2}_\d{4}$",
        description="<subject prefix>_<KIM number, 2 digits>_<serial, 4 digits>, e.g. m_profile_07_0001",
    )
    subject: Subject
    exam_part: Literal[1] = 1
    kim_number: int = Field(ge=1, le=30, description="Task number in the КИМ")
    topic: str | None = None
    condition: str = Field(
        min_length=10,
        description="Condition in Russian; inline math as $...$ (LaTeX). Rendered verbatim on the exam sheet.",
    )
    image: str | None = Field(None, description="Optional path to a figure, relative to the task file")
    answer: str = Field(pattern=ANSWER_RE, description="Official answer as written in the blank: '80', '-0,25'")
    answer_unit: str | None = Field(None, description="Physics: unit the answer is expressed in, e.g. 'м/с'")
    check: str | None = Field(
        None,
        description="Optional SymPy expression that evaluates to the answer (machine cross-check of the bank)",
    )
    source: str = Field(description="Where the task comes from (FIPI bank id, book, 'original')")
    tags: list[str] = []
    notes: str | None = None

    @model_validator(mode="after")
    def _id_matches(self) -> "Task":
        prefix = f"{_ID_PREFIX[self.subject]}_{self.kim_number:02d}_"
        if not self.id.startswith(prefix):
            raise ValueError(f"id {self.id!r} must start with {prefix!r} for subject/kim_number")
        if self.condition.count("$") % 2:
            raise ValueError("unbalanced $ in condition")
        return self

    @property
    def answer_decimal(self) -> Decimal:
        return parse_answer(self.answer)

def parse_answer(s: str) -> Decimal:
    """Parse an answer in blank form ('80', '-0,25', '0.25') to Decimal."""
    s = s.strip().replace(" ", "").replace("−", "-").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"not a number: {s!r}") from e

def answers_equal(a: str, b: str) -> bool:
    try:
        return parse_answer(a) == parse_answer(b)
    except ValueError:
        return False

# ------------------------------------------------------------------- video script

_SPAN_DESC = (
    "Exact substring of task.condition. Must not split a $...$ formula: either a whole "
    "$...$ chunk (possibly with surrounding words) or plain text."
)

class Hook(_Strict):
    caption: str = Field(max_length=60, description="Hand-written caption over the sheet, e.g. «Эту задачу из ЕГЭ решают за 5 секунд»")
    narration: str = Field(description="Spoken hook, 1 short sentence")

class Read(_Strict):
    given: list[str] = Field(min_length=1, max_length=3, description="Spans with the given data (yellow highlighter). " + _SPAN_DESC)
    given_narration: str
    target: str = Field(description="Span with the quantity to find (hand-drawn circle). " + _SPAN_DESC)
    target_narration: str

_SYMPY_DESC = (
    "Machine form in SymPy syntax (explicit *, ** for powers, log(a, base), sqrt(), pi; angles in radians). "
    "Either one expression or a chain of equalities separated by ' = '."
)


class Step(_Strict):
    tex: str = Field(description="Display-math LaTeX of the whole expression at this step, no $ delimiters")
    narration: str = Field(description="What the voice says while this step is on screen (1 sentence)")
    sympy: str | None = Field(
        None,
        description=_SYMPY_DESC + " Must be the same maths as `tex`. Consecutive single-expression steps must be "
        "equal; every equality must hold given substitute.values and the answer.",
    )

class Substitute(_Strict):
    flash: str = Field(description="Span of the known value that flashes red on the sheet. " + _SPAN_DESC)
    tex: str | None = Field(None, description="Short handwritten calculation shown on the sheet, e.g. '10\\cdot 8 = 80'")
    narration: str
    values: dict[str, str] = Field(
        default_factory=dict,
        description="Given facts used by the solution, SymPy syntax: {'2**x': '10'}, {'t': '2', 'm': '3'}",
    )

class Answer(_Strict):
    value: str = Field(pattern=ANSWER_RE, description="Final answer in blank form; must equal the official answer")
    narration: str
    symbol: str | None = Field(None, description="The unknown whose value is the answer, if any (e.g. 'x', 'v')")
    unit: str | None = Field(None, description="Physics: unit printed after the answer field; must equal task.answer_unit")

class Outro(_Strict):
    cta: str = "Больше разборов — в профиле"
    narration: str = "Больше разборов — в профиле."

class Meta(_Strict):
    title: str = Field(max_length=100)
    description: str = ""
    hashtags: list[str] = []

class VideoScript(_Strict):
    """Everything the renderer needs. Narration may contain inline $...$ math (spoken via the normalizer)."""

    schema_version: Literal[1] = 1
    task_id: str
    hook: Hook
    read: Read
    steps: list[Step] = Field(min_length=2, max_length=6)
    substitute: Substitute
    answer: Answer
    outro: Outro = Outro()
    has_short_trick: float = Field(ge=0, le=1, description="How 'shorts-worthy' the trick is; <0.5 → skip")
    meta: Meta

# ----------------------------------------------------------- cross-validation

def check_span(condition: str, span: str) -> str | None:
    """Return an error message if `span` cannot be isolated on the rendered sheet."""
    if not span.strip():
        return "empty span"
    idx = condition.find(span)
    if idx < 0:
        if "..." in span or "…" in span:
            return (f"span {span!r} not found in condition: spans are copied character for character, "
                    "never shortened with '...'; use a shorter exact phrase instead")
        return f"span {span!r} not found in condition"
    if span.count("$") % 2:
        return f"span {span!r} splits a $...$ formula"
    if condition[:idx].count("$") % 2:
        return f"span {span!r} starts inside a $...$ formula"
    return None

def expand_ellipsis(condition: str, span: str) -> str:
    """'вероятность того, что ... не будет вопроса' → the exact condition text from the first piece to the last,
    when the pieces occur in it in order (LLMs shorten long spans this way). Otherwise the span unchanged."""
    if condition.find(span) >= 0:
        return span
    pieces = [p.strip() for p in span.replace("…", "...").split("...")]
    if len(pieces) < 2 or not all(pieces):
        return span
    start = condition.find(pieces[0])
    pos = start
    for p in pieces[1:]:
        if pos < 0:
            return span
        pos = condition.find(p, pos + 1)
    if start < 0 or pos < 0:
        return span
    return condition[start:pos + len(pieces[-1])]

_NAMED_VALUE = re.compile(r"\$\s*[A-Za-z][A-Za-z_{}\\]*\s*=\s*")   # '$m=0{,}5$ кг' → '$0{,}5$ кг'


def _fuzzy_span(condition: str, span: str, min_ratio: float = 0.85) -> str | None:
    """The condition phrase (whole words) closest to `span`, if it is a near copy: an inflection changed
    ('скорости' for 'скорость') or a word added/dropped. None when nothing is close enough."""
    import difflib

    starts = [0] + [m.end() for m in re.finditer(r"\s+", condition)]
    ends = [m.start() for m in re.finditer(r"\s+", condition)] + [len(condition)]
    lo, hi = 0.7 * len(span), 1.3 * len(span)
    best, best_r = None, min_ratio
    for a in starts:
        for b in ends:
            if b <= a or not lo <= b - a <= hi:
                continue
            cand = condition[a:b].rstrip(".,;:?!")
            r = difflib.SequenceMatcher(None, span, cand, autojunk=False).ratio()
            if r > best_r and check_span(condition, cand) is None:
                best, best_r = cand, r
    return best


def snap_span(condition: str, span: str) -> str:
    """LLM span → exact condition text: '...' expanded, '$m=0{,}5$' → '$0{,}5$', near copies snapped."""
    if condition.find(span) >= 0:
        return span
    for fix in (lambda s: expand_ellipsis(condition, s), lambda s: _NAMED_VALUE.sub("$", s)):
        s = fix(span)
        if s != span and condition.find(s) >= 0:
            return s
    return _fuzzy_span(condition, span) or span


def fix_spans(script: VideoScript, task: Task) -> None:
    """Deterministic span repairs applied to LLM output before the gate (in place)."""
    c, r = task.condition, script.read
    r.given = [snap_span(c, g) for g in r.given]
    r.target = snap_span(c, r.target)
    # target containing a given ('… в момент $t = 2$ с' with given '$t = 2$'): end the target before it
    t0 = c.find(r.target)
    for g in r.given:
        g0 = c.find(g, t0) if t0 >= 0 else -1
        if g != r.target and t0 >= 0 and t0 < g0 < t0 + len(r.target) and g0 + len(g) <= t0 + len(r.target):
            cut = c[t0:g0].rstrip()
            if cut and check_span(c, cut) is None:
                r.target = cut
    flash = snap_span(c, script.substitute.flash)
    if c.find(flash) < 0 and r.given and c.find(r.given[0]) >= 0:
        flash = r.given[0]   # flashing an intermediate value (not in the condition): flash the given instead
    script.substitute.flash = flash

def script_errors(script: VideoScript, task: Task) -> list[str]:
    """Structural checks of a script against its task (first stage of the verification gate)."""
    errs: list[str] = []
    if script.task_id != task.id:
        errs.append(f"task_id mismatch: {script.task_id} != {task.id}")
    spans = [*script.read.given, script.read.target, script.substitute.flash]
    for s in spans:
        if (e := check_span(task.condition, s)) is not None:
            errs.append(e)
    # spans must be identical or disjoint (each becomes one Tex part)
    ranges = sorted({(task.condition.find(s), task.condition.find(s) + len(s)) for s in spans if task.condition.find(s) >= 0})
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        if b0 < a1:
            errs.append(f"spans overlap: {task.condition[a0:a1]!r} / {task.condition[b0:b1]!r}")
    if not answers_equal(script.answer.value, task.answer):
        errs.append(f"answer {script.answer.value!r} != official {task.answer!r}")
    return errs

def split_condition(condition: str, spans: list[str]) -> tuple[list[str], dict[str, int]]:
    """Split the condition into Tex parts so every span is its own part.

    Returns (parts, {span: part_index}). Assumes `script_errors` passed.
    """
    uniq = sorted(set(spans), key=condition.find)
    parts: list[str] = []
    index: dict[str, int] = {}
    pos = 0
    for s in uniq:
        i = condition.find(s)
        if i > pos:
            parts.append(condition[pos:i])
        index[s] = len(parts)
        parts.append(s)
        pos = i + len(s)
    if pos < len(condition):
        parts.append(condition[pos:])
    return parts, index
