"""Data contracts: the task bank entry and the LLM-produced video script.

Both are pydantic models; `factory schema` exports them as JSON Schema
(`schemas/*.schema.json`). The VideoScript schema is what the local LLM is
constrained to in M3 (vLLM structured outputs / llama.cpp json-schema grammar).
"""

from __future__ import annotations

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

class Step(_Strict):
    tex: str = Field(description="Display-math LaTeX of the whole expression at this step, no $ delimiters")
    narration: str = Field(description="What the voice says while this step is on screen (1 sentence)")

class Substitute(_Strict):
    flash: str = Field(description="Span of the known value that flashes red on the sheet. " + _SPAN_DESC)
    tex: str | None = Field(None, description="Short handwritten calculation shown on the sheet, e.g. '10\\cdot 8 = 80'")
    narration: str

class Answer(_Strict):
    value: str = Field(pattern=ANSWER_RE, description="Final answer in blank form; must equal the official answer")
    narration: str

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
        return f"span {span!r} not found in condition"
    if span.count("$") % 2:
        return f"span {span!r} splits a $...$ formula"
    if condition[:idx].count("$") % 2:
        return f"span {span!r} starts inside a $...$ formula"
    return None

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
