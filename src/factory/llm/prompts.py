"""Prompts for the two-pass scriptwriter: (1) solve freely with thinking, (2) fill the VideoScript JSON."""

from __future__ import annotations

import json

from ..models import Task, VideoScript

SUBJECT_RU = {"math_profile": "математика, профильный уровень", "math_base": "математика, базовый уровень",
              "physics": "физика"}

SOLVE_SYSTEM = """You are an expert teacher who prepares students for the Russian ЕГЭ exam (part 1, short answer).
Solve the task carefully. Then write the solution as the SHORTEST correct method a strong student would use:
the kind that fits a 30–40 second video, 2–5 transformations, no brute force, no case analysis unless unavoidable.
Write the solution in Russian. Mention the key trick in one sentence if there is one.
End with a line exactly in this form:
ОТВЕТ: <answer>
where <answer> is written as in the ЕГЭ answer blank: digits only, decimal comma, optional minus, no units, no spaces."""


def solve_messages(task: Task) -> list[dict]:
    unit = f"\nОтвет нужно дать в единицах: {task.answer_unit}." if task.answer_unit else ""
    user = (f"Предмет: {SUBJECT_RU[task.subject]}. Задание №{task.kim_number}.\n\n"
            f"Условие: {task.condition}{unit}")
    return [{"role": "system", "content": SOLVE_SYSTEM}, {"role": "user", "content": user}]


WRITE_SYSTEM = r"""You write scripts for vertical explainer videos (YouTube Shorts / TikTok, 25–45 s) about ЕГЭ tasks.
The video shows the exam sheet with the condition, highlights the given data, circles what to find, morphs the
solution step by step in handwriting, flashes the given value when it is substituted, and shows the answer.
A Russian TTS voice reads the narration; captions are generated from it.

Return ONLY a JSON object matching the schema. Rules:

LANGUAGE AND VOICE
- All narration, captions and meta are in Russian: a friendly tutor talking to «ты», short spoken sentences (≤ 12 words).
- Narration is read by TTS. Write numbers as WORDS whenever grammatical case matters («равно десяти», «из восьми»);
  digits are read in the nominative case only. Avoid abbreviations except «ЕГЭ». Prefer words for formulas
  («два в степени икс плюс три»); simple formulas may be written inside $...$.
- Decimals are said the Russian way: «ноль целых шесть десятых», «ноль целых пятьсот двенадцать тысячных» (or
  digits «0,6», read in the nominative). Never «точка шесть».
- One beat = one sentence or two. Each step's narration explains WHY that transformation is made.

SPANS (they are located on the rendered sheet, so they must be exact)
- read.given (1–3 items), read.target and substitute.flash are EXACT substrings of the condition, character for
  character. A span is either plain text or contains whole $...$ formulas — never starts or ends inside $...$.
- Spans must not overlap (the same span may be reused, e.g. flash may equal a given span). So read.target is the
  short phrase naming what to find («площадь его поверхности»), never a whole sentence that contains a given value.
- A $...$ formula is highlighted whole, with its $ signs, or not at all: you cannot highlight a part of it.
  If the condition is one formula to evaluate («Найдите значение выражения $…$»): read.given = [that whole
  formula with its $ signs], read.target = the words «Найдите значение выражения», substitute.flash = the same
  formula span as in given.

STEPS (2–6)
- tex: display LaTeX of the whole expression at this moment, without $. Each step is a small transformation of the
  previous one (it is animated as a morph), keep the structure stable.
- sympy: the SAME maths in SymPy syntax: explicit *, ** for powers, log(a, base), sqrt(a), pi, angles in radians.
  An equation or a chain of equalities uses a single ' = ' (e.g. "x + 3 = 2**5 = 32"); never '==', '<', '>'.
  * consecutive single-expression steps must be mathematically equal (identity rewrites);
  * every equality must be true given substitute.values and the answer;
  * the last step must evaluate to the answer after substituting substitute.values (or be "x = <answer>"):
    never stop at an intermediate (v**2 = 256 → add the step v = 16; a discriminant → finish to the answer).
- The sympy and tex of a step must describe exactly the same expression, with the same number of ' = ' parts.
- Names (the same in tex and sympy): one letter (a, S, t), a Greek name (nu, rho, alpha), an indexed name with an
  underscore (T_1, v_0; tex T_{1}) or a segment of 2–3 capital letters (AB, BC). Never glue a digit to a letter (T1).
- If a step names the unknown («S = 6*a**2», «N = nu*t», «BC = AB*sin(A)»), set answer.symbol to that name.

SUBSTITUTE, ANSWER
- substitute.flash: the span with the given value used at the end; substitute.tex: a short TRUE arithmetic line
  written on the sheet (e.g. "10\cdot 8 = 80") or null; substitute.values: the given facts in SymPy,
  e.g. {"2**x": "10"} or {"t": "2", "m": "3"}.
- answer.value: exactly the official answer in blank form. answer.narration must say it in words
  («Ответ: восемьдесят!»). answer.symbol: the unknown whose value is the answer ("x", "v") or null.
  answer.unit: the task's unit for physics, else null.

HOOK, META
- hook.caption ≤ 60 characters. «за 5 секунд»-style speed claims ONLY for genuine one-trick tasks
  (has_short_trick ≥ 0.8). Otherwise use a neutral hook: «90% ошибаются вот здесь», «Задача №7: разбор за минуту»,
  «Частая ловушка ЕГЭ».
- has_short_trick: honest score 0–1 of how elegant/short the solution is.
- meta.title ≤ 100 characters, at most one emoji; meta.description 1–2 sentences; meta.hashtags 4–6, including
  "#егэ" and "#shorts".
- outro: keep the default."""


def write_messages(task: Task, solution: str, example: VideoScript | None) -> list[dict]:
    t = {"id": task.id, "subject": SUBJECT_RU[task.subject], "kim_number": task.kim_number,
         "condition": task.condition, "official_answer": task.answer, "answer_unit": task.answer_unit}
    parts = [f"TASK:\n{json.dumps(t, ensure_ascii=False, indent=1)}", f"SOLUTION (verified answer):\n{solution}"]
    if example is not None:
        parts.insert(0, "EXAMPLE of a finished script for another task:\n"
                     + example.model_dump_json(indent=1, exclude_none=False))
    return [{"role": "system", "content": WRITE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def repair_messages(previous: list[dict], bad_json: str, problems: list[str]) -> list[dict]:
    fix = ("The script failed automatic verification:\n- " + "\n- ".join(problems)
           + "\nReturn the corrected FULL JSON object. Keep everything that was correct.")
    return previous + [{"role": "assistant", "content": bad_json}, {"role": "user", "content": fix}]


def script_schema(task: Task) -> dict:
    """VideoScript JSON Schema with $refs inlined (portable across grammar engines), task_id pinned."""
    schema = VideoScript.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            # drop the "title"/"default" annotations, but not fields that happen to be named so (meta.title)
            out = {k: ({p: inline(s) for p, s in v.items()} if k == "properties" else inline(v))
                   for k, v in node.items() if k not in ("title", "default")}
            if "properties" in out:  # every field present (nullable ones may be null): no silent omissions
                out["required"] = sorted(out["properties"])
            return out
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    s = inline(schema)
    s["properties"]["task_id"] = {"type": "string", "const": task.id}
    return s
