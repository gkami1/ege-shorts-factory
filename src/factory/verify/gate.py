"""Verification gate: a script reaches TTS/render only if the maths is machine-checked.

Checks (errors block the build; warnings are reported):
  structure   spans exist and don't split formulas, answer == official            (models.script_errors)
  bank        task.check reproduces the official answer (catches bank typos)
  facts       substitute.values + (answer.symbol = answer) are consistent; they bind the unknowns
  steps       each step's equalities hold; consecutive single-expression steps are equal
              (an identity rewrite, sampled at random points; or equal given the facts)
  final       the last step evaluates to the answer given the facts
  tex         every step's LaTeX is the same maths as its machine form (no showing one thing, checking another)
  substitute  the handwritten calculation on the sheet is arithmetically true
  narration   the answer narration says the answer
  units       physics: answer.unit == task.answer_unit
strict=True (LLM output): missing machine forms are errors, not warnings.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import sympy as sp

from ..models import Task, VideoScript, answers_equal, parse_answer, script_errors
from . import sym


@dataclass
class Report:
    task_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, **asdict(self)}

    def text(self) -> str:
        lines = [f"{self.task_id}: {'PASS' if self.ok else 'FAIL'}"]
        lines += [f"  ✗ {e}" for e in self.errors] + [f"  ! {w}" for w in self.warnings]
        lines += [f"  ✓ {p}" for p in self.passed]
        return "\n".join(lines)


def _unit_norm(u: str) -> str:
    return u.replace(" ", "").replace("^2", "²").replace("^3", "³").replace("·", "*").lower()


def bank_check(task: Task) -> str | None:
    if not task.check:
        return None
    t = sp.Symbol("t")
    try:
        val = sp.nsimplify(sp.sympify(task.check, locals={"t": t}), rational=True)
    except (sp.SympifyError, TypeError) as e:
        return f"bank: check {task.check!r} does not evaluate ({e})"
    if not sym.close(val, sym.number(task.answer)):
        return f"bank: check {task.check!r} = {val}, but the official answer is {task.answer}"
    return None


def verify(script: VideoScript, task: Task, strict: bool = False) -> Report:
    r = Report(task.id)
    soft = r.errors if strict else r.warnings

    errs = script_errors(script, task)
    r.errors += errs
    if not errs:
        r.passed.append("structure: spans and answer match the task")

    if (e := bank_check(task)) is not None:
        r.errors.append(e)
    elif task.check:
        r.passed.append("bank: check reproduces the official answer")

    # facts → numeric bindings
    facts: list[tuple[sp.Expr, sp.Expr]] = []
    for k, v in script.substitute.values.items():
        try:
            facts.append((sym.parse_sympy(k), sym.parse_sympy(v)))
        except sym.ParseError as e:
            r.errors.append(f"substitute.values: {e}")
    answer_val = sym.number(script.answer.value) if _is_number(script.answer.value) else None
    if script.answer.symbol and answer_val is not None:
        facts.append((sp.Symbol(script.answer.symbol), answer_val))
    env, problems = sym.bind_facts(facts)
    r.errors += [f"facts: {p}" for p in problems]

    # steps. A step 'P = <number>' whose P no fact binds introduces a name (P for the probability): bind it, so
    # it isn't sampled at random, and later steps that reuse it are checked against that value.
    env = dict(env)
    chains: list[list[sp.Expr] | None] = []
    shapes: list[tuple[str | None, bool]] = []   # per step: (label shown only in tex, tex ends in '= 0')
    for i, st in enumerate(script.steps):
        if not st.sympy:
            soft.append(f"step {i}: no machine form (sympy); not verified")
            chains.append(None)
            shapes.append((None, False))
            continue
        try:
            ch = sym.parse_chain(st.sympy)
        except sym.ParseError as e:
            r.errors.append(f"step {i}: {e}")
            chains.append(None)
            shapes.append((None, False))
            continue
        chains.append(ch)
        if len(ch) >= 2 and isinstance(ch[0], sp.Symbol) and ch[0] not in env:
            val = ch[1].subs(env)
            if not val.free_symbols:
                env[ch[0]] = val
        for a, b in zip(ch, ch[1:]):
            if sym.equal(a, b, env) is False:
                law = _bind_relation(a, b, env) if task.subject == "physics" else None
                if law:
                    env.update(law)
                    r.warnings.append(f"step {i}: {a} = {b} taken as a law, it gives "
                                      + ", ".join(f"{k} = {v}" for k, v in law.items()) + " (not checked on its own)")
                else:
                    r.errors.append(f"step {i}: {a} = {b} is false" + (" given the facts" if env else ""))
        parts, label, eq = _tex_parts(st.tex, len(ch))
        shapes.append((label, eq))
        if eq and sym.equal(ch[-1], sp.Integer(0), env) is False:
            r.errors.append(f"step {i}: tex says {ch[-1]} = 0, which is false given the facts")
        _check_tex(i, parts, ch, r)
    for i in range(1, len(chains)):
        a, b = chains[i - 1], chains[i]
        # consecutive steps are one value rewritten, unless the tex names different quantities
        # (P(A), then P(not A); y', then y(3)) or a step is an equation '… = 0'
        if a and b and len(a) == 1 and len(b) == 1 and shapes[i - 1][0] == shapes[i][0] \
                and not shapes[i - 1][1] and not shapes[i][1]:
            same = sym.equal(a[0], b[0]) or sym.equal(a[0], b[0], env)
            if same is False:
                r.errors.append(f"steps {i - 1}→{i}: {a[0]} ≠ {b[0]}")
    if all(c is not None for c in chains) and not any(e.startswith("step") for e in r.errors):
        r.passed.append(f"steps: {len(chains)} steps verified")

    # final value
    last = chains[-1] if chains else None
    if last is not None and answer_val is not None:
        v = last[-1].subs(env)
        if v.free_symbols:
            r.errors.append(f"final: last step {last[-1]} still has unknowns {sorted(map(str, v.free_symbols))}")
        elif not sym.close(v, answer_val):
            r.errors.append(f"final: last step gives {sp.N(v, 10)}, answer is {script.answer.value}")
        else:
            r.passed.append("final: last step evaluates to the answer")

    # handwritten calculation on the sheet
    if script.substitute.tex:
        try:
            ch = sym.tex_chain(script.substitute.tex)
            bad = [f"{a} = {b}" for a, b in zip(ch, ch[1:]) if sym.equal(a, b, env) is False]
            if bad:
                r.errors.append(f"substitute.tex: {'; '.join(bad)} is false")
            else:
                r.passed.append("substitute: calculation on the sheet is true")
        except sym.ParseError as e:
            r.warnings.append(f"substitute.tex: {e}")

    # narration says the answer
    from ..speech.qa import canonical

    said = canonical(script.answer.narration)
    want = script.answer.value.replace(".", ",").lstrip("+")
    if want.startswith("-"):
        ok = "минус" in script.answer.narration.lower() and want[1:] in said or want in said
    else:
        ok = want in said
    if ok:
        r.passed.append("narration: the voice says the answer")
    else:
        r.errors.append(f"narration: answer.narration {script.answer.narration!r} doesn't say {script.answer.value}; "
                        f"spell it correctly in words or write digits: 'Ответ: {script.answer.value}!'")

    # decimals read the English way ('точка шесть' for 0,6): wrong Russian, and the normalizer reads digits right
    lines = [script.hook.narration, script.read.given_narration, script.read.target_narration,
             *(st.narration for st in script.steps), script.substitute.narration, script.answer.narration]
    for ln in lines:
        m = next((m for m in _POINT_DECIMAL.finditer(ln) if not _POINT_PLACE.search(ln[:m.start()])), None)
        if m:
            r.errors.append(f"narration: {m.group(0)!r} in {ln!r}: "
                            "say decimals as «ноль целых шесть десятых» or write digits «0,6», never «точка …»")

    # units
    if task.answer_unit:
        if not script.answer.unit:
            soft.append(f"units: task answer is in {task.answer_unit}, script gives no answer.unit")
        elif _unit_norm(script.answer.unit) != _unit_norm(task.answer_unit):
            r.errors.append(f"units: answer.unit {script.answer.unit!r} != task {task.answer_unit!r}")
        else:
            r.passed.append("units: answer unit matches")

    if script.has_short_trick < 0.5:
        r.warnings.append(f"has_short_trick={script.has_short_trick}: probably not shorts-worthy")
    return r


_POINT_DECIMAL = re.compile(r"\bточк[аеиу] (ноль|одна|один|две|два|три|четыре|пять|шесть|семь|восемь|девять)\b",
                            re.IGNORECASE)
# …but «в точке три», «подходящую точку три» name a point x = 3: an adjective or a preposition before it
_POINT_PLACE = re.compile(r"(\b(в|во|к|ко|через|из)|\w+(ую|ой|ая|ей|ом|ем))\s+$", re.IGNORECASE)

# a name shown before the value: P, y', y(3), v_x(t), E_{\text{к}}, P(\text{не вопрос})
_LABEL = re.compile(r"[A-Za-z]\w*(_\{?[\w,]*\}?)?'*(\([^()]*\))?'*")
_TEX_ZERO = re.compile(r"0(\{,\}0+|[.,]0+)?")


def _tex_parts(tex: str, chain_len: int) -> tuple[list[str], str | None, bool]:
    """Raw tex parts aligned with the machine chain → (parts, label, ends_in_zero).

    The tex may carry one part more than the machine form: a leading name ('P = 0{,}8\\cdot 0{,}8' for
    '0.8*0.8'; 'v_x(t) = 2 + 6t') or a trailing '= 0' (the equation the step solves). The name is display
    only; the trailing 0 is checked by the caller."""
    parts = sym.split_chain(tex)
    if len(parts) == chain_len + 1:
        if _TEX_ZERO.fullmatch(parts[-1]):
            return parts[:-1], None, True
        if _LABEL.fullmatch(re.sub(r"\\(text|mathrm)\{[^}]*\}|\s", "", parts[0])):
            return parts[1:], parts[0], False
    return parts, None, False


def _bind_relation(a: sp.Expr, b: sp.Expr, env: dict) -> dict | None:
    """A physics law with a quantity the facts don't give (E_k = m v²/2 with m unlisted; V₁/T₁ = V₂/T₂):
    solve it for one unknown, preferring the left side's, when that gives a single real value or expression."""
    a2, b2 = sp.sympify(a).subs(env), sp.sympify(b).subs(env)
    lhs = sorted(a2.free_symbols, key=str)
    for s in lhs + sorted(b2.free_symbols - set(lhs), key=str):
        try:
            sols = {v for v in sp.solve(sp.Eq(a2, b2), s) if v.is_real is not False}
        except (NotImplementedError, TypeError, ValueError):
            continue
        if len(sols) == 1:
            return {s: sols.pop()}
    return None


def _check_tex(i: int, parts: list[str], chain: list[sp.Expr], r: Report) -> None:
    try:
        tc = [sym.parse_tex(p) for p in parts]
    except sym.ParseError as e:
        r.warnings.append(f"step {i}: tex not machine-readable ({e}); shown formula not cross-checked")
        return
    if len(tc) != len(chain):
        r.errors.append(f"step {i}: tex has {len(tc)} parts, sympy has {len(chain)}")
        return
    names = {str(s) for b in chain for s in b.free_symbols}
    for a, b in zip(tc, chain):
        a = a.subs({s: sp.Symbol(str(s)) for s in a.free_symbols})
        a = sym.fold_function_names(a, names)
        if sym.equal(a, b) is False:
            r.errors.append(f"step {i}: tex shows {a}, sympy says {b}")


def _is_number(s: str) -> bool:
    try:
        parse_answer(s)
        return True
    except ValueError:
        return False


def agreement(answers: list[str | None], official: str, min_share: float = 0.6) -> tuple[bool, str | None, dict]:
    """Self-consistency over N independent solutions: the majority must equal the official answer.

    Returns (ok, majority answer, counts). None answers (failed samples) count against agreement.
    """
    norm = [str(parse_answer(a)) if a and _is_number(a) else None for a in answers]
    counts = Counter(norm)
    if not answers:
        return False, None, {}
    maj, n = counts.most_common(1)[0]
    ok = maj is not None and answers_equal(maj, official) and n / len(answers) >= min_share
    return ok, maj, {str(k): v for k, v in counts.items()}
