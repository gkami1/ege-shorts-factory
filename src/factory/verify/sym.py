"""SymPy helpers for the verification gate: parse machine forms and LaTeX, compare numerically.

Equality is checked numerically at sample points rather than by `simplify(a - b) == 0`:
it is fast, never hangs, and handles logs/roots/powers uniformly. Unknowns that the facts
don't pin down are sampled in [0.7, 2.3] (positive, away from 0 and 1) over several trials,
which catches any wrong rewrite with overwhelming probability.
"""

from __future__ import annotations

import random
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCALS = {"e": sp.E, "pi": sp.pi, "ln": sp.log, "lg": lambda x: sp.log(x, 10), "tg": sp.tan, "ctg": sp.cot,
           "N": sp.Symbol("N"), "S": sp.Symbol("S"), "E": sp.Symbol("E"), "I": sp.Symbol("I"),
           "Q": sp.Symbol("Q"), "O": sp.Symbol("O"), "beta": sp.Symbol("beta"), "gamma": sp.Symbol("gamma")}
_REL = re.compile(r"(?<![<>=!])=(?!=)")


class ParseError(ValueError):
    pass


def split_chain(s: str) -> list[str]:
    """'a = b = c' → ['a', 'b', 'c'] (ignores '==', '<=', '>=')."""
    return [p.strip() for p in _REL.split(s) if p.strip()]


def parse_sympy(s: str) -> sp.Expr:
    try:
        return parse_expr(s.replace("−", "-").replace("·", "*"), local_dict=dict(_LOCALS),
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception as e:  # sympy raises many types here
        raise ParseError(f"cannot parse {s!r}: {e}") from e


def parse_chain(s: str) -> list[sp.Expr]:
    return [parse_sympy(p) for p in split_chain(s)]


def _clean_tex(t: str) -> str:
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("{,}", ".")
    t = re.sub(r"\\(left|right|displaystyle)\b", "", t)
    t = re.sub(r"\\[,;!]|\\quad|\\qquad", " ", t)
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)
    return t.replace("−", "-")


def parse_tex(t: str) -> sp.Expr:
    from sympy.parsing.latex import parse_latex

    try:
        e = parse_latex(_clean_tex(t), backend="lark")
    except Exception as ex:
        raise ParseError(f"cannot parse LaTeX {t!r}: {ex}") from ex
    if not isinstance(e, sp.Basic):  # lark ambiguity tree
        raise ParseError(f"ambiguous LaTeX {t!r}")
    return e


def tex_chain(t: str) -> list[sp.Expr]:
    return [parse_tex(p) for p in split_chain(t)]


def number(s: str) -> sp.Expr:
    return sp.nsimplify(s.strip().replace("−", "-").replace(",", "."), rational=True)


def bind_facts(facts: list[tuple[sp.Expr, sp.Expr]]) -> tuple[dict[sp.Symbol, sp.Expr], list[str]]:
    """Solve the facts for numeric symbol values, one unknown at a time. Returns (env, problems)."""
    env: dict[sp.Symbol, sp.Expr] = {}
    problems: list[str] = []
    pending = list(facts)
    progress = True
    while pending and progress:
        progress = False
        for lhs, rhs in list(pending):
            l2, r2 = lhs.subs(env), rhs.subs(env)
            free = (l2.free_symbols | r2.free_symbols)
            if not free:
                if not close(l2, r2):
                    problems.append(f"fact {lhs} = {rhs} is false given the others")
                pending.remove((lhs, rhs))
                progress = True
            elif len(free) == 1:
                (sym,) = free
                if isinstance(lhs, sp.Symbol) and lhs == sym:
                    sols = [r2]
                else:
                    try:
                        sols = [s for s in sp.solve(sp.Eq(l2, r2), sym) if s.is_real is not False]
                    except NotImplementedError:
                        sols = []
                    if not sols:
                        try:
                            sols = [sp.nsolve(l2 - r2, sym, 1.5)]
                        except Exception:
                            sols = []
                if not sols:
                    problems.append(f"cannot solve fact {lhs} = {rhs} for {sym}")
                else:
                    env[sym] = sols[0]
                pending.remove((lhs, rhs))
                progress = True
    for lhs, rhs in pending:
        problems.append(f"fact {lhs} = {rhs} has several unknowns")
    return env, problems


def _num(e: sp.Expr) -> complex | None:
    try:
        v = complex(sp.N(e, 30))
    except (TypeError, ValueError):
        return None
    if v != v or abs(v) == float("inf"):
        return None
    return v


def close(a: sp.Expr, b: sp.Expr, tol: float = 1e-9) -> bool:
    va, vb = _num(a), _num(b)
    if va is None or vb is None:
        return False
    return abs(va - vb) <= tol * max(1.0, abs(va), abs(vb))


def equal(a: sp.Expr, b: sp.Expr, env: dict | None = None, trials: int = 4, seed: int = 0) -> bool | None:
    """True/False if a == b at sample points (env binds known symbols), None if never evaluable."""
    env = env or {}
    a, b = sp.sympify(a).subs(env), sp.sympify(b).subs(env)
    free = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(seed)
    seen = 0
    for _ in range(trials * 3):
        pt = {s: sp.Rational(rng.randint(70, 230), 100) for s in free}
        va, vb = _num(a.subs(pt)), _num(b.subs(pt))
        if va is None or vb is None:
            continue
        seen += 1
        if abs(va - vb) > 1e-9 * max(1.0, abs(va), abs(vb)):
            return False
        if seen >= trials or not free:
            return True
    return True if seen else None
