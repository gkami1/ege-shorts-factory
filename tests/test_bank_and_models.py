from pathlib import Path

import pytest
import sympy as sp

from factory.bank import load_bank
from factory.config import ROOT
from factory.models import (
    VideoScript,
    answers_equal,
    check_span,
    parse_answer,
    script_errors,
    split_condition,
)

BANK = load_bank(ROOT / "tasks")


def test_bank_loads_and_has_both_subjects():
    subjects = {t.subject for t in BANK.values()}
    assert {"math_profile", "physics"} <= subjects


@pytest.mark.parametrize("task", [t for t in BANK.values() if t.check], ids=lambda t: t.id)
def test_bank_answers_match_independent_check(task):
    """Every seed task carries a SymPy expression that must reproduce the official answer."""
    t = sp.Symbol("t")
    val = sp.nsimplify(sp.sympify(task.check, locals={"t": t}))
    assert sp.Rational(str(task.answer_decimal)) == val, f"{task.id}: {val} != {task.answer}"


@pytest.mark.parametrize(
    "a,b,eq",
    [("80", "80", True), ("0,6", "0.6", True), ("0,60", "0,6", True), ("-0,25", "−0,25", True), ("8", "80", False), ("x", "1", False)],
)
def test_answers_equal(a, b, eq):
    assert answers_equal(a, b) is eq


def test_parse_answer_rejects_garbage():
    with pytest.raises(ValueError):
        parse_answer("восемь")


COND = "Найдите значение выражения $2^{x+3}$, если $2^x = 10$."


@pytest.mark.parametrize(
    "span,ok",
    [
        ("$2^x = 10$", True),
        ("$2^{x+3}$", True),
        ("если $2^x = 10$", True),
        ("Найдите значение", True),
        ("$2^x", False),          # splits a formula
        ("2^x = 10", False),      # starts inside $...$
        ("$3^x$", False),         # absent
    ],
)
def test_check_span(span, ok):
    assert (check_span(COND, span) is None) is ok


def test_split_condition_isolates_spans_and_roundtrips():
    parts, idx = split_condition(COND, ["$2^x = 10$", "$2^{x+3}$", "$2^x = 10$"])
    assert "".join(parts) == COND
    assert parts[idx["$2^x = 10$"]] == "$2^x = 10$"
    assert parts[idx["$2^{x+3}$"]] == "$2^{x+3}$"
    assert all(p.count("$") % 2 == 0 for p in parts)


def test_sample_script_is_valid():
    s = VideoScript.model_validate_json((ROOT / "video_scripts" / "m_profile_07_0001.json").read_text(encoding="utf-8"))
    assert script_errors(s, BANK[s.task_id]) == []


def test_script_with_wrong_answer_is_rejected():
    raw = (ROOT / "video_scripts" / "m_profile_07_0001.json").read_text(encoding="utf-8")
    s = VideoScript.model_validate_json(raw)
    s.answer.value = "800"
    assert any("official" in e for e in script_errors(s, BANK[s.task_id]))


def test_task_id_must_match_subject_and_number():
    from factory.models import Task

    with pytest.raises(ValueError):
        Task(id="p_07_0001", subject="math_profile", kim_number=7, condition="Найдите что-нибудь.", answer="1", source="x")
