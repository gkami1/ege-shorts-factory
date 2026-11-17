import copy
import json

import pytest

from factory.bank import load_bank
from factory.config import ROOT
from factory.models import Task, VideoScript
from factory.verify.gate import agreement, verify
from factory.verify.review import accept, send_to_review

BANK = load_bank(ROOT / "tasks")


def script(task_id, steps, answer, *, flash, given, target, values=None, sub_tex=None, narration=None,
           symbol=None, unit=None):
    return VideoScript.model_validate({
        "task_id": task_id,
        "hook": {"caption": "Решаем за 5 секунд", "narration": "Решаем за пять секунд."},
        "read": {"given": given, "given_narration": "Дано.", "target": target, "target_narration": "Найти."},
        "steps": [{"tex": t, "sympy": s, "narration": "Шаг."} for t, s in steps],
        "substitute": {"flash": flash, "tex": sub_tex, "narration": "Подставим.", "values": values or {}},
        "answer": {"value": answer, "narration": narration or f"Ответ: {answer}!", "symbol": symbol, "unit": unit},
        "has_short_trick": 0.9,
        "meta": {"title": "t"},
    })


POW_STEPS = [(r"2^{x+3}", "2**(x+3)"), (r"2^{x}\cdot 2^{3}", "2**x * 2**3"), (r"2^{x}\cdot 8", "2**x * 8")]


def pow_script(steps=POW_STEPS, **kw):
    args = dict(flash="$2^x = 10$", given=["$2^x = 10$"], target="$2^{x+3}$", values={"2**x": "10"},
                sub_tex=r"10\cdot 8 = 80", narration="Ответ: восемьдесят!")
    args.update(kw)
    return script("m_profile_07_0001", steps, "80", **args)


def test_handwritten_script_passes_strict():
    from factory.pipeline import load_script

    s = load_script(ROOT / "video_scripts" / "m_profile_07_0001.json")
    rep = verify(s, BANK[s.task_id], strict=True)
    assert rep.ok, rep.text()
    assert any(p.startswith("final") for p in rep.passed)
    assert any(p.startswith("narration") for p in rep.passed)


def test_wrong_rewrite_is_caught():
    steps = [POW_STEPS[0], (r"2^{x} + 2^{3}", "2**x + 2**3"), (r"2^{x}\cdot 8", "2**x * 8")]
    rep = verify(pow_script(steps), BANK["m_profile_07_0001"])
    assert not rep.ok
    assert any("steps 0→1" in e for e in rep.errors)


def test_wrong_final_value_is_caught():
    steps = POW_STEPS[:2] + [(r"2^{x}\cdot 9", "2**x * 9")]
    rep = verify(pow_script(steps), BANK["m_profile_07_0001"])
    assert any(e.startswith("final") for e in rep.errors)


def test_tex_must_match_machine_form():
    steps = POW_STEPS[:2] + [(r"2^{x}\cdot 9", "2**x * 8")]   # shows 9, verifies 8
    rep = verify(pow_script(steps), BANK["m_profile_07_0001"])
    assert any("tex shows" in e for e in rep.errors)


def test_false_calculation_on_sheet_is_caught():
    rep = verify(pow_script(sub_tex=r"10\cdot 8 = 18"), BANK["m_profile_07_0001"])
    assert any(e.startswith("substitute.tex") for e in rep.errors)


def test_narration_must_say_the_answer():
    rep = verify(pow_script(narration="Ответ: восемнадцать!"), BANK["m_profile_07_0001"])
    assert any(e.startswith("narration") for e in rep.errors)


LOG_STEPS = [(r"\log_{2}(x+3) = 5", "log(x+3, 2) = 5"), (r"x+3 = 2^{5}", "x + 3 = 2**5"),
             (r"x+3 = 32", "x + 3 = 32"), (r"x = 29", "x = 29")]


def log_script(steps=LOG_STEPS, answer="29"):
    return script("m_profile_06_0001", steps, answer, flash="$\\log_2(x+3) = 5$", given=["$\\log_2(x+3) = 5$"],
                  target="Найдите корень уравнения", symbol="x", narration=f"Ответ: {answer}.")


def test_equation_solving_chain():
    assert verify(log_script(), BANK["m_profile_06_0001"], strict=True).ok
    bad = LOG_STEPS[:2] + [(r"x+3 = 30", "x + 3 = 30"), (r"x = 27", "x = 27")]
    rep = verify(log_script(bad), BANK["m_profile_06_0001"])
    assert not rep.ok and any("is false" in e for e in rep.errors)


PHYS_STEPS = [(r"5 + 2t + 3t^2", "5 + 2*t + 3*t**2"), (r"v = 2 + 6t", "v = 2 + 6*t"),
              (r"v = 2 + 6\cdot 2 = 14", "v = 2 + 6*2 = 14")]


def phys_script(steps=PHYS_STEPS, unit="м/с"):
    return script("p_01_0001", steps, "14", flash="$t = 2$", given=["$x(t) = 5 + 2t + 3t^2$"],
                  target="Чему равна проекция скорости тела", values={"t": "2"}, symbol="v", unit=unit,
                  narration="Ответ: четырнадцать метров в секунду.")


def test_physics_derivative_and_units():
    rep = verify(phys_script(), BANK["p_01_0001"], strict=True)
    assert rep.ok, rep.text()
    wrong = [PHYS_STEPS[0], (r"v = 2 + 3t", "v = 2 + 3*t"), PHYS_STEPS[2]]
    assert not verify(phys_script(wrong), BANK["p_01_0001"]).ok
    rep = verify(phys_script(unit="км/ч"), BANK["p_01_0001"])
    assert any(e.startswith("units") for e in rep.errors)


def test_strict_mode_requires_machine_forms():
    steps = [(t, None) for t, _ in POW_STEPS]
    s = pow_script(steps)
    assert verify(s, BANK["m_profile_07_0001"]).ok             # lenient: warnings only
    assert not verify(s, BANK["m_profile_07_0001"], strict=True).ok


def test_bank_typo_is_caught():
    t = BANK["m_profile_07_0001"].model_copy(update={"check": "10 * 2**3 + 1"})
    rep = verify(pow_script(), t)
    assert any(e.startswith("bank") for e in rep.errors)


@pytest.mark.parametrize("answers,ok", [
    (["80", "80", "80,0", "81", "80"], True),
    (["80", "81", None], False),
    (["81", "81", "80"], False),
    ([], False),
])
def test_self_consistency(answers, ok):
    assert agreement(answers, "80")[0] is ok


def test_review_queue_roundtrip(tmp_path):
    s = pow_script(narration="Ответ: восемнадцать!")
    rep = verify(s, BANK["m_profile_07_0001"])
    d = send_to_review(s.model_dump_json(indent=1), rep, tmp_path / "review")
    assert (d / "REASONS.md").read_text(encoding="utf-8").count("narration") >= 1
    assert json.loads((d / "report.json").read_text(encoding="utf-8"))["ok"] is False
    (tmp_path / "scripts").mkdir()
    dst = accept("m_profile_07_0001", tmp_path / "review", tmp_path / "scripts")
    assert dst.exists() and not d.exists()


def test_python_style_equality_is_a_chain_and_relations_are_rejected():
    from factory.verify import sym

    assert [str(e) for e in sym.parse_chain("x + 3 == 32")] == ["x + 3", "32"]
    for bad in ("x > 3", "x <= 3", "True"):
        with pytest.raises(sym.ParseError):
            sym.parse_chain(bad)


def test_relational_step_is_reported_not_crashed():
    steps = [(r"2^{x+3} > 0", "2**(x+3) > 0")] + POW_STEPS[1:]
    rep = verify(pow_script(steps), BANK["m_profile_07_0001"], strict=True)
    assert not rep.ok and any("not a numeric expression" in e for e in rep.errors)


def test_segment_names_and_indices_parse_the_same_in_tex_and_sympy():
    from factory.verify import sym

    assert sym.parse_sympy("BC / AB") == sym.parse_tex(r"\frac{BC}{AB}") == sp_ratio("BC", "AB")
    assert sym.parse_sympy("V_2/V_1") == sym.parse_tex(r"\frac{V_{2}}{V_1}")
    assert sym.parse_sympy("sin(A)") == sym.parse_tex(r"\sin A")  # single letters stay single


def sp_ratio(a, b):
    import sympy as sp

    return sp.Symbol(a) / sp.Symbol(b)
