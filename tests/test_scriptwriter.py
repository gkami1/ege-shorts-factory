import json
import shutil

import pytest

from factory.bank import load_bank
from factory.config import ROOT, load_config
from factory.llm import prompts
from factory.llm.client import Completion
from factory.llm.scriptwriter import ScriptWriter, parse_answer_line

BANK = load_bank(ROOT / "tasks")
GOOD = (ROOT / "video_scripts" / "m_profile_07_0001.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("text,ans", [
    ("…\nОТВЕТ: 80", "80"), ("ответ: -0.25", "-0,25"), ("ОТВЕТ: **14**", "14"),
    ("сначала ОТВЕТ: 7, потом пересчитал. ОТВЕТ: 9", "9"), ("нет ответа", None),
])
def test_parse_answer_line(text, ans):
    assert parse_answer_line(text) == ans


def test_schema_is_inlined_and_pins_task_id():
    s = prompts.script_schema(BANK["m_profile_07_0001"])
    dumped = json.dumps(s)
    assert "$ref" not in dumped and "$defs" not in dumped
    assert s["properties"]["task_id"]["const"] == "m_profile_07_0001"
    step = s["properties"]["steps"]["items"]
    assert set(step["required"]) == {"tex", "narration", "sympy"}


def test_schema_keeps_every_model_field():
    """Fields named like JSON-Schema annotations (meta.title) must survive inlining (2026-11-17 pod run:
    the grammar forbade meta.title, so every LLM script failed validation)."""
    from factory.models import VideoScript

    s = prompts.script_schema(BANK["m_profile_07_0001"])
    meta = s["properties"]["meta"]
    assert "title" in meta["properties"] and "title" in meta["required"]
    full = VideoScript.model_json_schema()

    def check(ours, theirs):
        theirs = full["$defs"][theirs["$ref"].split("/")[-1]] if "$ref" in theirs else theirs
        if "properties" in theirs:
            assert set(ours["properties"]) == set(theirs["properties"])
            for k in theirs["properties"]:
                check(ours["properties"][k], theirs["properties"][k])

    check(s, full)


class FakeClient:
    """Scripted responses: solve answers, then a queue of JSON drafts."""

    def __init__(self, answers, drafts):
        self.answers, self.drafts, self.calls = answers, list(drafts), []

    def ready(self):
        return True

    def chat(self, messages, *, n=1, thinking=True, json_schema=None, **kw):
        self.calls.append({"n": n, "thinking": thinking, "schema": json_schema is not None, "messages": messages})
        if json_schema is None:
            return [Completion(f"Решение.\nОТВЕТ: {a}", "думаю", "stop", 0) for a in self.answers[:n]]
        return [Completion(self.drafts.pop(0), "", "stop", 0)]


@pytest.fixture
def writer(tmp_path):
    cfg = load_config()
    for k in ("scripts", "work", "review"):
        cfg["paths"][k] = tmp_path / k
        cfg["paths"][k].mkdir()
    shutil.copy(ROOT / "video_scripts" / "m_profile_07_0001.json", cfg["paths"]["scripts"] / "example.json")
    cfg["llm"]["example_script"] = "example"
    return ScriptWriter(cfg)


def test_happy_path_writes_verified_script(writer):
    writer.client = FakeClient(["80", "80", "80", "81", "80"], [GOOD])
    o = writer.write(BANK["m_profile_07_0001"])
    assert o.status == "written" and o.rounds == 1
    assert writer.client.calls[0]["thinking"] and not writer.client.calls[1]["thinking"]
    assert writer.client.calls[1]["schema"]


def test_verifier_errors_go_back_for_repair(writer):
    bad = GOOD.replace("Ответ: восемьдесят!", "Ответ: восемнадцать!")
    writer.client = FakeClient(["80"] * 5, [bad, GOOD])
    o = writer.write(BANK["m_profile_07_0001"])
    assert o.status == "written" and o.rounds == 2
    repair = writer.client.calls[2]["messages"][-1]["content"]
    assert "narration" in repair and "verification" in repair


def test_invalid_json_is_repaired_too(writer):
    writer.client = FakeClient(["80"] * 5, ['{"task_id": "m_profile_07_0001"}', GOOD])
    assert writer.write(BANK["m_profile_07_0001"]).status == "written"


def test_unreliable_solver_goes_to_review(writer, tmp_path):
    writer.client = FakeClient(["80", "81", "18", None, "80"], [])
    o = writer.write(BANK["m_profile_07_0001"])
    assert o.status == "unsolved"
    assert (tmp_path / "review" / "m_profile_07_0001" / "REASONS.md").exists()


def test_persistent_failure_goes_to_review(writer, tmp_path):
    bad = GOOD.replace("Ответ: восемьдесят!", "Ответ: восемнадцать!")
    writer.client = FakeClient(["80"] * 5, [bad, bad, bad])
    o = writer.write(BANK["m_profile_07_0001"])
    assert o.status == "review" and o.rounds == 3
    assert "восемнадцать" in (tmp_path / "review" / "m_profile_07_0001" / "script.json").read_text(encoding="utf-8")
