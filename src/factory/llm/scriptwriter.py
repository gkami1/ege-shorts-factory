"""Task → verified VideoScript with a local LLM.

  1. solve   N independent samples with thinking; parse «ОТВЕТ: …»; the majority must equal the
             official answer (self-consistency: if the model can't solve it reliably, its explanation
             can't be trusted either)
  2. write   one JSON-schema-constrained call (no thinking) turns the best agreeing solution into a
             VideoScript, with the hand-written script as a style example
  3. verify  the gate (strict); on failure the errors go back to the model for up to `repairs` rounds
  4. result  pass → video_scripts/<id>.json; fail → review/<id>/ with the reasons
Every request/response is logged in work/llm/<id>/ for debugging.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..models import Task, VideoScript, answers_equal
from ..verify.gate import agreement, verify
from ..verify.review import send_to_review
from . import prompts
from .client import LLMClient

_ANS = re.compile(r"ОТВЕТ\s*:\s*\**\s*([-−]?\d+(?:[.,]\d+)?)", re.IGNORECASE)


def parse_answer_line(text: str) -> str | None:
    m = _ANS.findall(text or "")
    return m[-1].replace("−", "-").replace(".", ",") if m else None


@dataclass
class Outcome:
    task_id: str
    status: str                     # written | review | unsolved | error
    path: str | None = None
    answers: list = field(default_factory=list)
    rounds: int = 0
    errors: list = field(default_factory=list)
    seconds: float = 0.0


class ScriptWriter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lc = cfg["llm"]
        self.client = LLMClient(self.lc)
        self.paths = cfg["paths"]
        ex = Path(self.paths["scripts"]) / f"{self.lc.get('example_script', 'm_profile_07_0001')}.json"
        self.example = VideoScript.model_validate_json(ex.read_text(encoding="utf-8")) if ex.exists() else None

    def _log(self, task_id: str, name: str, obj) -> None:
        d = Path(self.paths["work"]) / "llm" / task_id
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    def solve(self, task: Task, n: int | None = None) -> tuple[list[str | None], list[str]]:
        n = n or int(self.lc.get("samples", 5))
        comps = self.client.chat(prompts.solve_messages(task), n=n, thinking=True,
                                 max_tokens=int(self.lc.get("solve_max_tokens", 16384)),
                                 sampling=self.lc.get("solve_sampling"))
        self._log(task.id, "solve.json", [{"answer": parse_answer_line(c.content), "content": c.content,
                                           "reasoning": c.reasoning, "finish": c.finish_reason} for c in comps])
        return [parse_answer_line(c.content) for c in comps], [c.content for c in comps]

    def write(self, task: Task, force: bool = False) -> Outcome:
        t0 = time.perf_counter()
        out = Outcome(task.id, "error")
        dst = Path(self.paths["scripts"]) / f"{task.id}.json"
        if dst.exists() and not force:
            out.status, out.path = "exists", str(dst)
            return out
        answers, sols = self.solve(task)
        out.answers = answers
        ok, maj, counts = agreement(answers, task.answer, float(self.lc.get("min_agreement", 0.6)))
        if not ok:
            out.status = "unsolved"
            out.errors = [f"self-consistency: answers {counts}, official {task.answer}"]
            self._to_review(task, None, out.errors, out)
            out.seconds = time.perf_counter() - t0
            return out
        good = [s for a, s in zip(answers, sols) if a and answers_equal(a, task.answer)]
        solution = min(good, key=len)  # the shortest correct write-up makes the tightest video

        msgs = prompts.write_messages(task, solution, self.example)
        schema = prompts.script_schema(task)
        raw, problems = "", []
        for rnd in range(1 + int(self.lc.get("repairs", 2))):
            out.rounds = rnd + 1
            if rnd:
                msgs = prompts.repair_messages(msgs, raw, problems)
            comp = self.client.chat(msgs, thinking=bool(self.lc.get("write_thinking", False)), json_schema=schema,
                                    max_tokens=int(self.lc.get("write_max_tokens", 8192)),
                                    sampling=self.lc.get("write_sampling"))[0]
            raw = comp.content.strip()
            self._log(task.id, f"write_{rnd}.json", raw)
            try:
                script = VideoScript.model_validate_json(raw)
            except ValidationError as e:
                problems = [f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors()][:12]
                continue
            rep = verify(script, task, strict=True)
            if rep.ok:
                dst.write_text(script.model_dump_json(indent=2) + "\n", encoding="utf-8")
                out.status, out.path, out.errors = "written", str(dst), []
                out.seconds = time.perf_counter() - t0
                return out
            problems = rep.errors
        out.status, out.errors = "review", problems
        self._to_review(task, raw, problems, out)
        out.seconds = time.perf_counter() - t0
        return out

    def _to_review(self, task: Task, raw: str | None, problems: list[str], out: Outcome) -> None:
        from ..verify.gate import Report

        rep = Report(task.id, errors=list(problems))
        d = send_to_review(raw or "{}", rep, Path(self.paths["review"]),
                           extra={"answers": out.answers, "rounds": out.rounds, "model": self.lc["model"]})
        out.path = str(d)
