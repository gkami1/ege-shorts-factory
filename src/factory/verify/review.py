"""Review queue: scripts that failed the gate wait in review/<task_id>/ for a human.

    review/<id>/script.json   the script as produced (edit it here)
    review/<id>/report.json   machine-readable gate report
    review/<id>/REASONS.md    why it was rejected, in plain words

`factory review-accept <id>` re-runs the gate on the edited script and, if it passes, moves it to
video_scripts/<id>.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .gate import Report


def send_to_review(script_json: str, report: Report, review_dir: Path, extra: dict | None = None) -> Path:
    d = Path(review_dir) / report.task_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.json").write_text(script_json, encoding="utf-8")
    (d / "report.json").write_text(json.dumps({**report.to_dict(), **(extra or {})}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    md = [f"# {report.task_id}: rejected by the verification gate", ""]
    md += ["## Errors", ""] + [f"- {e}" for e in report.errors] + [""]
    if report.warnings:
        md += ["## Warnings", ""] + [f"- {w}" for w in report.warnings] + [""]
    md += ["Fix `script.json` here, then run `factory review-accept " + report.task_id + "`."]
    (d / "REASONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return d


def accept(task_id: str, review_dir: Path, scripts_dir: Path) -> Path:
    src = Path(review_dir) / task_id / "script.json"
    dst = Path(scripts_dir) / f"{task_id}.json"
    shutil.copy(src, dst)
    shutil.rmtree(Path(review_dir) / task_id)
    return dst
