"""Minimal OpenAI-compatible chat client (vLLM / llama.cpp server). Stdlib only.

vLLM with `--reasoning-parser qwen3` returns the thinking in `reasoning_content` and the answer in
`content`. Thinking is switched off per request with chat_template_kwargs.enable_thinking=False.
Structured output: response_format = json_schema (vLLM: xgrammar; llama.cpp: GBNF from schema).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    content: str
    reasoning: str
    finish_reason: str
    tokens: int


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: dict):
        self.base = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout_s", 1800))
        self.api_key = cfg.get("api_key", "EMPTY")

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:2000]
                if e.code < 500 or attempt == 2:
                    raise LLMError(f"HTTP {e.code}: {msg}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise LLMError(f"cannot reach {self.base}: {e}") from e
            time.sleep(3 * (attempt + 1))
        raise LLMError("unreachable")

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def chat(self, messages: list[dict], *, n: int = 1, thinking: bool = True, json_schema: dict | None = None,
             max_tokens: int = 16384, sampling: dict | None = None, seed: int | None = None) -> list[Completion]:
        body: dict = {"model": self.model, "messages": messages, "n": n, "max_tokens": max_tokens,
                      "chat_template_kwargs": {"enable_thinking": thinking}, **(sampling or {})}
        if seed is not None:
            body["seed"] = seed
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "video_script", "schema": json_schema, "strict": True}}
        data = self._post("/chat/completions", body)
        out = []
        for ch in data["choices"]:
            m = ch["message"]
            out.append(Completion(m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "",
                                  ch.get("finish_reason", ""), 0))
        if out:
            out[0].tokens = int(data.get("usage", {}).get("completion_tokens", 0))
        return out
