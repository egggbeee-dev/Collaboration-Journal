"""LLM wrappers. Every call returns a parsed JSON dict.

`key` identifies the call (e.g. "offer:R1", "plan:R2", "propose:R3", "graph:1") so the mock
can return scripted answers and the log can show who said what.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class BaseLLM:
    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def complete(self, system: str, user: str, *, key: str, images: list[str] | None = None,
                       retries: int = 1) -> dict:
        last_err = None
        for attempt in range(retries + 1):
            raw = await self._raw(system, user if attempt == 0 else
                                  user + f"\n\nYour previous output was not valid JSON ({last_err}). "
                                         "Return ONLY one JSON object.", key=key, images=images or [])
            self.calls += 1
            try:
                return parse_json(raw)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:120]
        raise ValueError(f"[{key}] invalid JSON after {retries + 1} attempts: {last_err}")

    async def _raw(self, system, user, *, key, images) -> str:
        raise NotImplementedError

    def usage(self) -> dict:
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


def _image_part(path: str, detail: str = "low") -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  [warn] image not found, skipped: {path}")
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}


class OpenAILLM(BaseLLM):
    """gpt-4o style models get `temperature`; reasoning models (gpt-5.x, gpt-6, o-series) get
    `reasoning_effort` instead, because they reject `temperature`."""

    def __init__(self, model: str = "gpt-4o", temperature: float | None = 0.2,
                 reasoning_effort: str | None = "medium", max_concurrency: int = 4,
                 image_detail: str = "low"):
        super().__init__()
        import asyncio
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.is_reasoning = not model.startswith(("gpt-4", "gpt-3"))
        self.sem = asyncio.Semaphore(max_concurrency)
        self.image_detail = image_detail

    async def _raw(self, system, user, *, key, images) -> str:
        content: list[dict] = [{"type": "text", "text": user}]
        content += [part for part in (_image_part(p, self.image_detail) for p in images) if part]
        kw = {}
        if self.is_reasoning and self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        elif not self.is_reasoning and self.temperature is not None:
            kw["temperature"] = self.temperature
        async with self.sem:
            r = await self.client.chat.completions.create(
                model=self.model, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                **kw,
            )
        if r.usage:
            self.prompt_tokens += r.usage.prompt_tokens
            self.completion_tokens += r.usage.completion_tokens
        return r.choices[0].message.content or ""


class MockLLM(BaseLLM):
    """Returns scripted JSON by key. Used to test the pipeline without an API key."""

    def __init__(self, script: dict | str | Path):
        super().__init__()
        if not isinstance(script, dict):
            script = json.loads(Path(script).read_text(encoding="utf-8"))
        self.script = script

    async def _raw(self, system, user, *, key, images) -> str:
        if key not in self.script:
            # graph rounds without a script entry -> no ops
            if key.startswith("graph"):
                return json.dumps({"ops": []})
            raise KeyError(f"MockLLM: no scripted response for '{key}'")
        return json.dumps(self.script[key])
