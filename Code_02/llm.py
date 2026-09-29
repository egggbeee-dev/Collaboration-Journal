"""LLM clients. The pipeline only depends on `complete_json`, so the real GPT-4o client
and the scripted mock client are interchangeable.

`tag` (e.g. "offer:agent_2", "graph:reasoner") is used for logging, per-stage call counts
(coordination-overhead metric) and for looking up canned answers in the mock.
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from collections import Counter
from pathlib import Path


class BaseLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls_by_stage: Counter[str] = Counter()

    def _record(self, tag: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls_by_stage[tag.split(":")[0]] += 1

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls_by_stage": dict(self.calls_by_stage),
        }

    async def complete_json(self, *, tag: str, system: str, user: str, images: list[str] | None = None) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- real client
def _image_part(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class OpenAIClient(BaseLLM):
    def __init__(self, model: str = "gpt-4o", max_concurrency: int = 4, temperature: float = 0.2, max_retries: int = 5):
        super().__init__()
        from openai import AsyncOpenAI  # imported lazily so mock runs need no API key

        self._client = AsyncOpenAI()
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)  # guards against rate limits

    async def complete_json(self, *, tag: str, system: str, user: str, images: list[str] | None = None) -> dict:
        import openai

        content: list[dict] = [{"type": "text", "text": user}]
        content += [_image_part(p) for p in (images or [])]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]

        for attempt in range(self._max_retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=self._temperature,
                    )
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError):
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)  # backoff outside the semaphore

        usage = resp.usage
        self._record(tag, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        return json.loads(resp.choices[0].message.content)  # JSONDecodeError is a ValueError -> retried upstream


# --------------------------------------------------------------------------- mock client
class ScriptedClient(BaseLLM):
    """Returns canned JSON by tag. A value may be a list: answers are then consumed in
    order and the last one repeats (handy for 'first answer invalid, second valid' tests)."""

    def __init__(self, script: dict[str, dict | list[dict]], latency: float = 0.01):
        super().__init__()
        self._script = {k: (list(v) if isinstance(v, list) else [v]) for k, v in script.items()}
        self._latency = latency

    async def complete_json(self, *, tag: str, system: str, user: str, images: list[str] | None = None) -> dict:
        await asyncio.sleep(self._latency)  # yield so agents genuinely interleave
        answers = self._script.get(tag)
        if not answers:
            raise KeyError(f"no scripted answer left for tag {tag!r}")
        ans = answers.pop(0) if len(answers) > 1 else answers[0]
        self._record(tag)
        return json.loads(json.dumps(ans))  # deep copy
