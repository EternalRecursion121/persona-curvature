"""Shared OpenRouter helpers for the persona-LoRA data generation scripts.

The API key is read from disk inside this process only. It is never printed,
logged, or passed on a command line.
"""

import asyncio
import json
import os
import random
import time

import aiohttp

# Chosen from https://openrouter.ai/api/v1/models -- small-to-mid instruct tier,
# cheap, strong instruction following. Change this to swap models.
MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
KEY_PATH = os.path.expanduser("~/.secrets/openrouter-api-key")

_KEY = None


def _key() -> str:
    global _KEY
    if _KEY is None:
        with open(KEY_PATH) as f:
            _KEY = f.read().strip()
    return _KEY


def headers() -> dict:
    return {
        "Authorization": "Bearer " + _key(),
        "Content-Type": "application/json",
        "X-Title": "persona-curvature",
    }


class Usage:
    """Accumulates token usage across a run."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.per_condition = {}

    def add(self, usage: dict, tag: str = "_"):
        if not usage:
            return
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.calls += 1
        d = self.per_condition.setdefault(
            tag, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        d["prompt_tokens"] += p
        d["completion_tokens"] += c
        d["calls"] += 1

    def cost(self, price_in: float, price_out: float) -> float:
        """price_* are per-token USD (as OpenRouter reports them)."""
        return self.prompt_tokens * price_in + self.completion_tokens * price_out

    def to_dict(self, price_in: float, price_out: float) -> dict:
        return {
            "model": MODEL,
            "price_per_token": {"prompt": price_in, "completion": price_out},
            "price_per_million": {
                "prompt": price_in * 1e6,
                "completion": price_out * 1e6,
            },
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.cost(price_in, price_out), 6),
            "per_condition": {
                k: dict(v, estimated_cost_usd=round(
                    v["prompt_tokens"] * price_in
                    + v["completion_tokens"] * price_out, 6))
                for k, v in sorted(self.per_condition.items())
            },
        }


async def get_pricing(session: aiohttp.ClientSession, model_id: str = MODEL):
    """Return (price_in_per_token, price_out_per_token) for model_id."""
    async with session.get(MODELS_URL, headers=headers()) as r:
        r.raise_for_status()
        data = (await r.json())["data"]
    for m in data:
        if m.get("id") == model_id:
            p = m.get("pricing", {})
            return float(p.get("prompt", 0)), float(p.get("completion", 0))
    raise RuntimeError(f"model {model_id} not found on OpenRouter")


class RetryableError(Exception):
    pass


async def chat(
    session: aiohttp.ClientSession,
    messages: list,
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    max_retries: int = 6,
    timeout: int = 180,
    extra: dict | None = None,
):
    """One chat completion. Returns (text, usage_dict). Retries 429/5xx with
    exponential backoff + jitter."""
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        body.update(extra)

    delay = 2.0
    last = None
    for attempt in range(max_retries):
        try:
            async with session.post(
                API_URL,
                headers=headers(),
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as r:
                if r.status == 429 or r.status >= 500:
                    txt = (await r.text())[:200]
                    raise RetryableError(f"HTTP {r.status}: {txt}")
                if r.status >= 400:
                    txt = (await r.text())[:300]
                    raise RuntimeError(f"HTTP {r.status}: {txt}")
                data = await r.json()
            if "error" in data and not data.get("choices"):
                raise RetryableError(str(data["error"])[:200])
            content = (data["choices"][0]["message"].get("content") or "").strip()
            return content, data.get("usage") or {}
        except (RetryableError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = e
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(delay + random.uniform(0, delay * 0.5))
            delay = min(delay * 2, 60)
    raise RuntimeError(f"giving up after {max_retries} attempts: {last}")


def extract_json_array(text: str):
    """Pull a JSON array of strings out of a model reply (tolerates fences/prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        arr = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in arr if isinstance(s, str) and s.strip()]


def fmt_elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m{s % 60:04.1f}s"
