"""Providers and targets.

A *provider* is a model. A *target* is the thing being evaluated, which is often
not a bare model but a whole service — so `HttpTarget` exists to point this
harness at a running RAG API or agent and score what it actually returns. That
is the difference between evaluating a prompt and evaluating a system.

Responses are cached on disk by content hash. Without caching, iterating on a
scorer means re-paying for every generation, which is the main reason people
stop running their evals.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

PRICES: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ProviderError(RuntimeError):
    pass


@dataclass
class Call:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    error: str = ""


class LLM(Protocol):
    name: str
    model: str

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str: ...


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


@dataclass
class ResponseCache:
    directory: Path
    enabled: bool = True
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        payload = json.dumps(
            {"model": model, "messages": list(messages), "kw": {k: kw[k] for k in sorted(kw)}},
            sort_keys=True,
            default=str,
        )
        return hashlib.blake2b(payload.encode(), digest_size=20).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        (self.directory / f"{key}.json").write_text(
            json.dumps(value, default=str), encoding="utf-8"
        )

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def _retry(client: Any, method: str, url: str, *, tries: int, **kw: Any) -> Any:
    import httpx

    last: Exception | None = None
    for attempt in range(tries):
        try:
            resp = client.request(method, url, **kw)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code != 429 and resp.status_code < 500:
                raise ProviderError(f"{resp.status_code}: {resp.text[:300]}")
            wait = resp.headers.get("retry-after")
            if wait:
                try:
                    time.sleep(min(float(wait), 30.0))
                    continue
                except ValueError:
                    pass
            last = ProviderError(f"{resp.status_code}: {resp.text[:200]}")
        if attempt < tries - 1:
            time.sleep(min(2**attempt + random.random(), 20.0))
    raise ProviderError(f"{url} failed after {tries} attempts: {last}")


@dataclass
class OpenAICompatLLM:
    api_key: str
    model: str
    base_url: str
    name: str = "openai-compatible"
    timeout: float = 90.0
    tries: int = 4
    cache: ResponseCache | None = None
    calls: list[Call] = field(default_factory=list)

    def __post_init__(self) -> None:
        import httpx

        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": kw.get("temperature", 0.0),
            "max_tokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        cache_key = ResponseCache.key(self.model, messages, **kw)
        if self.cache:
            hit = self.cache.get(cache_key)
            if hit:
                self.calls.append(Call(text=hit["text"], cached=True))
                return hit["text"]

        started = time.perf_counter()
        resp = _retry(self._client, "POST", f"{self.base_url}/chat/completions", tries=self.tries, json=payload)
        body = resp.json()
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        pt = usage.get("prompt_tokens") or approx_tokens(json.dumps(list(messages)))
        ct = usage.get("completion_tokens") or approx_tokens(text)

        call = Call(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=estimate_cost(self.model, pt, ct),
        )
        self.calls.append(call)
        if self.cache:
            self.cache.put(cache_key, {"text": text, "prompt_tokens": pt, "completion_tokens": ct})
        return text


@dataclass
class GeminiLLM:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    name: str = "gemini"
    timeout: float = 90.0
    tries: int = 4
    cache: ResponseCache | None = None
    calls: list[Call] = field(default_factory=list)

    def __post_init__(self) -> None:
        import httpx

        self._client = httpx.Client(timeout=self.timeout)

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        system = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        cfg: dict[str, Any] = {
            "temperature": kw.get("temperature", 0.0),
            "maxOutputTokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}

        cache_key = ResponseCache.key(self.model, messages, **kw)
        if self.cache:
            hit = self.cache.get(cache_key)
            if hit:
                self.calls.append(Call(text=hit["text"], cached=True))
                return hit["text"]

        started = time.perf_counter()
        resp = _retry(
            self._client,
            "POST",
            f"{self.base_url}/models/{self.model}:generateContent",
            tries=self.tries,
            params={"key": self.api_key},
            json=body,
        )
        data = resp.json()
        text = "".join(
            part["text"]
            for cand in data.get("candidates", [])
            for part in cand.get("content", {}).get("parts", [])
            if "text" in part
        )
        usage = data.get("usageMetadata") or {}
        pt = usage.get("promptTokenCount") or approx_tokens(json.dumps(list(messages)))
        ct = usage.get("candidatesTokenCount") or approx_tokens(text)

        self.calls.append(
            Call(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_ms=(time.perf_counter() - started) * 1000,
                cost_usd=estimate_cost(self.model, pt, ct),
            )
        )
        if self.cache:
            self.cache.put(cache_key, {"text": text, "prompt_tokens": pt, "completion_tokens": ct})
        return text


@dataclass
class ScriptedLLM:
    """Deterministic provider for testing the harness itself.

    An eval harness has to be tested like any other code, and testing it against
    a real model would make its own test suite flaky and expensive. This returns
    canned answers keyed by substring, which lets the tests assert that the
    harness computes the metrics it claims to.
    """

    name: str = "scripted"
    model: str = "scripted"
    responses: dict[str, str] = field(default_factory=dict)
    default: str = "I don't have enough information to answer that."
    calls: list[Call] = field(default_factory=list)
    # Fraction of calls that return a deliberately wrong answer, so variance
    # and confidence intervals can be exercised.
    noise: float = 0.0
    seed: int = 20260101

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        user = " ".join(m["content"] for m in messages if m["role"] == "user")
        text = self.default
        for needle, response in self.responses.items():
            if needle.lower() in user.lower():
                text = response
                break
        if self.noise and self._rng.random() < self.noise:
            text = "Unfortunately that is not something I can determine."
        self.calls.append(Call(text=text, prompt_tokens=approx_tokens(user), completion_tokens=approx_tokens(text)))
        return text


def build_llm(provider: str = "", cache: ResponseCache | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    groq, gemini, openai = (
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("OPENAI_API_KEY", ""),
    )

    if provider in {"scripted", "offline"} or not (groq or gemini or openai):
        return ScriptedLLM()
    if provider == "groq" and groq:
        return OpenAICompatLLM(
            groq,
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            name="groq",
            cache=cache,
        )
    if provider == "gemini" and gemini:
        return GeminiLLM(gemini, os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), cache=cache)
    if provider in {"openai", "openai-compatible", "vllm", "ollama"} and openai:
        return OpenAICompatLLM(
            openai,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            name=provider,
            cache=cache,
        )
    return ScriptedLLM()


# --------------------------------------------------------------------------
# targets: the thing under evaluation
# --------------------------------------------------------------------------


class Target(Protocol):
    name: str

    def run(self, item: dict[str, Any]) -> Call: ...


@dataclass
class PromptTarget:
    """Evaluate a model behind a prompt template."""

    llm: LLM
    name: str = "prompt"
    system: str = ""
    template: str = "{input}"
    temperature: float = 0.0
    max_tokens: int = 900
    json_mode: bool = False

    def run(self, item: dict[str, Any]) -> Call:
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        try:
            prompt = self.template.format(**item)
        except KeyError as exc:
            return Call(text="", error=f"template references missing field {exc}")
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        try:
            text = self.llm.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_mode=self.json_mode,
            )
        except Exception as exc:
            return Call(text="", error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - started) * 1000)

        last = self.llm.calls[-1] if getattr(self.llm, "calls", None) else Call(text=text)
        return Call(
            text=text,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            latency_ms=last.latency_ms or (time.perf_counter() - started) * 1000,
            cost_usd=last.cost_usd,
            cached=last.cached,
        )


@dataclass
class HttpTarget:
    """Evaluate a running service — a RAG API, an agent, anything with an HTTP
    endpoint. This is what makes the harness useful beyond prompt tuning: the
    thing shipped to users is the service, not the prompt."""

    url: str
    name: str = "http"
    method: str = "POST"
    # Template for the request body; `{input}` is substituted.
    body_template: dict[str, Any] = field(default_factory=lambda: {"query": "{input}"})
    # Dotted path to the answer in the response, e.g. "answer" or "data.reply".
    response_path: str = "answer"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    tries: int = 3

    def _render(self, template: Any, item: dict[str, Any]) -> Any:
        if isinstance(template, str):
            try:
                return template.format(**item)
            except (KeyError, IndexError):
                return template
        if isinstance(template, dict):
            return {k: self._render(v, item) for k, v in template.items()}
        if isinstance(template, list):
            return [self._render(v, item) for v in template]
        return template

    def _extract(self, data: Any) -> str:
        current = data
        for part in self.response_path.split("."):
            if isinstance(current, dict):
                current = current.get(part, "")
            else:
                return str(current)
        return current if isinstance(current, str) else json.dumps(current, default=str)

    def run(self, item: dict[str, Any]) -> Call:
        import httpx

        body = self._render(self.body_template, item)
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                resp = _retry(client, self.method, self.url, tries=self.tries, json=body)
            text = self._extract(resp.json())
        except Exception as exc:
            return Call(
                text="",
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return Call(text=text, latency_ms=(time.perf_counter() - started) * 1000)


@dataclass
class CallableTarget:
    """Wrap any Python callable, for evaluating a local pipeline in-process."""

    fn: Any
    name: str = "callable"

    def run(self, item: dict[str, Any]) -> Call:
        started = time.perf_counter()
        try:
            result = self.fn(item)
        except Exception as exc:
            return Call(text="", error=f"{type(exc).__name__}: {exc}")
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return Call(text=text, latency_ms=(time.perf_counter() - started) * 1000)
