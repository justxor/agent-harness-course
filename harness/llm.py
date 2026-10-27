"""Тонкий слой над LLM-провайдерами. Только стандартная библиотека Python.

Внутренний формат сообщений (нейтральный, похож на OpenAI):
    {"role": "system" | "user" | "assistant" | "tool",
     "content": str,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}],   # только у assistant
     "tool_call_id": str}                                            # только у tool

Провайдеры:
    ScriptedLLM      — заранее записанные ответы. Работает без ключа и интернета,
                       на нём построены все демо и тесты.
    OpenAICompatLLM  — любой /chat/completions: OpenAI, OpenRouter, DeepSeek,
                       Ollama, LM Studio, vLLM и т.д.
    AnthropicLLM     — нативный Anthropic Messages API.

Выбор через переменные окружения — см. from_env().
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end"

    def as_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        return msg


class LLMError(RuntimeError):
    pass


class LLM:
    """Базовый интерфейс. Харнесу всё равно, какой провайдер внутри."""

    name = "base"

    def complete(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 1024) -> Reply:
        raise NotImplementedError

    def stream_text(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        """По умолчанию — «стрим» одним куском. Провайдеры переопределяют."""
        yield self.complete(system, messages, None, max_tokens).text


# ---------------------------------------------------------------------------
# 1. Scripted — детерминированная модель для демо и тестов
# ---------------------------------------------------------------------------

class ScriptedLLM(LLM):
    """Возвращает ответы по сценарию.

    script — список Reply (отдаются по очереди) или функция
    (system, messages, tools) -> Reply, если ответ зависит от контекста.
    """

    name = "scripted"

    def __init__(self, script: list[Reply] | Callable[[str, list[dict], list[dict] | None], Reply]):
        self.script = script
        self.calls: list[dict] = []  # что модель «видела» — удобно проверять в тестах

    def complete(self, system, messages, tools=None, max_tokens=1024):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if callable(self.script):
            reply = self.script(system, messages, tools)
        else:
            if not self.script:
                raise LLMError("ScriptedLLM: сценарий закончился")
            reply = self.script.pop(0)
        # грубая оценка токенов, чтобы демо счётчиков тоже работали
        reply.input_tokens = reply.input_tokens or approx_tokens(system) + sum(approx_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
        reply.output_tokens = reply.output_tokens or approx_tokens(reply.text) + 20 * len(reply.tool_calls)
        return reply

    def stream_text(self, system, messages, max_tokens=1024):
        text = self.complete(system, messages, None, max_tokens).text
        for word in text.split(" "):
            yield word + " "


def approx_tokens(text: str) -> int:
    """Эвристика «символы / 4». Для русского текста занижает (~в 1.5–2 раза), но для бюджета ок."""
    return max(1, len(text) // 4) if text else 0


# ---------------------------------------------------------------------------
# 2. HTTP-утилиты
# ---------------------------------------------------------------------------

def _post(url: str, headers: dict, body: dict, timeout: int = 120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise LLMError(f"HTTP {e.code} от {url}: {detail}") from e


# ---------------------------------------------------------------------------
# 3. OpenAI-совместимые API
# ---------------------------------------------------------------------------

class OpenAICompatLLM(LLM):
    name = "openai-compatible"

    def __init__(self, model: str, api_key: str = "", base_url: str = "https://api.openai.com/v1"):
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")

    def _headers(self):
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _to_wire(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant", "content": m.get("content") or None,
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                   for c in m["tool_calls"]],
                })
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, system, messages, tools=None, max_tokens=1024):
        body = {"model": self.model, "max_tokens": max_tokens, "messages": self._to_wire(system, messages)}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        with _post(f"{self.base_url}/chat/completions", self._headers(), body) as r:
            data = json.load(r)
        choice = data["choices"][0]
        msg = choice["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": c["function"].get("arguments")}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        usage = data.get("usage") or {}
        return Reply(msg.get("content") or "", calls, usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0), choice.get("finish_reason") or "end")

    def stream_text(self, system, messages, max_tokens=1024):
        body = {"model": self.model, "max_tokens": max_tokens, "stream": True,
                "messages": self._to_wire(system, messages)}
        with _post(f"{self.base_url}/chat/completions", self._headers(), body) as r:
            for raw in r:  # Server-Sent Events: строки вида "data: {...}"
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]


# ---------------------------------------------------------------------------
# 4. Anthropic Messages API
# ---------------------------------------------------------------------------

class AnthropicLLM(LLM):
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, api_key: str):
        self.model, self.api_key = model, api_key

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                           for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # результаты инструментов идут в user-сообщении; подряд идущие склеиваем
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def complete(self, system, messages, tools=None, max_tokens=1024):
        body = {"model": self.model, "max_tokens": max_tokens, "messages": self._to_wire(messages)}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in tools]
        with _post(self.URL, self._headers(), body) as r:
            data = json.load(r)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"]) for b in data["content"] if b["type"] == "tool_use"]
        u = data.get("usage", {})
        return Reply(text, calls, u.get("input_tokens", 0), u.get("output_tokens", 0), data.get("stop_reason", "end"))

    def stream_text(self, system, messages, max_tokens=1024):
        body = {"model": self.model, "max_tokens": max_tokens, "stream": True,
                "messages": self._to_wire(messages), **({"system": system} if system else {})}
        with _post(self.URL, self._headers(), body) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                        yield ev["delta"]["text"]


# ---------------------------------------------------------------------------
# 5. Фабрика
# ---------------------------------------------------------------------------

def from_env(fallback: LLM | None = None) -> LLM:
    """LLM_PROVIDER = mock (по умолчанию) | openai | anthropic

    openai:    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL (по умолчанию https://api.openai.com/v1)
               OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1
               Ollama:     LLM_BASE_URL=http://localhost:11434/v1, ключ не нужен
    anthropic: LLM_MODEL, LLM_API_KEY (или ANTHROPIC_API_KEY)
    """
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    model = os.environ.get("LLM_MODEL", "")
    if provider == "mock":
        if fallback is None:
            raise LLMError("Нет сценария для mock-режима. Задай LLM_PROVIDER=openai|anthropic и LLM_MODEL.")
        return fallback
    if not model:
        raise LLMError("Задай LLM_MODEL — актуальные id смотри в документации провайдера.")
    if provider == "openai":
        return OpenAICompatLLM(model, os.environ.get("LLM_API_KEY", ""),
                               os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    if provider == "anthropic":
        key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicLLM(model, key)
    raise LLMError(f"Неизвестный LLM_PROVIDER={provider}")
