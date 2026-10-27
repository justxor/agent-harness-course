"""Агентный цикл — сердце любого харнеса.

    while True:
        ответ = модель(система, история, инструменты)
        если в ответе нет вызовов инструментов → готово
        выполнить инструменты → добавить результаты в историю → повторить

Всё остальное в харнесе (компакция, память, субагенты, UI) — обвязка вокруг этих 5 строк.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM, Reply
from .tools import Tool, validate_args


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0

    def add(self, r: Reply):
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.model_calls += 1

    def cost(self, usd_per_mtok_in: float, usd_per_mtok_out: float) -> float:
        return self.input_tokens / 1e6 * usd_per_mtok_in + self.output_tokens / 1e6 * usd_per_mtok_out


@dataclass
class RunResult:
    text: str
    messages: list[dict]
    usage: Usage
    steps: int
    stopped_by: str  # "done" | "max_steps" | "terminated"
    tool_log: list[dict] = field(default_factory=list)


def default_events(kind: str, data: dict):
    if kind == "tool_call":
        print(f"  🔧 {data['name']}({json.dumps(data['arguments'], ensure_ascii=False)[:120]})")
    elif kind == "tool_result":
        preview = data["content"].replace("\n", " ")[:120]
        print(f"  ↳ {preview}")
    elif kind == "assistant" and data["text"]:
        print(f"🤖 {data['text']}")


def ask_in_terminal(t: Tool, args: dict) -> bool:
    ans = input(f"⚠️  Агент хочет вызвать {t.name}({json.dumps(args, ensure_ascii=False)[:200]}). Разрешить? [y/N] ")
    return ans.strip().lower() in {"y", "yes", "д", "да"}


def run_agent(
    llm: LLM,
    system: str,
    messages: list[dict],
    tools: list[Tool] | None = None,
    *,
    max_steps: int = 20,
    on_event: Callable[[str, dict], None] = default_events,
    approve: Callable[[Tool, dict], bool] = ask_in_terminal,
    client_handler: Callable[[Tool, dict], str] | None = None,
    before_model_call: Callable[[list[dict]], list[dict]] | None = None,
    max_tokens: int = 1024,
) -> RunResult:
    """Запускает цикл. messages изменяется на месте — это и есть транскрипт.

    before_model_call — хук перед каждым вызовом модели (сюда подключают компакцию).
    client_handler   — исполнитель client_side-инструментов (вопрос человеку, UI-квиз).
    """
    tools = tools or []
    by_name = {t.name: t for t in tools}
    usage, tool_log = Usage(), []

    for step in range(1, max_steps + 1):
        if before_model_call:
            messages[:] = before_model_call(messages)

        reply = llm.complete(system, messages, [t.schema() for t in tools] or None, max_tokens)
        usage.add(reply)
        messages.append(reply.as_message())
        on_event("assistant", {"text": reply.text, "step": step})

        if not reply.tool_calls:
            return RunResult(reply.text, messages, usage, step, "done", tool_log)

        # Модель может попросить несколько инструментов сразу. На КАЖДЫЙ вызов
        # обязательно должен быть ответ, иначе следующий запрос к API упадёт.
        terminate = False
        for call in reply.tool_calls:
            on_event("tool_call", {"name": call.name, "arguments": call.arguments})
            started = time.time()
            content = _execute(call.name, call.arguments, by_name, approve, client_handler)
            usage.tool_calls += 1
            tool_log.append({"name": call.name, "arguments": call.arguments, "result": content,
                             "ms": round((time.time() - started) * 1000)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            on_event("tool_result", {"name": call.name, "content": content})
            t = by_name.get(call.name)
            if t and t.terminates and not content.startswith(("ОШИБКА", "ОТКАЗАНО")):
                terminate = True

        if terminate:  # например, probe_done: ход переходит к другому агенту
            return RunResult(reply.text, messages, usage, step, "terminated", tool_log)

    return RunResult("(остановлено: превышен лимит шагов)", messages, usage, max_steps, "max_steps", tool_log)


def _execute(name, args, by_name, approve, client_handler) -> str:
    t = by_name.get(name)
    if t is None:
        return f"ОШИБКА: инструмента '{name}' нет. Доступны: {sorted(by_name)}"
    err = validate_args(t, args)
    if err:
        return f"ОШИБКА валидации: {err}"
    if t.needs_approval and not approve(t, args):
        return "ОТКАЗАНО пользователем. Не повторяй этот вызов; предложи альтернативу или спроси, что делать."
    if t.client_side:
        if client_handler is None:
            return "ОШИБКА: нет клиента для этого инструмента"
        return client_handler(t, args)
    try:
        out = t.fn(**args)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    except Exception as e:  # ошибка инструмента → текст для модели, а не падение харнеса
        return f"ОШИБКА выполнения {name}: {type(e).__name__}: {e}"
