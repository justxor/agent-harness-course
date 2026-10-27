"""Урок 7. Оценка (evals): как понять, что изменение промпта или модели не сломало агента.

Три уровня проверок:
  1. детерминированные — вызвал ли нужный инструмент, уложился ли в N шагов, есть ли подстрока;
  2. по траектории — порядок и аргументы вызовов;
  3. LLM-судья — для оценки качества текста (дорого и шумно, используем точечно).

    python examples/07_evals.py
    LLM_PROVIDER=... LLM_MODEL=... python examples/07_evals.py   # прогнать на реальной модели
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import LLM, Reply, ScriptedLLM, ToolCall, from_env, run_agent, tool  # noqa: E402
from harness.llm import LLMError  # noqa: E402

ORDERS = {"A-100": "доставлен 12.10", "A-101": "в пути, прибудет 29.10"}


@tool("Статус заказа", params={"order_id": "номер вида A-123"})
def order_status(order_id: str) -> str:
    return ORDERS.get(order_id, "заказ не найден")


@tool("Передать разговор человеку-оператору", params={"reason": "причина"})
def escalate(reason: str) -> str:
    return "оператор подключится в течение 5 минут"


TOOLS = [order_status, escalate]
SYSTEM = """Ты — поддержка магазина. Статусы заказов — только через order_status.
Если клиент злится, просит возврат денег или человека — вызови escalate. Не обещай того, чего не знаешь."""


@dataclass
class Case:
    name: str
    user: str
    checks: list[Callable] = field(default_factory=list)


def called(name, **args):
    def check(res):
        ok = any(t["name"] == name and all(t["arguments"].get(k) == v for k, v in args.items())
                 for t in res.tool_log)
        return ok, f"вызван {name}{args or ''}"
    return check


def not_called(name):
    return lambda res: (all(t["name"] != name for t in res.tool_log), f"НЕ вызван {name}")


def contains(*words):
    return lambda res: (all(w.lower() in res.text.lower() for w in words), f"ответ содержит {words}")


def max_steps(n):
    return lambda res: (res.steps <= n, f"шагов ≤ {n} (было {res.steps})")


CASES = [
    Case("статус заказа", "Где мой заказ A-101?",
         [called("order_status", order_id="A-101"), contains("29.10"), not_called("escalate"), max_steps(3)]),
    Case("несуществующий заказ", "Что с заказом A-999?",
         [called("order_status", order_id="A-999"), not_called("escalate"), max_steps(3)]),
    Case("эскалация", "Верните деньги немедленно, это уже третий раз!",
         [called("escalate"), max_steps(3)]),
    Case("без галлюцинаций", "Когда у вас распродажа?",
         [not_called("order_status"), contains("не")]),
]


def mock_agent(system, messages, tools):
    """Мок-«модель» с одной намеренной ошибкой, чтобы было видно, как eval её ловит."""
    last = messages[-1]
    if last["role"] == "tool":
        content = last["content"]
        return Reply(f"Статус: {content}." if "оператор" not in content else "Передаю оператору, он скоро подключится.")
    text = last["content"]
    if "A-" in text:
        oid = "A-" + text.split("A-")[1][:3]
        return Reply("", [ToolCall("x", "order_status", {"order_id": oid})])
    if "деньги" in text:
        return Reply("Понимаю ваше недовольство, сейчас разберёмся!")   # ← баг: забыл escalate
    return Reply("Точных дат распродаж у меня нет — следите за рассылкой.")


def run_suite(llm: LLM):
    passed = total = 0
    for case in CASES:
        res = run_agent(llm, SYSTEM, [{"role": "user", "content": case.user}], TOOLS,
                        max_steps=5, on_event=lambda k, d: None)
        results = [c(res) for c in case.checks]
        ok = all(r[0] for r in results)
        passed += ok
        total += 1
        print(f"{'✅' if ok else '❌'} {case.name}")
        for good, desc in results:
            if not good:
                print(f"     ✗ {desc}")
        if not ok:
            print(f"     траектория: {json.dumps([t['name'] for t in res.tool_log], ensure_ascii=False)} · ответ: {res.text[:80]!r}")
    print(f"\nИтог: {passed}/{total}")
    return passed, total


try:
    llm = from_env(fallback=ScriptedLLM(mock_agent))
except LLMError as e:
    sys.exit(str(e))
run_suite(llm)
print("\nСовет: прогоняй каждый кейс 5–10 раз на реальной модели — агенты недетерминированы, "
      "смотри на долю успехов, а не на один прогон.")
