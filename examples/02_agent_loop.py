"""Урок 2. Агентный цикл и инструменты.

Модель сама решает, какие инструменты вызвать и сколько раз. Харнес исполняет
вызовы, возвращает результаты и крутит цикл, пока модель не ответит текстом.

    python examples/02_agent_loop.py
"""
import ast
import operator
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import Reply, ScriptedLLM, ToolCall, Workspace, from_env, run_agent, tool  # noqa: E402

# --- Инструменты -------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    # НИКОГДА не делай eval() на строке от модели. Разбираем AST сами.
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("допустимы только числа и + - * / **")


@tool("Вычислить арифметическое выражение. Используй для ЛЮБЫХ расчётов, не считай в уме.",
      params={"expression": "например '(12.5 * 4) / 3'"})
def calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval")))


@tool("Текущие дата и время")
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


ws = Workspace(Path(__file__).parent / "_workspace")
(ws.root / "budget.txt").write_text("аренда: 45000\nеда: 18000\nтранспорт: 4200\n", encoding="utf-8")
tools = [calculator, now, *ws.tools()]

# --- Сценарий для mock-режима (так выглядел бы ответ настоящей модели) --------
mock = ScriptedLLM([
    Reply("Сначала посмотрю, какие есть файлы.", [ToolCall("c1", "list_files", {})]),
    Reply("", [ToolCall("c2", "read_file", {"path": "budget.txt"})]),
    Reply("Посчитаю сумму и долю аренды.", [
        ToolCall("c3", "calculator", {"expression": "45000 + 18000 + 4200"}),
        ToolCall("c4", "calculator", {"expression": "45000 / (45000 + 18000 + 4200) * 100"}),
    ]),
    Reply("Расходы за месяц: 67 200 ₽. Аренда — около 67% бюджета, это главная статья."),
])
llm = from_env(fallback=mock)

system = """Ты — помощник с доступом к файлам пользователя и калькулятору.
Сначала собери данные инструментами, потом отвечай. Числа — только через calculator."""
messages = [{"role": "user", "content": "Сколько я трачу в месяц и какая доля уходит на аренду? Данные в рабочей папке."}]

result = run_agent(llm, system, messages, tools, max_steps=10)

print("\n— Итог —")
print(f"Шагов модели: {result.steps}, вызовов инструментов: {result.usage.tool_calls}, остановка: {result.stopped_by}")
print(f"Токены: {result.usage.input_tokens} вход / {result.usage.output_tokens} выход")
