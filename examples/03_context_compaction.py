"""Урок 3. Контекст: бюджет токенов, компакция и починка «висящих» вызовов.

    python examples/03_context_compaction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import (Reply, ScriptedLLM, build_system_prompt, compact, repair_dangling_tool_calls,  # noqa: E402
                     select_cutoff, total_tokens)

# --- 1. Системный промпт из фрагментов ---------------------------------------
prompt = build_system_prompt(Path(__file__).parent / "prompts" / "assistant", user_name="Гоша")
print("Системный промпт собран из фрагментов:\n" + "-" * 40 + f"\n{prompt}\n" + "-" * 40)

# --- 2. Длинная история с вызовами инструментов -------------------------------
history = []
for i in range(1, 9):
    history.append({"role": "user", "content": f"Шаг {i}: проверь лог сервиса номер {i}. " + "подробности " * 40})
    history.append({"role": "assistant", "content": f"Смотрю лог {i}.",
                    "tool_calls": [{"id": f"t{i}", "name": "read_file", "arguments": {"path": f"svc{i}.log"}}]})
    history.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"svc{i}: OK, 0 ошибок. " + "строка лога " * 60})
    history.append({"role": "assistant", "content": f"Сервис {i} в порядке."})

print(f"\nИстория: {len(history)} сообщений, ≈{total_tokens(history)} токенов")

cut = select_cutoff(history, keep_tokens=600)
print(f"Граница компакции: сообщение #{cut} (роль {history[cut]['role']!r}) — хвост никогда не начинается с tool")

summarizer = ScriptedLLM([Reply(
    "- Пользователь проверяет логи сервисов 1–8\n- Сервисы 1–6 проверены: ошибок нет\n- Инструмент: read_file(svcN.log)")])
compacted = compact(summarizer, history, keep_tokens=600)
print(f"После компакции: {len(compacted)} сообщений, ≈{total_tokens(compacted)} токенов")
print("Первое сообщение теперь:\n ", compacted[0]["content"].replace("\n", "\n  "))

# --- 3. Висящий вызов: процесс упал между вызовом и результатом ---------------
broken = [
    {"role": "user", "content": "Удали временные файлы"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "name": "list_files", "arguments": {}},
        {"id": "b", "name": "delete", "arguments": {"path": "tmp"}}]},
    {"role": "tool", "tool_call_id": "a", "content": "tmp/1.txt"},
    # ← здесь харнес упал, результата для 'b' нет; API откажется принимать такую историю
    {"role": "user", "content": "Ну что там?"},
]
fixed = repair_dangling_tool_calls(broken)
print("\nПосле починки:")
for m in fixed:
    print(f"  [{m['role']}] {m.get('content') or m.get('tool_calls')}")
