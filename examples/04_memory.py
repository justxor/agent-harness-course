"""Урок 4. Состояние и память: сессии на диске и долговременные факты.

Запусти два раза подряд — во второй раз агент «вспомнит» прошлую сессию.

    python examples/04_memory.py
    python examples/04_memory.py --reset   # стереть память
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import FactMemory, Ledger, Reply, ScriptedLLM, SessionStore, ToolCall, from_env  # noqa: E402

DATA = Path(__file__).parent / "_data"
if "--reset" in sys.argv:
    shutil.rmtree(DATA, ignore_errors=True)
    print("Память стёрта.")
    sys.exit()

sessions = SessionStore(DATA / "sessions")
memory = FactMemory(Ledger(DATA / "facts.jsonl"))


def scripted(system, messages, tools):
    """Mock: ведёт себя по-разному для извлечения фактов и для ответа."""
    if tools and tools[0]["name"] == "save_facts":
        return Reply("", [ToolCall("f", "save_facts", {"facts": [
            "Пользователь пишет бэкенд на Go",
            "Пользователь предпочитает PostgreSQL, а не MongoDB",
            "Проект пользователя — сервис бронирования переговорок",
        ]})])
    if "Что ты знаешь" in system:
        return Reply("Помню: ты пишешь на Go, любишь PostgreSQL и делаешь сервис бронирования переговорок. "
                     "Продолжим со схемой БД?")
    return Reply("Отлично: Go + PostgreSQL — хороший выбор для бронирований, транзакции и exclusion-constraint "
                 "защитят от двойных броней.")


llm = from_env(fallback=ScriptedLLM(scripted))
known = memory.facts()

if not known:
    print("— Сессия 1: памяти нет —")
    s = sessions.load("s1")
    s["messages"].append({"role": "user", "content":
                          "Я пишу бэкенд на Go для сервиса бронирования переговорок. Монгу не люблю, беру Postgres."})
    sessions.save(s)  # write-ahead: сохранили ДО вызова модели
    r = llm.complete("Ты — помощник разработчика.", s["messages"])
    s["messages"].append(r.as_message())
    sessions.save(s)
    print("🤖", r.text)

    added = memory.extract(llm, s["messages"][0]["content"], source="s1")
    print("\nИзвлечённые факты (сохранены в facts.jsonl):")
    for f in added:
        print("  •", f)
    print("\nЗапусти скрипт ещё раз — начнётся новая сессия с этой памятью.")
else:
    print("— Сессия 2: память есть —")
    relevant = memory.recall("база данных проект бронирования Go")
    facts_block = "\n".join(f"- {f}" for f in relevant)
    system = f"Ты — помощник разработчика.\n\nЧто ты знаешь о пользователе из прошлых сессий:\n{facts_block}"
    print("В системный промпт подмешано:\n" + facts_block)
    print("(Факт про PostgreSQL не нашёлся: в запросе «база данных», а в факте «PostgreSQL». "
          "Поиск по словам не видит синонимов — для этого нужны эмбеддинги, см. урок 4.4.)")
    r = llm.complete(system, [{"role": "user", "content": "Привет! На чём мы остановились?"}])
    print("\n🤖", r.text)
    print(f"\nСессии на диске: {sessions.list()}  ·  фактов в памяти: {len(known)}")
    print("Сырой журнал:", memory.ledger.path)
