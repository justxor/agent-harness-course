"""Урок 5. Субагенты и фоновые задачи: ack-and-steer.

Главный агент отдаёт исследование субагенту и продолжает отвечать пользователю.
Субагент работает в отдельном потоке со СВОИМ контекстом — главный видит только итог.

    python examples/05_subagents.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import JobRunner, Reply, ScriptedLLM, ToolCall, run_agent, steer_message, tool  # noqa: E402

runner = JobRunner()


# --- Субагент: свой системный промпт, свои инструменты, свой цикл ------------
def research_subagent(inp: dict) -> str:
    @tool("Поиск по базе знаний", params={"query": "запрос"})
    def search(query: str) -> str:
        time.sleep(0.5)  # имитация медленного веба
        return {"rate limit": "Anthropic и OpenAI ограничивают RPM и TPM; при 429 — экспоненциальный backoff.",
                "retry": "Повторять только идемпотентные запросы; учитывать заголовок retry-after."}.get(query, "ничего")

    sub_llm = ScriptedLLM([
        Reply("", [ToolCall("s1", "search", {"query": "rate limit"}), ToolCall("s2", "search", {"query": "retry"})]),
        Reply("Итог: при 429 делай экспоненциальный backoff с джиттером, уважай retry-after, "
              "повторяй только идемпотентные запросы; лимиты бывают по запросам и по токенам."),
    ])
    res = run_agent(sub_llm, "Ты — исследователь. Найди факты и верни сжатый итог.",
                    [{"role": "user", "content": inp["question"]}], [search],
                    on_event=lambda k, d: None)  # субагент работает «тихо»
    return res.text


# --- Инструмент главного агента: мгновенный ack ------------------------------
@tool("Запустить фоновое исследование. Возвращает id задачи сразу; результат придёт позже отдельным сообщением.",
      params={"question": "что исследовать"})
def research(question: str) -> str:
    jid = runner.spawn("research", {"question": question}, research_subagent)
    return f"Принято: {jid}. Продолжай работу, результат придёт сам."


main_llm = ScriptedLLM([
    Reply("Запущу исследование в фоне, а пока отвечу на то, что знаю.",
          [ToolCall("m1", "research", {"question": "Как правильно обрабатывать rate limit у LLM API?"})]),
    Reply("Пока идёт поиск: сначала логируй каждый 429 и заголовки ответа — это пригодится. Детали скоро."),
    # второй прогон — после пробуждения steer-сообщением
    Reply("Пришли результаты исследования: добавь backoff с джиттером и учитывай retry-after. "
          "Код-ревью: твой цикл повторяет и POST-запросы на оплату — так нельзя, они не идемпотентны."),
])

system = "Ты — главный агент. Долгие исследования отдавай инструменту research и не жди их."
messages = [{"role": "user", "content": "Мой сервис ловит 429 от LLM API. Что делать?"}]

t0 = time.time()
run_agent(main_llm, system, messages, [research])
print(f"\n⏱  Пользователь получил первый ответ через {time.time() - t0:.2f} c; в фоне: {runner.pending()}")

# --- Пробуждение: харнес ждёт результаты и запускает новый ход ---------------
for s in runner.collect(timeout=5):
    print(f"\n📬 Доставлено {s['steer_id']} (exactly-once)")
    messages.append(steer_message(s))
    run_agent(main_llm, system, messages, [research])

# повторная доставка того же результата будет отброшена
runner.inbox.put({"steer_id": "steer_job_1", "job": "job_1", "kind": "research", "status": "done", "result": "дубль"})
print(f"\nПовторная доставка steer_job_1 → получено {len(runner.collect())} сообщений (дубль отброшен)")
