"""Урок 1. Один вызов модели: сообщения, роли, токены, стоимость, стриминг.

    python examples/01_hello_llm.py                 # без ключа (mock)
    LLM_PROVIDER=anthropic LLM_MODEL=<id> LLM_API_KEY=... python examples/01_hello_llm.py
    LLM_PROVIDER=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3 python examples/01_hello_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import Reply, ScriptedLLM, from_env  # noqa: E402

mock = ScriptedLLM([
    Reply("Харнес — это всё, что окружает модель: цикл, инструменты, контекст, память и интерфейс."),
    Reply("Модель — двигатель. Харнес — машина вокруг него: руль, колёса, тормоза и навигатор."),
])
llm = from_env(fallback=mock)
print(f"Провайдер: {llm.name}\n")

system = "Ты — лаконичный преподаватель. Отвечай одним предложением, по-русски."
messages = [{"role": "user", "content": "Что такое агентный харнес?"}]

# --- 1. Обычный вызов -------------------------------------------------------
r = llm.complete(system, messages)
print("Ответ:", r.text)
print(f"Токены: вход {r.input_tokens}, выход {r.output_tokens}")

# Модель НИЧЕГО не помнит между вызовами. «Память» диалога — это то, что мы
# сами присылаем в messages каждый раз.
messages.append(r.as_message())
messages.append({"role": "user", "content": "А теперь объясни через метафору."})

# --- 2. Стриминг: токены приходят по мере генерации --------------------------
print("\nСтрим: ", end="")
for chunk in llm.stream_text(system, messages):
    print(chunk, end="", flush=True)
print()

# --- 3. Стоимость ------------------------------------------------------------
# Цены смотри на странице провайдера; здесь — условные $3 / $15 за 1M токенов.
price_in, price_out = 3.0, 15.0
cost = r.input_tokens / 1e6 * price_in + r.output_tokens / 1e6 * price_out
print(f"\nСтоимость первого вызова ≈ ${cost:.6f}")
print("Заметь: каждый следующий вызов платит за ВСЮ историю заново — поэтому контекст надо беречь.")
