"""Финальный проект: мини-репетитор по мотивам LearnHarness.

Фазы (как в оригинале): ЗАМЕР → ПЛАН → ОБУЧЕНИЕ, плюс журнал знаний.
  • probe   — агент-«замерщик» задаёт вопросы и вызывает probe_done с оценкой уровня;
  • plan    — учитель один раз строит план-граф зависимостей (plan_init), харнес проверяет, что это DAG;
  • teach   — объяснение шаг за шагом + quiz. quiz — client-side инструмент: его выполняет человек;
  • graph   — после урока утверждения записываются в append-only журнал знаний.

    python examples/capstone/mini_tutor.py --demo          # без ключа: сценарий + авто-ученик
    LLM_PROVIDER=... LLM_MODEL=... python examples/capstone/mini_tutor.py "Хочу понять, как работает TCP"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import FactMemory, Ledger, Reply, ScriptedLLM, ToolCall, from_env, run_agent, tool  # noqa: E402
from harness.llm import LLMError  # noqa: E402

DEMO = "--demo" in sys.argv or len([a for a in sys.argv[1:] if not a.startswith("--")]) == 0
GOAL = next((a for a in sys.argv[1:] if not a.startswith("--")), "Хочу понять, почему HTTPS безопасен")
DATA = Path(__file__).parent / "_data"
knowledge = FactMemory(Ledger(DATA / "knowledge.jsonl"))
state = {"phase": "probe", "assessment": None, "plan": None, "quiz_log": []}


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

@tool("Задать ученику вопрос с вариантами ответа. Ответ и уверенность ученика вернутся результатом.",
      client_side=True,
      params={"question": "вопрос", "options": "варианты ответа", "correct": "индекс правильного (с 0)",
              "plan_node": "id узла плана, который проверяется (или 'probe')"})
def quiz(question: str, options: list, correct: int, plan_node: str) -> str:
    raise RuntimeError("client-side: выполняется не здесь")


@tool("Завершить замер: записать, что ученик уже знает и чего нет.", terminates=True,
      params={"knows": "что уже понимает", "gaps": "пробелы", "goal": "цель ученика своими словами"})
def probe_done(knows: list, gaps: list, goal: str) -> str:
    state.update(phase="plan", assessment={"knows": knows, "gaps": gaps, "goal": goal})
    return "Замер сохранён. Передаю учителю."


@tool("Один раз создать план урока: граф зависимостей. edges — пары [от, к]: 'от' нужно понять раньше 'к'.",
      params={"nodes": "список объектов {id, title}", "edges": "список пар [from_id, to_id]"})
def plan_init(nodes: list, edges: list) -> str:
    if state["plan"]:
        return "ОШИБКА: план уже создан, он неизменяем. Прогресс отслеживается по квизам."
    ids = {n.get("id") for n in nodes}
    bad = [e for e in edges if len(e) != 2 or e[0] not in ids or e[1] not in ids]
    if bad:
        return f"ОШИБКА: рёбра ссылаются на несуществующие узлы: {bad}"
    order = topo_sort(ids, edges)
    if order is None:
        return "ОШИБКА: в плане есть цикл — это должен быть ациклический граф (DAG). Исправь и вызови снова."
    state.update(plan={"nodes": nodes, "edges": edges, "order": order}, phase="teach")
    return "План принят. Порядок обучения: " + " → ".join(order)


@tool("Записать в журнал знаний утверждения, которые ученик понял на этом уроке.",
      params={"claims": "самостоятельные утверждения, по одному на пункт"})
def record_knowledge(claims: list) -> str:
    added = [c for c in claims if isinstance(c, str) and knowledge.add(c, source=GOAL)]
    return f"Записано новых утверждений: {len(added)} (дубликаты пропущены)"


def topo_sort(ids, edges):
    indeg = {i: 0 for i in ids}
    for a, b in edges:
        indeg[b] += 1
    ready, order = sorted(i for i, d in indeg.items() if d == 0), []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for a, b in edges:
            if a == n:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
    return order if len(order) == len(ids) else None


def plan_status() -> str:
    """Статусы НЕ хранятся — выводятся из лога квизов (как в LearnHarness)."""
    if not state["plan"]:
        return ""
    passed = {q["node"] for q in state["quiz_log"] if q["ok"]}
    lines = [f"  {'✅' if n in passed else '⬜'} {n}" for n in state["plan"]["order"]]
    return "Прогресс по плану:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Клиент: как «интерфейс» выполняет quiz
# ---------------------------------------------------------------------------

AUTO_ANSWERS = iter([(1, 4), (0, 2), (0, 3), (0, 4), (2, 4), (1, 3)])  # (вариант, уверенность 1–5) для --demo


def client(t, args) -> str:
    print(f"\n❓ {args['question']}")
    for i, o in enumerate(args["options"]):
        print(f"   {i}) {o}")
    if DEMO:
        choice, conf = next(AUTO_ANSWERS, (args["correct"], 3))
        print(f"   [авто-ученик] выбираю {choice}, уверенность {conf}/5")
        thinking = "кажется, так" if choice == args["correct"] else "раз ключ публичный, наверное, прочитать может любой?"
    else:
        choice = int(input("   Твой ответ (номер): ") or -1)
        conf = int(input("   Уверенность 1–5: ") or 3)
        thinking = input("   Как рассуждал (можно пусто): ")
    ok = choice == args["correct"]
    state["quiz_log"].append({"node": args["plan_node"], "ok": ok})
    print("   ✔ верно" if ok else f"   ✘ неверно (правильно: {args['correct']})")
    # грейдинг делает КОД, а не модель — детерминированно и бесплатно
    return json.dumps({"correct": ok, "chosen": choice, "confidence": conf, "thinking": thinking}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Промпты агентов
# ---------------------------------------------------------------------------

PROBER = """Ты — замерщик. Цель ученика: «{goal}».
Задай 2 коротких вопроса через quiz (plan_node='probe'), чтобы найти, где кончается понимание.
Уверенность и рассуждение ученика важнее правильности. Затем вызови probe_done. Не объясняй материал."""

TEACHER = """Ты — учитель. Замер: {assessment}.
1) Если плана нет — вызови plan_init: 3–5 узлов от того, что ученик знает, к цели.
2) Иди по порядку плана. На каждом узле: короткое объяснение, опирающееся на предыдущий узел, затем quiz по узлу.
   Ошибка или низкая уверенность → объясни иначе и проверь снова, прежде чем идти дальше.
3) В конце вызови record_knowledge с выученными утверждениями и подведи итог."""


def demo_prober(system, messages, tools):
    n_quiz = sum(1 for m in messages if m["role"] == "tool")
    if n_quiz == 0:
        return Reply("Сначала пара вопросов, чтобы понять, откуда начать.", [ToolCall("p1", "quiz", {
            "question": "Что делает шифрование?", "options": ["Сжимает данные", "Делает данные нечитаемыми без ключа", "Ускоряет сеть"],
            "correct": 1, "plan_node": "probe"})])
    if n_quiz == 1:
        return Reply("", [ToolCall("p2", "quiz", {
            "question": "Чем асимметричное шифрование отличается от симметричного?",
            "options": ["Два разных ключа: публичный и приватный", "Оно быстрее", "Не использует ключи"],
            "correct": 0, "plan_node": "probe"})])
    return Reply("", [ToolCall("p3", "probe_done", {
        "knows": ["что такое шифрование"], "gaps": ["асимметричная криптография", "сертификаты", "TLS-рукопожатие"],
        "goal": "понять, почему HTTPS защищает от перехвата"})])


DEMO_TEACHER = [
    Reply("", [ToolCall("t1", "plan_init", {
        "nodes": [{"id": "sym", "title": "Симметричный ключ"}, {"id": "asym", "title": "Публичный и приватный ключ"},
                  {"id": "cert", "title": "Сертификат и CA"}, {"id": "tls", "title": "TLS-рукопожатие"}],
        "edges": [["sym", "tls"], ["asym", "cert"], ["cert", "tls"], ["asym", "tls"]]})]),
    Reply("**Публичный и приватный ключ.** Замок, который защёлкнет любой (публичный ключ), но откроет только хозяин "
          "(приватный). Так можно передать секрет, не встречаясь заранее.", [ToolCall("t2", "quiz", {
              "question": "Сайт прислал публичный ключ. Кто сможет прочитать то, что ты им зашифруешь?",
              "options": ["Любой, у кого есть публичный ключ", "Только владелец приватного ключа", "Никто"],
              "correct": 1, "plan_node": "asym"})]),
    Reply("Посмотрим на это с другой стороны: публичный ключ — это открытый почтовый ящик с прорезью. Бросить письмо может "
          "каждый, а достать — только у кого ключ от ящика.", [ToolCall("t3", "quiz", {
              "question": "У кого ключ «от ящика»?", "options": ["У владельца сайта (приватный ключ)", "У всех", "У провайдера"],
              "correct": 0, "plan_node": "asym"})]),
    Reply("**Сертификат.** Как понять, что ключ правда от bank.ru, а не от злоумышленника? Его подписывает центр "
          "сертификации (CA), которому заранее доверяет браузер.", [ToolCall("t4", "quiz", {
              "question": "Зачем нужен CA?", "options": ["Шифрует трафик", "Ускоряет загрузку", "Подтверждает, что ключ принадлежит сайту"],
              "correct": 2, "plan_node": "cert"})]),
    Reply("**Симметричный ключ и TLS.** Асимметрия медленная, поэтому ей передают только общий симметричный ключ, "
          "а дальше трафик шифруется им — быстро.", [ToolCall("t5", "quiz", {
              "question": "Чем шифруется основной трафик HTTPS?", "options": ["Приватным ключом сайта", "Общим симметричным ключом сессии", "Сертификатом"],
              "correct": 1, "plan_node": "tls"})]),
    Reply("", [ToolCall("t6", "record_knowledge", {"claims": [
        "Публичным ключом шифруют, расшифровать может только владелец приватного ключа",
        "Сертификат, подписанный CA, связывает публичный ключ с доменом",
        "Основной трафик HTTPS шифруется симметричным ключом сессии, согласованным в TLS-рукопожатии"]})]),
    Reply("Итог: ты понял, зачем нужны два ключа, кто ручается за ключ сайта и почему сам трафик шифруется "
          "быстрым симметричным ключом. Узел «Симметричный ключ» стоит закрепить отдельной практикой."),
]


def main():
    try:
        prober_llm = from_env(fallback=ScriptedLLM(demo_prober))
        teacher_llm = from_env(fallback=ScriptedLLM(DEMO_TEACHER))
    except LLMError as e:
        sys.exit(str(e))

    print(f"🎯 Цель: {GOAL}\n\n— ФАЗА 1: замер —")
    msgs = [{"role": "user", "content": GOAL}]
    run_agent(prober_llm, PROBER.format(goal=GOAL), msgs, [quiz, probe_done], client_handler=client, max_steps=8)
    if state["phase"] != "plan":
        sys.exit("Замер не завершён.")
    print(f"\n📋 Оценка: {json.dumps(state['assessment'], ensure_ascii=False)}")

    print("\n— ФАЗА 2–3: план и обучение —")
    # Учитель стартует с ЧИСТОГО контекста: ему передаётся только итог замера, не вся переписка
    msgs = [{"role": "user", "content": f"Начинаем. Цель: {GOAL}"}]
    run_agent(teacher_llm, TEACHER.format(assessment=json.dumps(state["assessment"], ensure_ascii=False)),
              msgs, [plan_init, quiz, record_knowledge], client_handler=client, max_steps=25)

    print("\n" + plan_status())
    print(f"\n🧠 Журнал знаний ({len(knowledge.facts())} утверждений): {knowledge.ledger.path}")


if __name__ == "__main__":
    main()
