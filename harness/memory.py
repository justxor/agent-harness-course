"""Память и состояние: транскрипт на диске, append-only журналы, долговременные факты."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from .llm import LLM


# ---------------------------------------------------------------------------
# 1. Транскрипт сессии — write-ahead + атомарная запись
# ---------------------------------------------------------------------------

class SessionStore:
    """Сессия = JSON-файл. Сохраняем ДО вызова модели (write-ahead): если процесс упадёт
    посреди ответа, сообщение пользователя не потеряется, а сессию можно продолжить."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sid: str) -> Path:
        if not re.fullmatch(r"[\w-]+", sid):
            raise ValueError("недопустимый id сессии")
        return self.root / f"{sid}.json"

    def load(self, sid: str) -> dict:
        p = self.path(sid)
        if not p.exists():
            return {"id": sid, "created": time.time(), "messages": []}
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, session: dict) -> None:
        p = self.path(session["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)  # атомарно: либо старая версия, либо новая, никогда — половина

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


# ---------------------------------------------------------------------------
# 2. Append-only журнал (JSONL)
# ---------------------------------------------------------------------------

class Ledger:
    """Только дописываем. Чтение терпит битые строки (обрыв записи при падении).
    Всё «текущее состояние» вычисляется из журнала при чтении."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # битая строка — пропускаем, не угадываем
        return out


# ---------------------------------------------------------------------------
# 3. Долговременная память: извлечение фактов и поиск
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Извлеки из текста устойчивые факты о пользователе и его задаче,
которые пригодятся в будущих сессиях (предпочтения, решения, контекст проекта).
Не включай разовые детали. Каждый факт — одно самостоятельное утверждение.
Вызови инструмент save_facts."""

SAVE_FACTS_SCHEMA = {
    "name": "save_facts",
    "description": "Сохранить список фактов",
    "parameters": {"type": "object", "required": ["facts"],
                   "properties": {"facts": {"type": "array", "items": {"type": "string"}}}},
}


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def fact_id(text: str) -> str:
    return "f_" + hashlib.sha256(_norm(text).encode()).hexdigest()[:12]


class FactMemory:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def facts(self) -> dict[str, str]:
        """Текущее состояние = свёртка журнала (add / forget)."""
        state: dict[str, str] = {}
        for r in self.ledger.read():
            if r.get("op") == "add":
                state[r["id"]] = r["text"]
            elif r.get("op") == "forget":
                state.pop(r["id"], None)
        return state

    def add(self, text: str, source: str) -> bool:
        fid = fact_id(text)
        if fid in self.facts():
            return False  # дедупликация по нормализованному тексту
        self.ledger.append({"op": "add", "id": fid, "text": text, "source": source})
        return True

    def forget(self, fid: str) -> None:
        self.ledger.append({"op": "forget", "id": fid})

    def extract(self, llm: LLM, text: str, source: str) -> list[str]:
        r = llm.complete(EXTRACT_PROMPT, [{"role": "user", "content": text}], [SAVE_FACTS_SCHEMA], 600)
        added = []
        for c in r.tool_calls:
            if c.name == "save_facts":
                for f in c.arguments.get("facts", []):
                    if isinstance(f, str) and f.strip() and self.add(f.strip(), source):
                        added.append(f.strip())
        return added

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Простейший поиск по пересечению слов. В продакшене — эмбеддинги + BM25."""
        q = set(_norm(query).split())
        scored = []
        for text in self.facts().values():
            words = set(_norm(text).split())
            score = len(q & words) / (len(words) ** 0.5 or 1)
            if score > 0:
                scored.append((score, text))
        return [t for _, t in sorted(scored, reverse=True)[:k]]
