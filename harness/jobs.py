"""Фоновые задачи и субагенты по схеме ack-and-steer.

1. Главный агент вызывает инструмент `research(...)`.
2. Инструмент СРАЗУ отвечает «принято, job_3» (ack) — агент продолжает работу.
3. Субагент работает в фоне со своим контекстом и своими инструментами.
4. Результат приходит обратно как «steer»-сообщение в очередь сессии.
5. Когда главный агент свободен, харнес будит его этим сообщением — ровно один раз.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    id: str
    kind: str
    input: dict
    status: str = "running"   # running | done | error
    result: str = ""
    started: float = field(default_factory=time.time)


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.inbox: "queue.Queue[dict]" = queue.Queue()   # steer-сообщения для главного агента
        self.delivered: set[str] = set()                   # exactly-once
        self._n = 0
        self._lock = threading.Lock()

    def spawn(self, kind: str, inp: dict, work: Callable[[dict], str]) -> str:
        with self._lock:
            self._n += 1
            jid = f"job_{self._n}"
        job = self.jobs[jid] = Job(jid, kind, inp)

        def run():
            try:
                job.result, job.status = work(inp), "done"
            except Exception as e:
                job.result, job.status = f"{type(e).__name__}: {e}", "error"
            # детерминированный id → повторная доставка распознаётся и отбрасывается
            self.inbox.put({"steer_id": f"steer_{jid}", "job": jid, "kind": kind,
                            "status": job.status, "result": job.result})

        threading.Thread(target=run, daemon=True).start()
        return jid

    def collect(self, timeout: float = 0.0) -> list[dict]:
        """Забрать готовые результаты. Дубликаты (тот же steer_id) отбрасываются."""
        out = []
        deadline = time.time() + timeout
        while True:
            remaining = max(0.0, deadline - time.time())
            try:
                msg = self.inbox.get(timeout=remaining) if remaining else self.inbox.get_nowait()
            except queue.Empty:
                break
            if msg["steer_id"] in self.delivered:
                continue
            self.delivered.add(msg["steer_id"])
            out.append(msg)
            if timeout and not self.pending():
                break
        return out

    def pending(self) -> list[str]:
        return [j.id for j in self.jobs.values() if j.status == "running"]


def steer_message(s: dict) -> dict:
    """Результат фоновой задачи как user-сообщение для главного агента."""
    return {"role": "user", "steer_id": s["steer_id"],
            "content": f"[ФОНОВАЯ ЗАДАЧА {s['job']} ({s['kind']}) завершена со статусом {s['status']}]\n{s['result']}"}
