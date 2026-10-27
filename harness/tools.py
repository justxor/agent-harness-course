"""Инструменты агента: описание (JSON Schema) + функция + политика безопасности."""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    needs_approval: bool = False      # спросить человека перед запуском
    client_side: bool = False         # выполняется не сервером, а «клиентом» (UI / человек)
    terminates: bool = False          # успешный вызов завершает цикл (передача хода, финальный отчёт)
    tags: set[str] = field(default_factory=set)

    def schema(self) -> dict:
        """То, что уходит в модель. Сама функция модели не видна никогда."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(description: str, *, needs_approval: bool = False, client_side: bool = False,
         terminates: bool = False, params: dict[str, str] | None = None) -> Callable[[Callable], Tool]:
    """Декоратор: строит JSON Schema из сигнатуры функции.

        @tool("Сложить два числа", params={"a": "первое", "b": "второе"})
        def add(a: float, b: float) -> str: ...
    """
    params = params or {}

    def wrap(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        props, required = {}, []
        for name, p in sig.parameters.items():
            t = hints.get(name, str)
            origin = typing.get_origin(t) or t
            prop = {"type": _PY_TO_JSON.get(origin, "string")}
            if name in params:
                prop["description"] = params[name]
            props[name] = prop
            if p.default is inspect.Parameter.empty:
                required.append(name)
        schema = {"type": "object", "properties": props, "required": required}
        return Tool(fn.__name__, description, schema, fn, needs_approval, client_side, terminates)

    return wrap


def validate_args(t: Tool, args: dict) -> str | None:
    """Минимальная валидация. Возвращает текст ошибки — его увидит модель и исправится."""
    if "__invalid_json__" in args:
        return f"аргументы не являются валидным JSON: {args['__invalid_json__']!r}"
    props = t.parameters.get("properties", {})
    for r in t.parameters.get("required", []):
        if r not in args:
            return f"нет обязательного аргумента '{r}'"
    for k, v in args.items():
        if k not in props:
            return f"неизвестный аргумент '{k}'. Допустимые: {sorted(props)}"
        want = props[k].get("type")
        ok = {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool),
              "number": isinstance(v, (int, float)) and not isinstance(v, bool),
              "boolean": isinstance(v, bool), "array": isinstance(v, list),
              "object": isinstance(v, dict)}.get(want, True)
        if not ok:
            return f"аргумент '{k}' должен быть {want}, получено {type(v).__name__}"
    return None


# ---------------------------------------------------------------------------
# Песочница для файловых инструментов
# ---------------------------------------------------------------------------

class Workspace:
    """Все пути — только внутри root. Классическая защита от '../../etc/passwd'."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"путь '{rel}' выходит за пределы рабочей папки")
        return p

    def tools(self) -> list[Tool]:
        ws = self

        @tool("Прочитать текстовый файл из рабочей папки", params={"path": "относительный путь"})
        def read_file(path: str) -> str:
            p = ws.resolve(path)
            if not p.exists():
                return f"файл '{path}' не найден. Есть: {[x.name for x in ws.root.iterdir()]}"
            text = p.read_text(encoding="utf-8", errors="replace")
            return text if len(text) < 20_000 else text[:20_000] + "\n…[обрезано]"

        @tool("Список файлов в рабочей папке")
        def list_files() -> str:
            return "\n".join(sorted(str(p.relative_to(ws.root)) for p in ws.root.rglob("*") if p.is_file())) or "(пусто)"

        @tool("Записать текстовый файл в рабочую папку (перезапишет существующий)",
              needs_approval=True, params={"path": "относительный путь", "content": "содержимое"})
        def write_file(path: str, content: str) -> str:
            p = ws.resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"записано {len(content)} символов в {path}"

        return [read_file, list_files, write_file]
