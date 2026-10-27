"""Детерминированные тесты харнеса: без ключей и без интернета.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import (AnthropicLLM, FactMemory, JobRunner, Ledger, OpenAICompatLLM, Reply,  # noqa: E402
                     ScriptedLLM, SessionStore, ToolCall, Workspace, compact, repair_dangling_tool_calls,
                     run_agent, select_cutoff, tool, validate_args)

quiet = dict(on_event=lambda k, d: None)


@tool("сложить", params={"a": "a", "b": "b"})
def add(a: int, b: int) -> str:
    return str(a + b)


@tool("упасть")
def boom() -> str:
    raise ValueError("сломалось")


@tool("опасно", needs_approval=True)
def danger() -> str:
    return "сделано"


@tool("передать ход", terminates=True)
def handoff() -> str:
    return "ок"


class LoopTests(unittest.TestCase):
    def test_tool_roundtrip(self):
        llm = ScriptedLLM([Reply("", [ToolCall("1", "add", {"a": 2, "b": 3})]), Reply("5")])
        res = run_agent(llm, "", [{"role": "user", "content": "2+3"}], [add], **quiet)
        self.assertEqual(res.text, "5")
        self.assertEqual(res.stopped_by, "done")
        # модель на втором шаге видела результат инструмента
        self.assertEqual(llm.calls[1]["messages"][-1], {"role": "tool", "tool_call_id": "1", "content": "5"})

    def test_errors_become_tool_results(self):
        llm = ScriptedLLM([Reply("", [ToolCall("1", "nope", {}), ToolCall("2", "boom", {}),
                                      ToolCall("3", "add", {"a": "x", "b": 1})]), Reply("ок")])
        res = run_agent(llm, "", [{"role": "user", "content": "?"}], [add, boom], **quiet)
        results = [m["content"] for m in res.messages if m["role"] == "tool"]
        self.assertIn("инструмента 'nope' нет", results[0])
        self.assertIn("ValueError", results[1])
        self.assertIn("должен быть integer", results[2])

    def test_approval_denied(self):
        llm = ScriptedLLM([Reply("", [ToolCall("1", "danger", {})]), Reply("понял")])
        res = run_agent(llm, "", [{"role": "user", "content": "!"}], [danger], approve=lambda t, a: False, **quiet)
        self.assertTrue(res.tool_log[0]["result"].startswith("ОТКАЗАНО"))

    def test_max_steps(self):
        llm = ScriptedLLM(lambda s, m, t: Reply("", [ToolCall(str(len(m)), "add", {"a": 1, "b": 1})]))
        res = run_agent(llm, "", [{"role": "user", "content": "∞"}], [add], max_steps=3, **quiet)
        self.assertEqual(res.stopped_by, "max_steps")

    def test_terminating_tool(self):
        llm = ScriptedLLM([Reply("", [ToolCall("1", "handoff", {})])])
        res = run_agent(llm, "", [{"role": "user", "content": "go"}], [handoff], **quiet)
        self.assertEqual(res.stopped_by, "terminated")

    def test_client_side_tool(self):
        @tool("спросить", client_side=True, params={"q": "вопрос"})
        def ask(q: str) -> str: ...
        llm = ScriptedLLM([Reply("", [ToolCall("1", "ask", {"q": "как дела?"})]), Reply("ясно")])
        res = run_agent(llm, "", [{"role": "user", "content": "x"}], [ask],
                        client_handler=lambda t, a: "хорошо", **quiet)
        self.assertEqual(res.tool_log[0]["result"], "хорошо")


class ContextTests(unittest.TestCase):
    def history(self):
        h = []
        for i in range(6):
            h += [{"role": "user", "content": "в" * 200},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "name": "r", "arguments": {}}]},
                  {"role": "tool", "tool_call_id": f"t{i}", "content": "р" * 400},
                  {"role": "assistant", "content": "ок"}]
        return h

    def test_cutoff_never_on_tool(self):
        h = self.history()
        for budget in range(10, 1500, 7):
            cut = select_cutoff(h, budget)
            if cut:
                self.assertNotEqual(h[cut]["role"], "tool", budget)

    def test_compact_shrinks(self):
        h = self.history()
        out = compact(ScriptedLLM([Reply("сводка")]), h, keep_tokens=200)
        self.assertLess(len(out), len(h))
        self.assertTrue(out[0].get("compaction"))

    def test_repair(self):
        h = [{"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "x", "arguments": {}},
                                                                {"id": "b", "name": "y", "arguments": {}}]},
             {"role": "tool", "tool_call_id": "a", "content": "1"},
             {"role": "user", "content": "?"}]
        fixed = repair_dangling_tool_calls(h)
        self.assertEqual([m["role"] for m in fixed], ["assistant", "tool", "tool", "user"])
        self.assertIn("INTERRUPTED", fixed[2]["content"])


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_ledger_tolerates_corruption(self):
        lg = Ledger(self.tmp / "l.jsonl")
        lg.append({"op": "add", "id": "1", "text": "a"})
        with lg.path.open("a") as f:
            f.write('{"op": "add", "id": "2", "te')   # оборванная запись
        self.assertEqual(len(lg.read()), 1)

    def test_fact_dedupe_and_forget(self):
        m = FactMemory(Ledger(self.tmp / "f.jsonl"))
        self.assertTrue(m.add("Любит Go", "s"))
        self.assertFalse(m.add("любит  go!", "s"))
        fid = next(iter(m.facts()))
        m.forget(fid)
        self.assertEqual(m.facts(), {})

    def test_session_store_rejects_bad_id(self):
        with self.assertRaises(ValueError):
            SessionStore(self.tmp).path("../evil")

    def test_workspace_sandbox(self):
        ws = Workspace(self.tmp / "ws")
        with self.assertRaises(PermissionError):
            ws.resolve("../secret")

    def test_jobs_exactly_once(self):
        r = JobRunner()
        r.spawn("k", {}, lambda inp: "готово")
        got = r.collect(timeout=2)
        self.assertEqual(len(got), 1)
        r.inbox.put(dict(got[0]))
        self.assertEqual(r.collect(), [])


class WireTests(unittest.TestCase):
    MSGS = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "add", "arguments": {"a": 1, "b": 2}},
                                                                 {"id": "2", "name": "add", "arguments": {"a": 3, "b": 4}}]},
            {"role": "tool", "tool_call_id": "1", "content": "3"},
            {"role": "tool", "tool_call_id": "2", "content": "7"}]

    def test_anthropic_merges_tool_results(self):
        wire = AnthropicLLM._to_wire(self.MSGS)
        self.assertEqual([m["role"] for m in wire], ["user", "assistant", "user"])
        self.assertEqual(len(wire[2]["content"]), 2)

    def test_openai_shape(self):
        wire = OpenAICompatLLM._to_wire("sys", self.MSGS)
        self.assertEqual(wire[0]["role"], "system")
        self.assertEqual(json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]), {"a": 1, "b": 2})

    def test_validate(self):
        self.assertIsNone(validate_args(add, {"a": 1, "b": 2}))
        self.assertIn("нет обязательного", validate_args(add, {"a": 1}))


class FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for w in ["При", "вет"]:
                self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": w}}]})}\n\n'.encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if body.get("tools") and body["messages"][-1]["role"] == "user":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 20, "b": 22}'}}]}
        else:
            msg = {"role": "assistant", "content": "Ответ: " + body["messages"][-1]["content"]}
        data = {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        out = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.llm = OpenAICompatLLM("fake", "", f"http://127.0.0.1:{cls.srv.server_port}/v1")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_full_loop_over_http(self):
        res = run_agent(self.llm, "s", [{"role": "user", "content": "20+22"}], [add], **quiet)
        self.assertEqual(res.text, "Ответ: 42")
        self.assertEqual(res.usage.input_tokens, 20)

    def test_stream(self):
        self.assertEqual("".join(self.llm.stream_text("", [{"role": "user", "content": "x"}])), "Привет")


if __name__ == "__main__":
    unittest.main()
