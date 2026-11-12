"""Tool Calling, fallback-механизмы, провайдеры (моки httpx через respx), кэш и логирование вызовов."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from backend.core.config import Settings
from backend.core.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from backend.core.llm.manager import LLMManager, build_llm_manager, make_provider
from backend.core.llm.providers.api import AnthropicProvider, OpenAICompatibleProvider
from backend.core.llm.providers.mock import MockProvider, critic_policy, editor_policy, select_model_policy
from backend.core.llm.providers.ollama import OllamaProvider
from backend.core.llm.tools import default_registry
from backend.data.cache import MemoryCache
from backend.utils.tracing import MemoryRecorder, Tracer

MSGS = [Message(role="system", content="[ROLE:model_selector]"),
        Message(role="user", content='<context>{"available_tools": ["fit_random_forest"], "meta": {"n_rows": 1000}}</context>')]


class FailingProvider(LLMProvider):
    name = "failing"

    def __init__(self):
        super().__init__("bad")
        self.calls = 0

    async def chat(self, messages, **kw):
        self.calls += 1
        raise LLMError("down")


class TextProvider(LLMProvider):
    """Слабая модель: отвечает текстом, иногда с JSON."""
    name = "text"

    def __init__(self, replies):
        super().__init__("weak-0.5b")
        self.replies = list(replies)

    async def chat(self, messages, **kw):
        return LLMResponse(content=self.replies.pop(0), model=self.model, provider=self.name,
                           prompt_tokens=5, completion_tokens=5)


def tools(kind="model"):
    r = default_registry()
    return r.list(kind=kind, task_type="tabular", problem_type="classification")


async def test_call_tool_native_and_logging():
    rec = MemoryRecorder()
    llm = LLMManager([MockProvider()], recorder=rec, request_id="r")
    tr = Tracer("r", rec)
    async with tr.span("ModelSelector"):
        res = await llm.call_tool(MSGS, tools(), agent_name="ModelSelector")
    assert res.mode == "native" and res.call.name == "fit_random_forest" and res.attempts == 1
    call = rec.llm_calls[0]
    assert call["agent_name"] == "ModelSelector" and call["tokens_estimated"] and call["tokens_used"] > 0
    assert call["span_id"] is not None and tr.total_tokens == call["tokens_used"]


async def test_call_tool_text_fallback_then_retry():
    weak = TextProvider(["думаю, лучше лес", '{"name": "fit_logistic_regression", "arguments": {"C": 0.5}}'])
    res = await LLMManager([weak]).call_tool(MSGS, tools(), agent_name="ModelSelector")
    assert res.mode == "text" and res.attempts == 2 and res.call.arguments["C"] == 0.5
    assert res.errors and "attempt 1" in res.errors[0]


async def test_call_tool_invalid_args_and_unknown_tool_go_default():
    weak = TextProvider(['{"name": "fit_logistic_regression", "arguments": {"C": -5}}',
                         '{"name": "fit_magic", "arguments": {}}', "nothing"])
    res = await LLMManager([weak]).call_tool(MSGS, tools(), agent_name="ModelSelector", max_attempts=3)
    assert res.mode == "default" and res.call is None and len(res.errors) == 3


async def test_mock_fail_times_three_gives_default():
    res = await LLMManager([MockProvider(fail_times=3)]).call_tool(MSGS, tools(), agent_name="ModelSelector")
    assert res.mode == "default"
    res2 = await LLMManager([MockProvider(fail_times=2)]).call_tool(MSGS, tools(), agent_name="ModelSelector")
    assert res2.mode == "native" and res2.attempts == 3


async def test_multi_tool_calls_and_empty():
    llm = LLMManager([TextProvider(['{"tool_calls": []}'])])
    res = await llm.call_tool(MSGS, tools("editor"), agent_name="Editor", multi=True)
    assert res.calls == [] and res.mode == "text"


async def test_provider_fallback_chain_and_errors_logged():
    rec = MemoryRecorder()
    bad = FailingProvider()
    llm = LLMManager([bad, MockProvider()], recorder=rec, max_retries=1, backoff=0)
    resp = await llm.chat(MSGS, agent_name="x")
    assert resp.provider == "mock" and bad.calls == 2
    assert [c["status"] for c in rec.llm_calls] == ["error", "error", "ok"]
    with pytest.raises(LLMError):
        await LLMManager([FailingProvider()], max_retries=0).chat(MSGS, agent_name="x")
    res = await LLMManager([FailingProvider()], max_retries=0).call_tool(MSGS, tools(), agent_name="x")
    assert res.mode == "default"
    with pytest.raises(ValueError):
        LLMManager([])


async def test_cache_hit_recorded():
    rec, cache = MemoryRecorder(), MemoryCache()
    llm = LLMManager([MockProvider()], recorder=rec, cache=cache)
    await llm.chat(MSGS, agent_name="a")
    r2 = await llm.chat(MSGS, agent_name="a")
    assert r2.cached and rec.llm_calls[-1]["cached"] is True and sum(cache.hits.values()) == 1


async def test_structured_with_fallback():
    from backend.agents.critic import CriticVerdict

    msgs = [Message(role="system", content="[ROLE:critic]"),
            Message(role="user", content='<context>{"metrics": {"roc_auc": 0.9}, "model_type": "fit_random_forest"}</context>')]
    v, attempts = await LLMManager([MockProvider()]).structured(msgs, CriticVerdict, agent_name="Critic")
    assert v.score > 5 and attempts == 1
    weak = TextProvider(["не JSON", '{"score": 99}', '{"score": 4}'])
    v2, a2 = await LLMManager([weak]).structured(msgs, CriticVerdict, agent_name="Critic")
    assert v2.score == 4 and a2 == 3
    none, _ = await LLMManager([TextProvider(["x", "y", "z"])]).structured(msgs, CriticVerdict, agent_name="Critic")
    assert none is None
    v3, _ = await LLMManager([FailingProvider()], max_retries=0).structured(msgs, CriticVerdict, agent_name="Critic")
    assert v3 is None


def test_mock_policies():
    call = select_model_policy({"available_tools": ["fit_logistic_regression", "fit_random_forest"],
                                "rag_cases": [{"best_model": "fit_random_forest", "hyperparams": {"n_estimators": 100}}]})
    assert call.name == "fit_random_forest" and call.arguments["n_estimators"] == 100
    assert select_model_policy({"available_tools": ["fit_logistic_regression"], "meta": {"n_rows": 100}}).name == \
        "fit_logistic_regression"
    assert select_model_policy({"available_tools": ["fit_ridge_regression"], "meta": {"n_rows": 5000}}).name == \
        "fit_ridge_regression"
    v = critic_policy({"problem_type": "regression", "metrics": {"r2": 0.1},
                       "issues": [{"severity": "critical"}], "model_type": "fit_tabular_xgboost"})
    assert v["critical"] and v["score"] < 3
    calls = editor_policy({"issues": [{"code": "leakage"}, {"code": "low_quality"}, {"code": "poor_calibration"}],
                           "leaking_features": ["x"], "available_ops": ["score_dynamics", "ratios"],
                           "problem_type": "classification", "best_f1_threshold": 0.3, "decision_threshold": 0.5,
                           "model_type": "fit_tabular_xgboost"})
    assert [c.name for c in calls] == ["drop_features", "add_features", "change_threshold", "change_hyperparams"]
    sw = editor_policy({"issues": [{"code": "low_quality"}], "model_type": "fit_logistic_regression", "iteration": 1})
    assert sw[-1].name == "switch_model"


async def test_mock_roles():
    p = MockProvider()
    r = await p.chat([Message(role="system", content="[ROLE:explainer]"),
                      Message(role="user", content='<context>{"drivers": ["absences"]}</context>')])
    assert "absences" in r.content
    assert (await p.chat([Message(role="user", content="hi")])).content == "OK"
    r = await p.chat([Message(role="system", content="[ROLE:editor]"), Message(role="user", content="<context>{bad</context>")])
    assert json.loads(r.content) == {"tool_calls": []}


@respx.mock
async def test_ollama_provider():
    respx.post("http://ollama:11434/api/chat").mock(return_value=httpx.Response(200, json={
        "model": "qwen2.5:7b", "prompt_eval_count": 120, "eval_count": 15,
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fit_random_forest", "arguments": {"n_estimators": 200}}},
            {"function": {"name": "fit_ridge_regression", "arguments": '{"alpha": 2}'}}]}}))
    p = OllamaProvider("qwen2.5:7b", "http://ollama:11434")
    r = await p.chat([Message(role="user", content="x", tool_calls=[ToolCall(name="a", arguments={})])],
                     tools=[default_registry().get("fit_random_forest").schema()], json_mode=True)
    assert r.tool_calls[0].arguments == {"n_estimators": 200} and r.tool_calls[1].arguments == {"alpha": 2}
    assert r.prompt_tokens == 120 and r.provider == "ollama"
    body = json.loads(respx.calls.last.request.content)
    assert body["tools"] and body["format"] == "json"
    await p.aclose()


@respx.mock
async def test_ollama_error():
    respx.post("http://ollama:11434/api/chat").mock(return_value=httpx.Response(500))
    with pytest.raises(LLMError):
        await OllamaProvider("m", "http://ollama:11434").chat([Message(role="user", content="x")])


@respx.mock
async def test_openai_provider():
    respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json={
        "model": "gpt", "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        "choices": [{"message": {"content": None, "tool_calls": [
            {"function": {"name": "fit_ridge_regression", "arguments": "{\"alpha\": 3}"}},
            {"function": {"name": "x", "arguments": "{bad"}}]}}]}))
    p = OpenAICompatibleProvider("gpt", "key", "https://api.test/v1", name="groq")
    r = await p.chat([Message(role="tool", content="x"),
                      Message(role="assistant", content="", tool_calls=[ToolCall(name="t", arguments={"a": 1})])],
                     tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    assert r.tool_calls[0].arguments == {"alpha": 3} and r.tool_calls[1].arguments == {} and r.provider == "groq"
    respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError):
        await p.chat([Message(role="user", content="x")], json_mode=True)
    await p.aclose()


@respx.mock
async def test_anthropic_provider():
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json={
        "model": "claude", "usage": {"input_tokens": 8, "output_tokens": 4},
        "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "name": "fit_arima", "input": {"order": [1, 0, 1]}}]}))
    p = AnthropicProvider("claude", "k")
    r = await p.chat([Message(role="system", content="s"), Message(role="user", content="u")],
                     tools=[default_registry().get("fit_arima").schema()])
    assert r.content == "ok" and r.tool_calls[0].name == "fit_arima" and r.completion_tokens == 4
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(401))
    with pytest.raises(LLMError):
        await p.chat([Message(role="user", content="u")])
    await p.aclose()


def test_build_manager_and_factory():
    s = Settings(llm_provider="ollama", llm_fallback_providers=["mock"])
    m = build_llm_manager(s)
    assert [p.name for p in m.providers] == ["ollama", "mock"] and m.model_name.startswith("ollama:")
    assert make_provider("openai", s).name == "openai"
    assert make_provider("anthropic", s).name == "anthropic"
    with pytest.raises(ValueError):
        make_provider("nope", s)
