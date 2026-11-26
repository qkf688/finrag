"""llm.py 的纯逻辑测试：payload 形状与响应解析，不发网络请求。"""

import pytest

from finrag.answer import llm_answer
from finrag.config import Settings
from finrag.llm import LLMError, build_payload, parse_response


def _settings(**kw) -> Settings:
    base = {"llm_model": "test/model", "llm_max_tokens": 128}
    base.update(kw)
    return Settings(**base)


def test_payload_carries_model_prompt_and_system_rules():
    payload = build_payload("【资料片段】\n[a]\n正文", _settings(llm_api_key="k"))
    assert payload["model"] == "test/model"
    assert payload["max_tokens"] == 128
    assert payload["messages"][0]["role"] == "system"
    assert "只依据【资料片段】回答问题" in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"].endswith("正文")


def test_parse_response_returns_stripped_content():
    data = {"choices": [{"message": {"content": '  {"answer": "42", "citations": []} '}}]}
    assert parse_response(data) == '{"answer": "42", "citations": []}'


def test_parse_response_rejects_empty_content():
    data = {"choices": [{"message": {"content": None}}]}
    with pytest.raises(LLMError, match="未返回内容"):
        parse_response(data)


def test_parse_response_rejects_unexpected_shape():
    with pytest.raises(LLMError, match="OpenAI 兼容"):
        parse_response({"error": "bad"})


def test_settings_read_key_fallback(monkeypatch):
    monkeypatch.delenv("FINRAG_LLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert Settings.from_env().llm_api_key == "tok"


def test_llm_answer_accepts_supported_answer():
    text, refused = llm_answer({"answer": "上限 15 亿元。", "citations": ["em-002#root#1"]})
    assert (text, refused) == ("上限 15 亿元。", False)


def test_llm_answer_treats_missing_citations_as_refusal():
    text, refused = llm_answer({"answer": "上限 15 亿元。", "citations": []})
    assert refused is True
    assert text == "知识库中没有相关信息"