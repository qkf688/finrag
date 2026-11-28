"""Web 接口测试：页面可达、查询返回引用、拒答与参数校验。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from finrag.serve import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "FinRAG" in resp.text


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "backends": ["hash", "local"]}


def test_query_returns_answer_and_citations(client):
    resp = client.post("/api/query", json={"question": "华友钴业回购计划的上限是多少？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is False
    assert data["answer"]
    assert data["citations"]
    first = data["citations"][0]
    assert {
        "chunk_id",
        "doc_id",
        "title",
        "section",
        "source",
        "text",
        "score",
        "bm25",
    } <= first.keys()
    assert data["elapsed_ms"] >= 0


def test_query_refuses_unknown_entity(client):
    resp = client.post("/api/query", json={"question": "星海量子 2024 年营业收入是多少？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["refuse_reason"] == "entity"
    assert data["citations"] == []
    assert data["top_score"] > data["refuse_score"]


def test_top_k_limits_citations(client):
    resp = client.post(
        "/api/query",
        json={"question": "华友钴业回购计划的上限是多少？", "top_k": 1},
    )
    assert resp.status_code == 200
    assert len(resp.json()["citations"]) <= 1


def test_query_with_llm_uses_generated_answer(client, monkeypatch):
    monkeypatch.setattr(
        "finrag.serve.chat",
        lambda prompt, settings: (
            '{"answer": "回购上限为 15 亿元。", "citations": ["em-002#root#1"]}'
        ),
    )
    resp = client.post(
        "/api/query",
        json={"question": "华友钴业回购计划的上限是多少？", "use_llm": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is False
    assert data["answer"] == "回购上限为 15 亿元。"


def test_query_with_llm_refuses_when_model_gives_no_citation(client, monkeypatch):
    monkeypatch.setattr(
        "finrag.serve.chat",
        lambda prompt, settings: '{"answer": "资料不足。", "citations": []}',
    )
    resp = client.post(
        "/api/query",
        json={"question": "华友钴业回购计划的上限是多少？", "use_llm": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["refuse_reason"] == "llm"
    assert data["citations"] == []
    assert data["answer"] == "知识库中没有相关信息"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x", "top_k": 0},
        {"question": "x", "top_k": 11},
        {"question": "x", "backend": "unknown"},
    ],
)
def test_invalid_requests(client, payload):
    assert client.post("/api/query", json=payload).status_code == 422
