"""最小 Web 演示：FastAPI + 单页 HTML，复用检索与回答流程。

启动：
    pip install -e ".[serve]"
    python -m uvicorn finrag.serve:app --port 8000
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import assemble_answer, build_prompt, extractive_answer, llm_answer, parse_model_json
from .config import Settings
from .index import IndexBuildError, build_index
from .llm import LLMError, chat
from .retrieve import hybrid_search

WEB_DIR = Path(__file__).with_name("web")

app = FastAPI(title="FinRAG", description="金融知识库检索演示", version="1.0.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=200)
    backend: Literal["hash", "local"] = "hash"
    top_k: int | None = Field(default=None, ge=1, le=10)
    use_llm: bool = False


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str
    text: str
    score: float
    bm25: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    backend: str
    elapsed_ms: float
    index_ms: float
    top_score: float
    refuse_score: float
    refuse_reason: Literal["score", "entity", "llm"] | None = None
    citations: list[Citation]


@lru_cache(maxsize=2)
def load_index(backend: str):
    """按后端缓存索引；首次请求承担构建成本。

    ``local`` 后端缺依赖（RuntimeError）或模型下载失败（OSError，
    huggingface_hub 的 RepositoryNotFoundError 属于这一类）时，
    统一转成 IndexBuildError，让接口回 503 而不是 500。
    """
    try:
        return build_index(Settings.from_env(), backend)
    except IndexBuildError:
        raise
    except (RuntimeError, OSError) as exc:
        raise IndexBuildError(f"{type(exc).__name__}: {exc}") from exc


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "backends": ["hash", "local"]}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = Settings.from_env()
    k = req.top_k or settings.top_k

    t0 = time.perf_counter()
    try:
        chunks, ids, texts, bm25, _vec, _emb = load_index(req.backend)
    except IndexBuildError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    t1 = time.perf_counter()

    # 检索用 BM25 单路。默认 top_k 下 RRF 融合的候选集合恒等于向量候选
    #（每条向量候选至少 1/(60+5)+1/(60+375) ≈ 0.018 > BM25 第 1 名的 1/61 ≈ 0.016），
    # 而本仓库的向量是词袋哈希，质量低于 BM25（同语料 recall@5 0.550 vs 0.775，
    # 见 README「为什么混合召回低于 BM25」与 2026-11-28 生成侧审计）。
    ranked = hybrid_search(req.question, texts, ids, bm25=bm25, k=k)
    by_id = {c.chunk_id: c for c in chunks}
    text_by_id = {c.chunk_id: c.text for c in chunks}
    material = [(cid, by_id[cid].text) for cid, _, _ in ranked if cid in by_id]
    answer_text = extractive_answer(req.question, material)
    if req.use_llm:
        try:
            answer_text, llm_refused = llm_answer(
                parse_model_json(chat(build_prompt(req.question, material), settings))
            )
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if llm_refused:
            t2 = time.perf_counter()
            top = max((max(s.values()) for _, _, s in ranked), default=0.0)
            return QueryResponse(
                question=req.question,
                answer=answer_text,
                refused=True,
                backend=req.backend,
                elapsed_ms=round((t2 - t1) * 1000, 1),
                index_ms=round((t1 - t0) * 1000, 1),
                top_score=round(top, 4),
                refuse_score=settings.refuse_score,
                refuse_reason="llm",
                citations=[],
            )
    result = assemble_answer(
        req.question,
        ranked,
        refuse_score=settings.refuse_score,
        answer_text=answer_text,
        require_entity=True,
        chunk_texts=text_by_id,
    )
    t2 = time.perf_counter()

    raw = {cid: (score, scores) for cid, score, scores in ranked}
    citations: list[Citation] = []
    for cid in result["citations"]:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        score, scores = raw[cid]
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                section=chunk.section,
                source=chunk.source,
                text=chunk.text,
                score=round(score, 5),
                bm25=round(scores.get("bm25", 0.0), 4),
            )
        )

    top_score = max((max(scores.values()) for _, _, scores in ranked), default=0.0)
    refuse_reason: Literal["score", "entity", "llm"] | None = None
    if result["refused"]:
        refuse_reason = "score" if top_score < settings.refuse_score else "entity"

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        refused=result["refused"],
        backend=req.backend,
        elapsed_ms=round((t2 - t1) * 1000, 1),
        index_ms=round((t1 - t0) * 1000, 1),
        top_score=round(top_score, 4),
        refuse_score=settings.refuse_score,
        refuse_reason=refuse_reason,
        citations=citations,
    )
