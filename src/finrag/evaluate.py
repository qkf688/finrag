"""评测：Recall@K、MRR@K、引用支撑度、延迟。"""

from __future__ import annotations

import time
from collections.abc import Callable

from .embed import tokenize


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def citation_support(answer: str, cited_texts: list[str]) -> float:
    """引用支撑度的近似：答案里能被引用片段覆盖的词占比。"""
    if not answer:
        return 0.0
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    cited_tokens: set[str] = set()
    for text in cited_texts:
        cited_tokens.update(tokenize(text))
    if not cited_tokens:
        return 0.0
    return len(answer_tokens & cited_tokens) / len(answer_tokens)


def evaluate_qa(
    items: list[dict],
    retrieve: Callable[[str], list[str]],
    *,
    k: int = 5,
) -> dict:
    """items: [{"question": str, "relevant_chunk_ids": [str, …]}]。"""
    recalls: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []
    # 相关 chunk 标的是「哪一篇文档」，不是某个精确切分块——自动生成的 QA
    # 只能保证答案在该文档里。因此同 doc_id 的任意 chunk 都算命中，
    # 否则切分参数一变，评测集就失效。
    relevant_docs: set[str] = set()
    for item in items:
        for cid in item.get("relevant_chunk_ids") or []:
            relevant_docs.add(cid.split("#")[0])

    for item in items:
        question = item["question"]
        relevant = set(item.get("relevant_chunk_ids") or [])
        start = time.perf_counter()
        retrieved = retrieve(question)
        # 自动生成的 QA 只标到文档级；同文档的任意 chunk 都算命中，
        # 这样切分参数变化不会让评测集失效。
        relevant |= {cid for cid in retrieved if cid.split("#")[0] in relevant_docs}
        latencies.append((time.perf_counter() - start) * 1000)
        r = recall_at_k(retrieved, relevant, k)
        m = mrr_at_k(retrieved, relevant, max(k, 10))
        recalls.append(r)
        mrrs.append(m)
        details.append(
            {
                "question": question,
                "retrieved": retrieved[:k],
                "relevant": sorted(relevant),
                "recall_at_k": r,
                "mrr_at_k": m,
            }
        )
    n = len(items) or 1
    latencies_sorted = sorted(latencies)
    if latencies_sorted:
        idx = min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))
        p95 = latencies_sorted[idx]
    else:
        p95 = 0.0
    return {
        "count": len(items),
        "recall_at_k": sum(recalls) / n,
        "mrr_at_k": sum(mrrs) / n,
        "latency_p50_ms": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0,
        "latency_p95_ms": p95,
        "details": details,
    }
