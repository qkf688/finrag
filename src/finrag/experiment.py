"""检索实验：BM25 / 向量 / 混合 / 混合+重排 四组对照。"""

from __future__ import annotations

import time

from .answer import assemble_answer, extractive_answer
from .evaluate import citation_support, mrr_at_k, recall_at_k
from .retrieve import BM25, VectorIndex, hybrid_search, lexical_overlap


def _rank_bm25(question: str, texts: list[str], ids: list[str], bm25: BM25, k: int) -> list[str]:
    scores = bm25.score(question)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in order[:k]]


def _rank_vector(question: str, vec: VectorIndex, ids: list[str], emb, k: int) -> list[str]:
    qv = emb.encode_queries([question])[0]
    return [ids[i] for i, _ in vec.search(qv, k)]


def _rank_hybrid(
    question: str,
    texts: list[str],
    ids: list[str],
    bm25: BM25,
    vec: VectorIndex,
    emb,
    k: int,
    top_n: int,
):
    qv = emb.encode_queries([question])[0]
    return hybrid_search(
        question,
        texts,
        ids,
        bm25=bm25,
        vec_scores=vec.search(qv, top_n),
        k=top_n,
    )


def _rerank(question: str, ranked, texts_by_id: dict[str, str], k: int):
    """语义+词面重排：0.6×向量余弦 + 0.4×查询词覆盖。

    刻意不用 BM25 原始分——RRF 的候选已经由 BM25 排过一遍，再用 BM25 打分
    等于同一信号计两次，会打乱顺序（实测 MRR 从 0.727 掉到 0.662）。
    """
    out = []
    for cid, _, raw in ranked:
        vec = raw.get("vector", 0.0)
        lex = lexical_overlap(question, texts_by_id.get(cid, ""))
        out.append((cid, 0.6 * vec + 0.4 * lex))
    out.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in out[:k]]


def run_experiments(
    texts: list[str],
    ids: list[str],
    items: list[dict],
    emb,
    *,
    k: int = 5,
    top_n: int = 10,
) -> list[dict]:
    bm25 = BM25().fit(texts)
    vec = VectorIndex().fit(ids, emb.encode(texts))
    texts_by_id = dict(zip(ids, texts, strict=False))

    modes = {
        "bm25": lambda q: _rank_bm25(q, texts, ids, bm25, k),
        "vector": lambda q: _rank_vector(q, vec, ids, emb, k),
        "hybrid": lambda q: [
            cid for cid, _, _ in _rank_hybrid(q, texts, ids, bm25, vec, emb, k, top_n)
        ][:k],
        "hybrid+rerank": lambda q: _rerank(
            q, _rank_hybrid(q, texts, ids, bm25, vec, emb, k, top_n), texts_by_id, k
        ),
    }

    results: list[dict] = []
    for mode, run in modes.items():
        recalls: list[float] = []
        mrrs: list[float] = []
        latencies: list[float] = []
        for item in items:
            relevant = set(item.get("relevant_chunk_ids") or [])
            start = time.perf_counter()
            retrieved = run(item["question"])
            latencies.append((time.perf_counter() - start) * 1000)
            recalls.append(recall_at_k(retrieved, relevant, k))
            mrrs.append(mrr_at_k(retrieved, relevant, max(k, 10)))
        latencies.sort()
        results.append(
            {
                "mode": mode,
                "count": len(items),
                f"recall@{k}": round(sum(recalls) / (len(items) or 1), 4),
                "mrr@10": round(sum(mrrs) / (len(items) or 1), 4),
                "p50_ms": round(latencies[len(latencies) // 2], 3) if latencies else 0.0,
                "p95_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3)
                if latencies
                else 0.0,
            }
        )
    return results


def evaluate_refusal(
    answerable: list[dict],
    unanswerable: list[dict],
    search,
    *,
    refuse_score: float = 0.15,
    chunk_texts: dict[str, str] | None = None,
) -> dict:
    """误拒率（可答问题被拒）与拒答率（不可答问题被拒）。

    带 chunk_texts 时启用实体门禁：问句里的主体名必须出现在引用片段中，
    否则即使分数够也拒答——金融问答里「同名指标不同公司」靠分数是挡不住的。
    """

    def refused(question: str) -> bool:
        return assemble_answer(
            question,
            search(question),
            refuse_score=refuse_score,
            require_entity=chunk_texts is not None,
            chunk_texts=chunk_texts,
        )["refused"]

    false_refusal = sum(1 for it in answerable if refused(it["question"])) / (len(answerable) or 1)
    correct_refusal = sum(1 for it in unanswerable if refused(it["question"])) / (
        len(unanswerable) or 1
    )
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "false_refusal_rate": round(false_refusal, 4),
        "unanswerable_refusal_rate": round(correct_refusal, 4),
    }


def evaluate_faithfulness(
    answerable: list[dict],
    search,
    chunk_texts: dict[str, str],
    *,
    refuse_score: float = 0.15,
) -> dict:
    """抽取式回答的引用支撑度：答案里的词有多少能在引用片段中找到。"""
    supports: list[float] = []
    refused = 0
    for item in answerable:
        ranked = search(item["question"])
        material = [(cid, chunk_texts[cid]) for cid, _, _ in ranked if cid in chunk_texts]
        answer_text = extractive_answer(item["question"], material)
        out = assemble_answer(
            item["question"], ranked, refuse_score=refuse_score, answer_text=answer_text
        )
        if out["refused"]:
            refused += 1
            continue
        cited = [chunk_texts[c] for c in out["citations"] if c in chunk_texts]
        supports.append(citation_support(out["answer"], cited))
    return {
        "count": len(answerable),
        "refused": refused,
        "mean_citation_support": round(sum(supports) / (len(supports) or 1), 4),
    }


def render_markdown(results: list[dict], *, k: int = 5) -> str:
    header = (
        f"| 方案 | Recall@{k} | MRR@10 | P50 (ms) | P95 (ms) |\n"
        "|---|---:|---:|---:|---:|"
    )
    rows = [
        f"| {r['mode']} | {r[f'recall@{k}']:.3f} | {r['mrr@10']:.3f} "
        f"| {r['p50_ms']:.3f} | {r['p95_ms']:.3f} |"
        for r in results
    ]
    return "\n".join([header, *rows])
