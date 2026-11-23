"""检索层：BM25、向量索引、混合召回（RRF 融合）。"""

from __future__ import annotations

import math
from collections import Counter

from .embed import cosine, tokenize


class BM25:
    """Okapi BM25，纯 Python 实现，零依赖。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0

    def fit(self, docs: list[str]) -> BM25:
        self.docs = [tokenize(d) for d in docs]
        self.df = Counter()
        for toks in self.docs:
            for tok in set(toks):
                self.df[tok] += 1
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        return self

    def score(self, query: str) -> list[float]:
        if not self.docs:
            return []
        n = len(self.docs)
        scores: list[float] = []
        q_tokens = tokenize(query)
        for toks in self.docs:
            tf = Counter(toks)
            dl = len(toks) or 1
            total = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                idf = math.log(1 + (n - self.df[tok] + 0.5) / (self.df[tok] + 0.5))
                freq = tf[tok]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                total += idf * freq * (self.k1 + 1) / (denom or 1)
            scores.append(total)
        return scores


class VectorIndex:
    """暴力余弦检索（语料规模到十万 chunk 之前够用，换 ANN 只需替换本类）。"""

    def __init__(self) -> None:
        self.vectors: list[list[float]] = []
        self.ids: list[str] = []

    def fit(self, ids: list[str], vectors: list[list[float]]) -> VectorIndex:
        self.ids = list(ids)
        self.vectors = list(vectors)
        return self

    def search(self, query_vec: list[float], k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, cosine(query_vec, v)) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def lexical_overlap(query: str, text: str) -> float:
    """查询词被文本覆盖的比例，用于重排（0–1）。"""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：把多路排名按 1/(k+rank) 合并。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    chunk_texts: list[str],
    chunk_ids: list[str],
    *,
    bm25: BM25 | None = None,
    vec_scores: list[tuple[int, float]] | None = None,
    k: int = 5,
) -> list[tuple[str, float, dict[str, float]]]:
    """混合召回：BM25 排名 + 向量排名 RRF 融合。返回 (chunk_id, 融合分, 各路原始分)。"""
    bm = bm25 or BM25().fit(chunk_texts)
    bm_scores = bm.score(query)
    # BM25 是对数似然量纲，与余弦不同量级；用 s/(s+k) 压到 0–1，
    # 保留绝对大小信息（不按自身最大值归一化，否则任何查询的最高分都是 1，永远拒不了答）。
    bm_norm = [s / (s + 10.0) for s in bm_scores]
    bm_rank = [i for i, _ in sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)]

    vec_rank: list[int] = []
    if vec_scores:
        vec_rank = [i for i, _ in sorted(vec_scores, key=lambda x: x[1], reverse=True)]

    fused = rrf_fuse([r for r in (bm_rank, vec_rank) if r])
    out: list[tuple[str, float, dict[str, float]]] = []
    for doc_id, score in fused[:k]:
        out.append(
            (
                chunk_ids[doc_id],
                score,
                {
                    "bm25": bm_norm[doc_id] if doc_id < len(bm_norm) else 0.0,
                    "vector": next((s for i, s in (vec_scores or []) if i == doc_id), 0.0),
                },
            )
        )
    return out
