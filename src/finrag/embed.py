"""向量化后端。

- HashEmbedding：确定性、零依赖、离线可跑，作为基线与测试替身。
- LocalEmbedding：调用本地 sentence-transformers 模型（首次使用需已下载模型）。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_WORD_RE = re.compile(r"[a-z0-9]+|[一-鿿]+")

# BGE 中文检索模型要求 query 侧加指令前缀；document 侧不加。
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def tokenize(text: str) -> list[str]:
    """英文按词切；中文按「单字 + 相邻字二元组」切。"""
    out: list[str] = []
    for m in _WORD_RE.finditer(text.lower()):
        tok = m.group(0)
        if "一" <= tok[0] <= "鿿":
            out.extend(list(tok))
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        else:
            out.append(tok)
    return out


class EmbeddingBackend(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedding:
    """哈希基线：token → 维度桶 → L2 归一化。同一输入永远同一向量。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)


class LocalEmbedding:
    """本地 embedding 模型；模型没下载时给出可执行的提示，而不是静默降级。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 512

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - 依赖缺失路径
                raise RuntimeError(
                    "缺少 sentence-transformers；先跑 `pip install sentence-transformers`，"
                    "或用 HashEmbedding 作为离线基线。"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            # sentence-transformers 6.x 把方法改名为 get_embedding_dimension；
            # 两个名字都判一下，兼容 3.x–6.x。
            dim_fn = getattr(
                self._model, "get_embedding_dimension", None
            ) or self._model.get_sentence_embedding_dimension
            self.dim = int(dim_fn())
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """BGE 检索：query 加指令前缀，document 不加。"""
        return self.encode([_QUERY_PREFIX + t for t in texts])


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
