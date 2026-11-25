"""索引构建：切分语料并建 BM25 + 向量索引。CLI 与 Web 服务共用同一入口。"""

from __future__ import annotations

from .config import Settings
from .embed import HashEmbedding, LocalEmbedding
from .retrieve import BM25, VectorIndex
from .split import load_corpus


class IndexBuildError(RuntimeError):
    """语料为空或索引无法构建。"""


def build_index(settings: Settings, backend: str):
    """返回 (chunks, ids, texts, bm25, vec, emb)。

    backend: ``hash`` 为离线确定性基线，``local`` 用本地 sentence-transformers 模型。
    """
    chunks = load_corpus(
        settings.corpus_dir,
        max_chars=settings.chunk_chars,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise IndexBuildError("语料为空：先在 corpus/ 放 .md 文档")
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    if backend == "local":
        emb = LocalEmbedding(settings.embedding_model)
    else:
        emb = HashEmbedding(settings.embedding_dim)
    vectors = emb.encode(texts)
    vec = VectorIndex().fit(ids, vectors)
    bm25 = BM25().fit(texts)
    return chunks, ids, texts, bm25, vec, emb
