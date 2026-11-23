from finrag.embed import HashEmbedding, cosine, tokenize
from finrag.retrieve import BM25, VectorIndex, hybrid_search, rrf_fuse

DOCS = ["基金季报显示规模增长", "公司发布股份回购公告", "行业景气度回升"]


def test_tokenize_chinese_bigrams():
    tokens = tokenize("回购")
    assert "回" in tokens and "购" in tokens and "回购" in tokens


def test_bm25_ranks_relevant_first():
    scores = BM25().fit(DOCS).score("回购公告")
    assert scores[1] == max(scores)


def test_hash_embedding_is_deterministic_and_normalized():
    emb = HashEmbedding(dim=64)
    v1 = emb.encode(["回购公告"])[0]
    v2 = emb.encode(["回购公告"])[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9


def test_cosine_of_same_vector_is_one():
    emb = HashEmbedding(dim=32)
    v = emb.encode(["规模增长"])[0]
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_rrf_fuse_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    assert fused[0][0] == 0


def test_hybrid_ranks_relevant_chunk_first():
    chunks = ["基金季报显示规模增长", "公司发布股份回购公告"]
    ids = ["c0", "c1"]
    emb = HashEmbedding(dim=128)
    vec = VectorIndex().fit(ids, emb.encode(chunks))
    ranked = hybrid_search(
        "回购公告",
        chunks,
        ids,
        bm25=BM25().fit(chunks),
        vec_scores=vec.search(emb.encode(["回购公告"])[0], 2),
        k=2,
    )
    assert ranked[0][0] == "c1"
    assert "bm25" in ranked[0][2] and "vector" in ranked[0][2]
