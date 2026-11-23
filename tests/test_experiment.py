from finrag.embed import HashEmbedding
from finrag.experiment import render_markdown, run_experiments


def _toy():
    texts = [
        "安重工 2024 年营业收入 20.5 亿元，同比增长 12.0%。",
        "瑞材料 2024 年研发投入占营业收入比重 5.2%。",
        "华智造 2024 年营业收入 9.8 亿元，同比增长 3.1%。",
    ]
    ids = ["a#经营情况#0", "b#研发投入#0", "c#经营情况#0"]
    items = [
        {"question": "安重工 2024 年营业收入是多少？", "relevant_chunk_ids": ["a#经营情况#0"]},
        {"question": "瑞材料研发投入占比是多少？", "relevant_chunk_ids": ["b#研发投入#0"]},
    ]
    return texts, ids, items


def test_run_experiments_returns_four_modes():
    texts, ids, items = _toy()
    results = run_experiments(texts, ids, items, HashEmbedding(128), k=2)
    assert [r["mode"] for r in results] == ["bm25", "vector", "hybrid", "hybrid+rerank"]
    for r in results:
        assert r["count"] == 2
        assert 0.0 <= r["recall@2"] <= 1.0
        assert 0.0 <= r["mrr@10"] <= 1.0


def test_render_markdown_has_header_and_rows():
    texts, ids, items = _toy()
    table = render_markdown(run_experiments(texts, ids, items, HashEmbedding(64), k=2), k=2)
    assert "Recall@2" in table
    assert table.count("\n") >= 4
