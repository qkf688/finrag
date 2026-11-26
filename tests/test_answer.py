from finrag.answer import (
    assemble_answer,
    build_prompt,
    extract_entity,
    extractive_answer,
    parse_model_json,
)
from finrag.evaluate import citation_support, evaluate_qa, mrr_at_k, recall_at_k


def test_parse_model_json_with_fence():
    raw = '```json\n{"answer": "ok", "citations": ["c1"]}\n```'
    data = parse_model_json(raw)
    assert data["answer"] == "ok"
    assert data["citations"] == ["c1"]


def test_parse_model_json_garbage():
    data = parse_model_json("没有 JSON 的回答")
    assert data["answer"] == "没有 JSON 的回答"
    assert data["citations"] == []


def test_refuse_when_all_scores_below_threshold():
    ranked = [("c1", 0.016, {"bm25": 0.0, "vector": 0.08})]
    out = assemble_answer("问题", ranked, refuse_score=0.15)
    assert out["refused"] is True
    assert out["citations"] == []
    assert out["answer"] == "知识库中没有相关信息"


def test_citations_only_from_above_threshold():
    ranked = [
        ("c1", 0.9, {"bm25": 0.62, "vector": 0.56}),
        ("c2", 0.016, {"bm25": 0.0, "vector": 0.08}),
    ]
    out = assemble_answer("问题", ranked, refuse_score=0.15, answer_text="答案正文")
    assert out["refused"] is False
    assert out["citations"] == ["c1"]
    assert out["answer"] == "答案正文"


def test_refuse_uses_raw_support_not_rrf_score():
    # RRF 融合分只有 1/(60+rank) 量级，阈值必须比原始分判断，否则永远拒答。
    ranked = [("c1", 0.016, {"bm25": 0.62, "vector": 0.56})]
    out = assemble_answer("问题", ranked, refuse_score=0.15)
    assert out["refused"] is False
    assert out["citations"] == ["c1"]


def test_prompt_contains_chunk_ids_and_question():
    prompt = build_prompt("问题", [("c1", "片段内容")])
    assert "[c1]" in prompt
    assert "片段内容" in prompt
    assert "问题" in prompt


def test_extractive_answer_picks_relevant_sentence():
    chunks = [
        ("c1", "安重工是一家主营电子业务的公司。安重工 2024 年营业收入 20.5 亿元，同比增长 12%。"),
        ("c2", "瑞材料 2024 年研发投入占比 5.2%。"),
    ]
    answer = extractive_answer("安重工 2024 年营业收入是多少", chunks)
    assert "20.5" in answer


def test_extract_entity():
    assert extract_entity("星海量子 2024 年营业收入是多少？") == "星海量子"
    assert extract_entity("今天天气怎么样") == ""


def test_entity_gate_refuses_unknown_company():
    ranked = [("c1", 0.9, {"bm25": 0.9, "vector": 0.8})]
    texts = {"c1": "安重工 2024 年营业收入 20.5 亿元。"}
    out = assemble_answer(
        "星海量子 2024 年营业收入是多少",
        ranked,
        refuse_score=0.15,
        require_entity=True,
        chunk_texts=texts,
    )
    assert out["refused"] is True
    assert out["citations"] == []


def test_entity_gate_passes_known_company():
    ranked = [("c1", 0.9, {"bm25": 0.9, "vector": 0.8})]
    texts = {"c1": "安重工 2024 年营业收入 20.5 亿元。"}
    out = assemble_answer(
        "安重工 2024 年营业收入是多少",
        ranked,
        refuse_score=0.15,
        require_entity=True,
        chunk_texts=texts,
        answer_text="20.5 亿元",
    )
    assert out["refused"] is False
    assert out["citations"] == ["c1"]


def test_recall_and_mrr():
    retrieved = ["a", "b", "c"]
    relevant = {"b"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert mrr_at_k(retrieved, relevant, 3) == 0.5


def test_recall_counts_fraction():
    assert recall_at_k(["a"], {"a", "b"}, 5) == 0.5


def test_citation_support():
    assert citation_support("规模增长", ["基金规模增长"]) > 0.0
    assert citation_support("规模增长", ["完全无关"]) == 0.0


def test_evaluate_qa_smoke():
    items = [{"question": "q1", "relevant_chunk_ids": ["c1"]}]
    report = evaluate_qa(items, lambda q: ["c1", "c2"], k=2)
    assert report["count"] == 1
    assert report["recall_at_k"] == 1.0
    assert report["mrr_at_k"] == 1.0


def test_evaluate_qa_does_not_borrow_other_items_documents():
    """别的题目标注的文档不能算命中——否则分母与检索结果同源、指标饱和。"""
    items = [
        {"question": "q1", "relevant_chunk_ids": ["docA#root#0"]},
        {"question": "q2", "relevant_chunk_ids": ["docB#root#0"]},
    ]
    retrieved = {
        "q1": ["docB#root#3", "docB#root#4"],
        "q2": ["docA#root#7"],
    }
    report = evaluate_qa(items, lambda q: retrieved[q], k=2)
    assert report["recall_at_k"] == 0.0
    assert report["mrr_at_k"] == 0.0
