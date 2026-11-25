from finrag.financebench import Chunk, evaluate, page_chunks, render_report, tokens


def test_page_chunks_keep_page_numbers():
    pages = ["a" * 10, "", "b" * 3000]
    chunks = page_chunks("DOC", pages, window=1200, overlap=200)
    assert [c.page for c in chunks] == [0, 2, 2, 2]
    assert chunks[0].chunk_id == "DOC#p0#c0"
    assert all(c.doc == "DOC" for c in chunks)
    assert all(len(c.text) <= 1200 for c in chunks)
    # 最后一窗收尾，不重复产出
    assert chunks[-1].text.endswith("b")


def test_tokens_drop_punctuation_and_lowercase():
    assert tokens("Revenue $1577.00 in FY2018") == {"revenue", "$1577.00", "in", "fy2018"}


def test_evaluate_reports_page_and_doc_hits():
    chunks = [
        Chunk("D1#p0#c0", "D1", 0, "table of contents"),
        Chunk("D1#p1#c0", "D1", 1, "capital expenditures were $1577 million in fiscal 2018"),
        Chunk("D2#p0#c0", "D2", 0, "capital expenditures discussion in another company"),
    ]
    questions = [
        {
            "financebench_id": "q1",
            "doc_name": "D1",
            "question": "What was capital expenditures in fiscal 2018?",
            "question_type": "metrics-generated",
            "evidence": [
                {
                    "evidence_page_num": 1,
                    "evidence_text": "capital expenditures were $1577 million in fiscal 2018",
                }
            ],
        }
    ]
    result = evaluate(questions, chunks, top_k=3)
    assert result["questions"] == 1
    assert result["page_hit"][1] == 1.0
    assert result["coverage"][1] == 1.0
    assert result["doc_hit"][1] == 1.0
    assert result["misses"] == []

    report = render_report(result)
    assert "FinanceBench" in report
    assert "| 题型 | 题数 | page hit@3 |" in report


def test_evaluate_counts_misses():
    chunks = [Chunk("D1#p0#c0", "D1", 0, "unrelated boilerplate text")]
    questions = [
        {
            "financebench_id": "q2",
            "doc_name": "D1",
            "question": "How many shares were repurchased?",
            "question_type": "novel-generated",
            "evidence": [
                {"evidence_page_num": 9, "evidence_text": "repurchased 12 million shares"}
            ],
        }
    ]
    result = evaluate(questions, chunks, top_k=3)
    assert result["page_hit"][3] == 0.0
    assert result["misses"][0]["got_pages"] == [0]
    assert render_report(result).count("- `q2`") == 1
