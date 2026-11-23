from finrag.split import load_corpus, parse_front_matter, split_markdown

DOC = """---
id: doc-1
title: 测试文档
source: local
---

# 一级标题

正文第一段。

正文第二段。
"""


def test_parse_front_matter():
    meta, body = parse_front_matter(DOC)
    assert meta["id"] == "doc-1"
    assert meta["title"] == "测试文档"
    assert "一级标题" in body


def test_split_keeps_sections_and_ids():
    chunks = split_markdown(DOC, doc_id="doc-1", source="s")
    assert chunks
    assert chunks[0].chunk_id.startswith("doc-1#")
    assert chunks[0].section == "一级标题"
    assert chunks[0].title == "测试文档"


def test_split_respects_max_chars_for_paragraphs():
    doc = "---\nid: d\n---\n\n# S\n\n" + "\n\n".join(f"第{i}段内容。" for i in range(10))
    chunks = split_markdown(doc, doc_id="d", source="s", max_chars=30, overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 30


def test_overlap_prefixes_previous_chunk():
    para = "这是用于测试重叠逻辑的一段比较长的中文内容，" * 2
    doc = "---\nid: d\n---\n\n# S\n\n" + "\n\n".join(para for _ in range(4))
    plain = split_markdown(doc, doc_id="d", source="s", max_chars=60, overlap=0)
    overlapped = split_markdown(doc, doc_id="d", source="s", max_chars=60, overlap=10)
    assert len(plain) == len(overlapped)
    assert len(overlapped[1].text) > len(plain[1].text)


def test_load_corpus_reads_markdown(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert chunks
    assert chunks[0].doc_id == "doc-1"
