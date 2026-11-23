"""从真实财经语料自动生成评测集。

策略：从每篇文档的正文里挑「含数字的句子」作为答案句，用它前半段构造问题，
chunk_id 由 split 规则推导。自动生成的问题质量不如人工标注，但可复现、可扩展；
正式评测仍需人工抽检。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from finrag.config import Settings
from finrag.split import load_corpus, parse_front_matter

NUMBER_RE = re.compile(r"\d")


def make_question(sentence: str, title: str) -> str:
    """把句子改成问句：取数字前的短语 + 「是多少」。"""
    clean = re.sub(r"^[（(].*?[)）]", "", sentence).strip()
    m = NUMBER_RE.search(clean)
    if not m:
        return ""
    head = clean[: m.start()].strip("，。、；： ")
    head = re.sub(r"^(今日|今日，|近日|报道称|记者从|数据显示|最新|此外|同时|其中)[，, ]*", "", head)
    if len(head) < 4 or len(head) > 30:
        head = title[:20]
    return f"{head}相关数字是多少？"


def main() -> int:
    settings = Settings.from_env()
    chunks = load_corpus(settings.corpus_dir, max_chars=600, overlap=0)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    items: list[dict] = []
    for doc_id, cs in by_doc.items():
        if not doc_id.startswith("em-"):
            continue
        path = settings.corpus_dir / f"{doc_id}.md"
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta.get("title", doc_id)
        for c in cs:
            sentences = [s.strip() for s in re.split(r"[。\n]", c.text) if len(s.strip()) >= 15]
            for sent in sentences:
                q = make_question(sent, title)
                if q:
                    items.append({"question": q, "relevant_chunk_ids": [c.chunk_id]})
                if sum(1 for i in items if i["relevant_chunk_ids"] == [c.chunk_id]) >= 2:
                    break

    # 去重 + 限量（每个 chunk 最多 2 条，整体不超过 120 条）
    seen: set[str] = set()
    unique: list[dict] = []
    for it in items:
        if it["question"] in seen:
            continue
        seen.add(it["question"])
        unique.append(it)
        if len(unique) >= 120:
            break

    out = settings.eval_file
    out.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in unique) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"已生成 {len(unique)} 条评测问题 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())