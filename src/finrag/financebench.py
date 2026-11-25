"""FinanceBench 页级检索评测（公开 150 题子集）。

数据来源：https://github.com/patronus-ai/financebench
  - data/financebench_open_source.jsonl   150 题，含 question / answer / evidence
  - data/financebench_document_information.jsonl  361 份文档元信息
  - pdfs/<doc_name>.pdf                   原文 PDF（按需下载）

口径说明（写进报告，避免误读）：
  - 只测检索，不测生成：指标是页级命中与证据覆盖，不是答案准确率。金答案多为
    ``$1577.00`` 这类短数值，抽取式回答无法端到端判定。
  - 页级命中采用 oracle 文档设定：题目自带 ``doc_name``，在已知文档内排序。
    这是「页定位」的上界，不代表跨文档检索能力；另报全库 ``doc hit`` 作参考。
  - 页码为 evidence 里的零基 ``evidence_page_num``，与 pymupdf 页序一致
    （2026-11-25 用 63 条证据的词元重合度核验，最佳页即标注页）。
  - 证据覆盖用词元重合度 ≥ 0.8，而不是精确子串：PDF 抽取有连字符断词与连字。
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .retrieve import BM25

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{RAW_BASE}/data/financebench_open_source.jsonl"
PDF_URL = RAW_BASE + "/pdfs/{doc}.pdf"

WINDOW = 1200
OVERLAP = 200
COVERAGE = 0.8


class BenchError(RuntimeError):
    """数据缺失或解析失败。"""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    page: int
    text: str


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise BenchError(f"找不到题目文件：{path}（可用 --questions 指定）")
    items = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not items:
        raise BenchError(f"题目文件为空：{path}")
    return items


def download_pdfs(docs: list[str], dest: Path, *, workers: int = 8) -> list[str]:
    """下载缺失的 PDF，返回失败清单。已存在且 >10KB 的文件跳过。"""
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(doc: str) -> str | None:
        out = dest / f"{doc}.pdf"
        if out.exists() and out.stat().st_size > 10_000:
            return None
        try:
            with urllib.request.urlopen(PDF_URL.format(doc=doc), timeout=120) as resp:
                out.write_bytes(resp.read())
            return None
        except OSError as exc:
            return f"{doc}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [err for err in pool.map(fetch, docs) if err]


def page_texts(pdf: Path) -> list[str]:
    try:
        import pymupdf  # noqa: PLC0415  (可选依赖，按需导入)
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError as exc:
            raise BenchError("需要 pymupdf：pip install -e '.[bench]'") from exc
    with pymupdf.open(pdf) as doc:
        pages = []
        for page in doc:
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            pages.append(text)
    return pages


def page_chunks(
    doc: str, pages: list[str], *, window: int = WINDOW, overlap: int = OVERLAP
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pno, text in enumerate(pages):
        if not text:
            continue
        step = max(1, window - overlap)
        for start in range(0, max(1, len(text)), step):
            piece = text[start : start + window]
            if not piece:
                break
            chunks.append(Chunk(f"{doc}#p{pno}#c{start}", doc, pno, piece))
            if start + window >= len(text):
                break
    return chunks


def build_chunks(pdf_dir: Path, docs: list[str]) -> tuple[list[Chunk], int]:
    chunks: list[Chunk] = []
    pages_total = 0
    for doc in docs:
        pdf = pdf_dir / f"{doc}.pdf"
        if not pdf.exists():
            raise BenchError(f"缺少 PDF：{pdf}（加 --download 自动下载）")
        pages = page_texts(pdf)
        pages_total += len(pages)
        chunks.extend(page_chunks(doc, pages))
    return chunks, pages_total


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$.%]+", text.lower()))


def _coverage(gold: set[str], got: str) -> float:
    if not gold:
        return 0.0
    return len(gold & tokens(got)) / len(gold)


def evaluate(questions: list[dict], chunks: list[Chunk], *, top_k: int = 5) -> dict:
    bm25 = BM25().fit([c.text for c in chunks])
    by_doc: dict[str, list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        by_doc[chunk.doc].append(idx)

    ks = sorted({1, 3, top_k})
    page_hits = dict.fromkeys(ks, 0)
    doc_hits = dict.fromkeys(ks, 0)
    cov_hits = dict.fromkeys(ks, 0)
    by_type: dict[str, dict] = {}
    details = []
    used = [q for q in questions if q["doc_name"] in by_doc]

    for q in used:
        doc = q["doc_name"]
        scores = bm25.score(q["question"])
        in_doc = sorted(by_doc[doc], key=lambda i: scores[i], reverse=True)
        global_order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[: max(ks)]
        gold_pages = {e["evidence_page_num"] for e in q["evidence"]}
        gold_tokens = tokens(" ".join(e["evidence_text"] for e in q["evidence"]))
        hit_at: dict[int, bool] = {}
        for k in ks:
            top = in_doc[:k]
            hit = bool({chunks[i].page for i in top} & gold_pages)
            page_hits[k] += hit
            hit_at[k] = hit
            if _coverage(gold_tokens, " ".join(chunks[i].text for i in top)) >= COVERAGE:
                cov_hits[k] += 1
            if any(chunks[i].doc == doc for i in global_order[:k]):
                doc_hits[k] += 1
        qtype = q.get("question_type", "unknown")
        stat = by_type.setdefault(qtype, {"n": 0, "page@3": 0})
        stat["n"] += 1
        stat["page@3"] += hit_at.get(3, False)
        if not hit_at.get(3, False):
            details.append(
                {
                    "id": q["financebench_id"],
                    "doc": doc,
                    "type": qtype,
                    "gold_pages": sorted(gold_pages),
                    "got_pages": [chunks[i].page for i in in_doc[:3]],
                }
            )

    n = len(used)
    return {
        "questions": n,
        "skipped": len(questions) - n,
        "chunks": len(chunks),
        "docs": len(by_doc),
        "page_hit": {k: page_hits[k] / n for k in ks},
        "doc_hit": {k: doc_hits[k] / n for k in ks},
        "coverage": {k: cov_hits[k] / n for k in ks},
        "by_type": {
            t: {"n": s["n"], "page_hit@3": s["page@3"] / s["n"]} for t, s in by_type.items()
        },
        "misses": details,
    }


def render_report(result: dict, *, source: str = QUESTIONS_URL) -> str:
    ks = sorted(result["page_hit"])
    lines = [
        "# FinanceBench 页级检索（公开子集）",
        "",
        f"数据来源：{source}",
        "",
        f"- 题目：{result['questions']} 条（跳过 {result['skipped']} 条无对应 PDF）",
        f"- 文档：{result['docs']} 份；页文本按 {WINDOW} 字符窗 / {OVERLAP} 重叠切分，"
        f"共 {result['chunks']} 个 chunk",
        "- 检索：BM25（与本仓库 `finrag.retrieve.BM25` 同一实现），未启用向量与融合",
        "",
        "## 指标",
        "",
        "| 指标 | " + " | ".join(f"@{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
        "| 页级命中（oracle 文档内排序） | "
        + " | ".join(f"{result['page_hit'][k]:.3f}" for k in ks)
        + " |",
        "| 证据覆盖（词元重合 ≥ 0.8） | "
        + " | ".join(f"{result['coverage'][k]:.3f}" for k in ks)
        + " |",
        "| 全库文档命中（参考） | "
        + " | ".join(f"{result['doc_hit'][k]:.3f}" for k in ks)
        + " |",
        "",
        "## 按题型（页级命中 @3）",
        "",
        "| 题型 | 题数 | page hit@3 |",
        "|---|---|---|",
    ]
    for qtype, stat in sorted(result["by_type"].items()):
        lines.append(f"| {qtype} | {stat['n']} | {stat['page_hit@3']:.3f} |")
    lines += [
        "",
        "## 口径与局限",
        "",
        "- 只测检索，不测生成：FinanceBench 的金答案多为短数值，端到端准确率需要生成模型与判定器。",
        "- 页级命中是 oracle 文档设定（题目自带文档名），是页定位上界，不等于跨文档检索能力。",
        "- 公开子集为 150/10231 题；全量数据在 Patronus 平台，未纳入。",
        "- 固定字符窗切分未做版面还原与表格解析，10-K 的目录页与跨页表格是主要噪声来源。",
        "- 官方仓库无检索基线，本表数字无法与论文的 Oracle 85% / 长上下文 79% 直接比较"
        "（那是答案准确率）。",
        "",
        f"页级 @3 未命中的 {len(result['misses'])} 条：",
        "",
    ]
    for miss in result["misses"][:20]:
        lines.append(
            f"- `{miss['id']}` {miss['doc']} gold={miss['gold_pages']} "
            f"got={miss['got_pages']} ({miss['type']})"
        )
    return "\n".join(lines) + "\n"
