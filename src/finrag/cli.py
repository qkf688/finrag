"""命令行入口。

    python -m finrag.cli check-corpus          # 语料元数据体检
    python -m finrag.cli eval                  # 跑评测（只用检索，不调 LLM）
    python -m finrag.cli eval --backend local  # 用本地 embedding 模型
    python -m finrag.cli demo "问题"           # 单问检索 + 生成（默认离线模板回答）
    python -m finrag.cli demo "问题" --llm     # 调本地 LLM（Ollama）生成答案
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .answer import SYSTEM_PROMPT, assemble_answer, build_prompt, extractive_answer
from .config import Settings
from .evaluate import evaluate_qa
from .index import IndexBuildError, build_index
from .qa import load_qa
from .retrieve import hybrid_search


def cmd_check_corpus(args: argparse.Namespace) -> int:
    from .corpus_check import check_corpus

    problems = check_corpus(Path(args.corpus_dir))
    for p in problems:
        print(f"[ERROR] {p}")
    print(f"检查完成：{len(problems)} 个问题")
    return 1 if problems else 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    chunks, ids, texts, bm25, vec, emb = build_index(settings, args.backend)
    items = load_qa(settings.eval_file)
    if not items:
        print(f"评测集为空：{settings.eval_file}")
        return 1

    def retrieve(question: str) -> list[str]:
        qv = emb.encode_queries([question])[0]
        ranked = hybrid_search(
            question,
            texts,
            ids,
            bm25=bm25,
            vec_scores=vec.search(qv, args.top_k),
            k=args.top_k,
        )
        return [cid for cid, _, _ in ranked]

    report = evaluate_qa(items, retrieve, k=args.top_k)
    detail = report.pop("details")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for d in detail:
        print(f"- {d['question']} → recall={d['recall_at_k']:.2f} mrr={d['mrr_at_k']:.2f}")
    if args.out:
        args.out.write_text(
            json.dumps({**report, "details": detail}, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        print(f"已写入 {args.out}")
    return 0


def call_ollama(prompt: str, settings: Settings) -> str:
    payload = json.dumps(
        {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{settings.llm_base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


def cmd_experiment(args: argparse.Namespace) -> int:
    from .experiment import render_markdown, run_experiments

    settings = Settings.from_env()
    chunks, ids, texts, _bm25, _vec, emb = build_index(settings, args.backend)
    items = load_qa(settings.eval_file)
    if not items:
        print(f"评测集为空：{settings.eval_file}")
        return 1
    results = run_experiments(texts, ids, items, emb, k=args.top_k)
    table = render_markdown(results, k=args.top_k)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8", newline="\n")
        print(f"已写入 {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .experiment import (
        evaluate_faithfulness,
        evaluate_refusal,
        render_markdown,
        run_experiments,
    )

    settings = Settings.from_env()
    chunks, ids, texts, bm25, vec, emb = build_index(settings, args.backend)
    items = load_qa(settings.eval_file)
    unanswerable = load_qa(settings.eval_file.parent / "qa_unanswerable.jsonl")
    if not items or not unanswerable:
        print("评测集不完整：需要 qa.jsonl 与 qa_unanswerable.jsonl")
        return 1

    chunk_texts = {c.chunk_id: c.text for c in chunks}
    doc_ids = {c.doc_id for c in chunks}
    real_docs = {d for d in doc_ids if d.startswith("em-")}

    def search(question: str):
        qv = emb.encode_queries([question])[0]
        return hybrid_search(
            question,
            texts,
            ids,
            bm25=bm25,
            vec_scores=vec.search(qv, settings.top_k),
            k=settings.top_k,
        )

    table = render_markdown(run_experiments(texts, ids, items, emb, k=args.top_k), k=args.top_k)
    refusal = evaluate_refusal(
        items, unanswerable, search, refuse_score=settings.refuse_score, chunk_texts=chunk_texts
    )
    faith = evaluate_faithfulness(
        items, search, chunk_texts, refuse_score=settings.refuse_score
    )

    report = f"""# 评测报告

生成命令：`python -m finrag.cli report`。
语料：{len(doc_ids)} 篇文档（其中 {len(real_docs)} 篇为真实新闻，其余为合成/样例干扰项）。
可答问题 {refusal['answerable']} 条，不可答 {refusal['unanswerable']} 条。

## 检索（k={args.top_k}）

{table}

## 拒答（阈值 {settings.refuse_score}）

| 指标 | 值 |
|---|---:|
| 可答问题误拒率 | {refusal['false_refusal_rate']:.3f} |
| 不可答问题拒答率 | {refusal['unanswerable_refusal_rate']:.3f} |

## 抽取式回答的引用支撑度

| 指标 | 值 |
|---|---:|
| 平均引用支撑度 | {faith['mean_citation_support']:.3f} |
| 被拒答 | {faith['refused']} / {faith['count']} |
"""
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print(f"已写入 {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    chunks, ids, texts, bm25, vec, emb = build_index(settings, args.backend)
    qv = emb.encode_queries([args.question])[0]
    ranked = hybrid_search(
        args.question,
        texts,
        ids,
        bm25=bm25,
        vec_scores=vec.search(qv, settings.top_k),
        k=settings.top_k,
    )
    by_id = {c.chunk_id: c for c in chunks}
    material = [(cid, by_id[cid].text) for cid, _, _ in ranked if cid in by_id]
    prompt = build_prompt(args.question, material)
    if args.llm:
        answer_text = call_ollama(prompt, settings)
    else:
        answer_text = extractive_answer(args.question, material)
    out = assemble_answer(
        args.question,
        ranked,
        refuse_score=settings.refuse_score,
        answer_text=answer_text,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-corpus")
    p.add_argument("--corpus-dir", default="corpus")
    p.set_defaults(func=cmd_check_corpus)

    p = sub.add_parser("eval")
    p.add_argument("--backend", choices=("hash", "local"), default="hash")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=argparse.FileType("w", encoding="utf-8"), default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("experiment")
    p.add_argument("--backend", choices=("hash", "local"), default="hash")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("results/experiments.md"))
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("report")
    p.add_argument("--backend", choices=("hash", "local"), default="hash")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("results/report.md"))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo")
    p.add_argument("question")
    p.add_argument("--backend", choices=("hash", "local"), default="hash")
    p.add_argument("--llm", action="store_true", help="调用本地 Ollama 生成答案")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    try:
        return args.func(args)
    except IndexBuildError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
