"""命令行入口。

    python -m finrag.cli check-corpus          # 语料元数据体检
    python -m finrag.cli eval                  # 跑评测（只用检索，不调 LLM）
    python -m finrag.cli eval --backend local  # 用本地 embedding 模型
    python -m finrag.cli demo "问题"           # 单问检索 + 生成（默认离线模板回答）
    python -m finrag.cli demo "问题" --llm     # 调生成模型（默认本地 /v1 网关，可换 Ollama）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import assemble_answer, build_prompt, extractive_answer, llm_answer, parse_model_json
from .config import Settings
from .evaluate import evaluate_qa
from .index import IndexBuildError, build_index
from .llm import LLMError, chat
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
        try:
            # 模型按系统提示返回 {"answer","citations"}；无引用即视为拒答。
            # 引用 id 仍由检索排序决定，避免模型编造不存在的 chunk_id。
            answer_text, llm_refused = llm_answer(parse_model_json(chat(prompt, settings)))
        except LLMError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        if llm_refused:
            print(
                json.dumps(
                    {
                        "question": args.question,
                        "answer": answer_text,
                        "citations": [],
                        "refused": True,
                        "refuse_reason": "llm",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
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


def cmd_financebench(args: argparse.Namespace) -> int:
    from .financebench import (
        BenchError,
        build_chunks,
        download_pdfs,
        evaluate,
        load_questions,
        render_report,
    )

    try:
        questions = load_questions(args.questions)
        if args.limit:
            questions = questions[: args.limit]
        docs = sorted({q["doc_name"] for q in questions})
        if args.download:
            failed = download_pdfs(docs, args.pdf_dir)
            for err in failed:
                print(f"[ERROR] 下载失败 {err}", file=sys.stderr)
            if failed:
                return 1
        chunks, pages = build_chunks(args.pdf_dir, docs)
    except BenchError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"文档 {len(docs)} 份，{pages} 页，{len(chunks)} 个 chunk；开始评测 {len(questions)} 题")
    result = evaluate(questions, chunks, top_k=args.top_k)
    report = render_report(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print(f"已写入 {args.out}")
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
    p.add_argument("--llm", action="store_true", help="调用生成模型（OpenAI 兼容端点）生成答案")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("financebench")
    p.add_argument("--questions", type=Path, default=Path("data/financebench_open_source.jsonl"))
    p.add_argument("--pdf-dir", type=Path, default=Path("data/financebench_pdfs"))
    p.add_argument("--download", action="store_true", help="缺 PDF 时从 financebench 仓库下载")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    p.add_argument("--out", type=Path, default=Path("results/financebench.md"))
    p.set_defaults(func=cmd_financebench)

    args = parser.parse_args()
    try:
        return args.func(args)
    except IndexBuildError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
