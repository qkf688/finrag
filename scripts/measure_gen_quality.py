"""测量生成侧（LLM 路径）的质量：引用遵循率、引用支撑度、对金标片段的覆盖、拒答行为。

检索评测只看「正确片段有没有被捞出来」，本脚本补上「模型拿到片段后答得怎么样」。
需要本机有 OpenAI 兼容端点（见 README「接生成模型」，环境变量 FINRAG_LLM_BASE_URL /
FINRAG_LLM_MODEL / FINRAG_LLM_API_KEY）。

用法：
    python scripts/measure_gen_quality.py --limit 50
    make gen-quality

指标口径：
- 引用遵循率：模型自报的 citations 是否都来自本次检索提供给它的片段（否则算编造 chunk_id）。
- 引用支撑度：答案 token 落在它引用片段里的比例（词级近似，非语义评审）。
- 金标覆盖：答案 token 与「该题标注 chunk」token 的交集 / 各自大小，代理答案正确性。
- 拒答率：可答题被误拒 vs 不可答题正确拒答。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from finrag.answer import build_prompt, extractive_answer, llm_answer, parse_model_json
from finrag.config import Settings
from finrag.embed import tokenize
from finrag.evaluate import citation_support
from finrag.index import build_index
from finrag.llm import LLMError, chat
from finrag.retrieve import hybrid_search


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def spread(items: list, n: int) -> list:
    """等距抽样，保证覆盖全集而不是只取开头。"""
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def cover(a: str, b: str) -> tuple[float, float]:
    """返回 (a 被 b 覆盖的比例, b 被 a 覆盖的比例)。"""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0, 0.0
    inter = len(ta & tb)
    return inter / len(ta), inter / len(tb)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成侧质量测量（需要可用的 LLM 端点）")
    parser.add_argument("--questions", type=Path, default=Path("eval/qa.jsonl"))
    parser.add_argument("--unanswerable", type=Path, default=Path("eval/qa_unanswerable.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=50, help="可答题数量；不可答题取 min(limit, 10)"
    )
    parser.add_argument("--backend", choices=["hash", "local"], default="hash")
    parser.add_argument("--json", type=Path, default=Path("results/gen-quality.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("FINRAG_LLM_MAX_TOKENS", "1024")  # 推理型模型思考也占额度
    settings = Settings.from_env()
    print(f"后端 {args.backend}  模型 {settings.llm_model}  base {settings.llm_base_url}")

    chunks, ids, texts, bm25, vec, emb = build_index(settings, args.backend)
    text_by_id = {c.chunk_id: c.text for c in chunks}
    print(f"语料 {len({c.doc_id for c in chunks})} 篇 / {len(chunks)} chunk")

    def search(question: str):
        qv = emb.encode_queries([question])[0]
        return hybrid_search(
            question, texts, ids, bm25=bm25,
            vec_scores=vec.search(qv, settings.top_k), k=settings.top_k,
        )

    answerable = spread(load_jsonl(args.questions), args.limit)
    rows: list[dict] = []
    errs = 0
    t0 = time.perf_counter()
    for i, item in enumerate(answerable, 1):
        q = item["question"]
        ranked = search(q)
        offered = [cid for cid, _, _ in ranked]
        material = [(cid, text_by_id[cid]) for cid in offered if cid in text_by_id]
        try:
            data = parse_model_json(chat(build_prompt(q, material), settings))
        except LLMError as exc:
            errs += 1
            print(f"[{i}/{len(answerable)}] ERROR {exc}")
            continue
        text, refused = llm_answer(data)
        model_cids = data.get("citations") or []
        hallu = [c for c in model_cids if c not in offered]
        supported = [c for c in model_cids if c in text_by_id]
        support = 0.0 if refused else citation_support(text, [text_by_id[c] for c in supported])
        gold = set(item.get("relevant_chunk_ids") or [])
        gold_text = " ".join(text_by_id[c] for c in gold if c in text_by_id)
        cov_ans, cov_gold = (0.0, 0.0) if refused else cover(text, gold_text)
        extr = extractive_answer(q, material)
        rows.append({
            "q": q, "refused": refused, "n_cit": len(model_cids), "hallu": len(hallu),
            "support": support, "cov_ans": cov_ans, "cov_gold": cov_gold,
            "extr_support": citation_support(extr, [t for _, t in material]),
            "retrieved_gold": bool(gold & set(offered)),
            "cited_gold": bool(gold & set(model_cids)),
        })
        print(f"[{i}/{len(answerable)}] {'拒答' if refused else '作答'} cit={len(model_cids)} "
              f"hallu={len(hallu)} sup={support:.2f}")

    elapsed = time.perf_counter() - t0
    n = len(rows)
    answered = [r for r in rows if not r["refused"]]
    hit = [r for r in rows if r["retrieved_gold"]]
    miss = [r for r in rows if not r["retrieved_gold"]]
    print(f"\n=== 可答题（{len(answerable)} 条，成功 {n}，错误 {errs}）===")
    if n:
        print(f"误拒率                 {sum(r['refused'] for r in rows) / n:.3f}")
        print(f"有引用的回答占比       {sum(1 for r in rows if r['n_cit'] > 0) / n:.3f}")
        kept = sum(1 for r in rows if r["hallu"] == 0 and r["n_cit"] > 0)
        print(f"引用零编造（全 ⊆ 检索）{kept / n:.3f}")
        print(f"平均引用条数           {sum(r['n_cit'] for r in rows) / n:.2f}")
        print(f"检索命中金标比例       {len(hit) / n:.3f}")
    if answered:
        print(f"平均引用支撑度         {sum(r['support'] for r in answered) / len(answered):.3f}")
        print(f"答案覆盖金标(token)    {sum(r['cov_ans'] for r in answered) / len(answered):.3f}")
        print(f"金标被答案覆盖(token)  {sum(r['cov_gold'] for r in answered) / len(answered):.3f}")
        extr = sum(r["extr_support"] for r in answered) / len(answered)
        print(f"抽取式基线支撑度       {extr:.3f}")
    print(f"平均耗时               {elapsed / max(n, 1):.2f}s/题")
    print("\n--- 交叉表：检索命中金标 / 未命中 × 模型作答 / 拒答 ---")
    print(f"命中且作答 {sum(not r['refused'] for r in hit):>2} / 命中共 {len(hit):>2}"
          f"  → 命中时作答率 {sum(not r['refused'] for r in hit) / max(len(hit), 1):.3f}")
    print(f"未命中且拒答 {sum(r['refused'] for r in miss):>2} / 未命中共 {len(miss):>2}"
          f"  → 未命中时拒答率 {sum(r['refused'] for r in miss) / max(len(miss), 1):.3f}")
    print(f"引用了金标 chunk 的题  {len([r for r in rows if r['cited_gold']])} / {n}")

    if not args.unanswerable.exists():
        print(f"\n未找到 {args.unanswerable}，跳过不可答题")
        unanswerable = []
    else:
        unanswerable = spread(load_jsonl(args.unanswerable), min(args.limit, 10))
    if unanswerable:
        refused_ok = 0
        for i, item in enumerate(unanswerable, 1):
            q = item["question"]
            ranked = search(q)
            material = [(cid, text_by_id[cid]) for cid, _, _ in ranked if cid in text_by_id]
            try:
                _, refused = llm_answer(parse_model_json(chat(build_prompt(q, material), settings)))
            except LLMError as exc:
                print(f"[unans {i}] ERROR {exc}")
                continue
            refused_ok += int(refused)
            print(f"[unans {i}/{len(unanswerable)}] {'拒答' if refused else '作答!'}")
        print(f"\n=== 不可答题（{len(unanswerable)} 条）===")
        print(f"正确拒答率             {refused_ok / max(len(unanswerable), 1):.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n明细已写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
