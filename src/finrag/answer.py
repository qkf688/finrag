"""回答拼装：提示词、带引用输出、拒答阈值。"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "你是金融知识库问答助手。只依据【资料片段】回答问题；"
    "片段不足以回答时，answer 写「知识库中没有相关信息」，citations 给空数组。"
    '只输出 JSON：{"answer": "…", "citations": ["chunk_id", …]}'
)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def build_prompt(question: str, chunks: list[tuple[str, str]]) -> str:
    """chunks: [(chunk_id, text)]。"""
    blocks = []
    for cid, text in chunks:
        blocks.append(f"[{cid}]\n{text}")
    material = "\n\n".join(blocks)
    return f"【资料片段】\n{material}\n\n【问题】\n{question}"


def parse_model_json(raw: str) -> dict:
    """从模型输出里抠出 JSON；容忍 ```json 包裹与前后废话。"""
    if not raw:
        return {"answer": "", "citations": []}
    m = _JSON_RE.search(raw)
    if not m:
        return {"answer": raw.strip(), "citations": []}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"answer": raw.strip(), "citations": []}
    if not isinstance(data, dict):
        return {"answer": str(data), "citations": []}
    data.setdefault("answer", "")
    data.setdefault("citations", [])
    if not isinstance(data["citations"], list):
        data["citations"] = []
    return data


def assemble_answer(
    question: str,
    ranked: list[tuple[str, float, dict[str, float]]],
    *,
    refuse_score: float = 0.05,
    answer_text: str = "",
    max_citations: int = 3,
) -> dict:
    """把检索结果拼成最终回答。

    拒答阈值比对的是**各路原始分**（已归一化到 0–1），不是 RRF 融合分——
    融合分的量级是 1/(60+rank)，与阈值不同量纲，拿它判拒答会永远拒答。
    """
    def support(raw: dict[str, float]) -> float:
        return max(raw.values()) if raw else 0.0

    kept = [(cid, score) for cid, score, raw in ranked if support(raw) >= refuse_score]
    if not kept:
        return {
            "question": question,
            "answer": "知识库中没有相关信息",
            "citations": [],
            "refused": True,
        }
    citations = [cid for cid, _ in kept[:max_citations]]
    return {
        "question": question,
        "answer": answer_text or "",
        "citations": citations,
        "refused": False,
    }
