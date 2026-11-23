"""评测集：每行一条 {"question": ..., "relevant_chunk_ids": [...]}。

chunk_id 形如 "<doc_id>#<section>#<序号>"，与 split.split_markdown 生成的一致。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QA_FILE = ROOT / "eval" / "qa.jsonl"


def load_qa(path: Path | None = None) -> list[dict]:
    p = path or QA_FILE
    if not p.exists():
        return []
    items: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items
