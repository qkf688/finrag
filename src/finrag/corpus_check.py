"""语料体检：检查 corpus/ 下每篇文档的 front-matter 元数据是否齐全。"""

from __future__ import annotations

import sys
from pathlib import Path

from .split import parse_front_matter

REQUIRED = ("id", "title", "source", "license", "date")


def check_corpus(corpus_dir: Path) -> list[str]:
    """返回问题清单；空列表表示全部合规。"""
    problems: list[str] = []
    if not corpus_dir.exists():
        return [f"语料目录不存在：{corpus_dir}"]
    for path in sorted(corpus_dir.glob("*.md")):
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        for key in REQUIRED:
            if not meta.get(key):
                problems.append(f"{path.name}: 缺少 front-matter 字段 {key}")
    return problems


def main() -> int:
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus")
    problems = check_corpus(corpus_dir)
    for p in problems:
        print(f"[ERROR] {p}")
    print(f"检查完成：{len(problems)} 个问题")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
