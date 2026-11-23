"""文档切分：先按标题层级切，再按段落打包成不超过 max_chars 的 chunk。"""

from __future__ import annotations

import re
from dataclasses import dataclass

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source: str


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """拆出 YAML front-matter（简单 key: value，不引第三方 YAML 依赖）。"""
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[m.end():]


def split_markdown(
    text: str,
    *,
    doc_id: str,
    source: str,
    max_chars: int = 500,
    overlap: int = 80,
    title: str = "",
) -> list[Chunk]:
    """把一篇 markdown 切成 chunk。切分点优先标题，其次空行。"""
    meta, body = parse_front_matter(text)
    doc_title = title or meta.get("title") or doc_id
    source = source or meta.get("source", "")

    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in body.splitlines():
        h = HEADING_RE.match(line)
        if h:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = h.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    for heading, lines in sections:
        blocks: list[str] = []
        buffer = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    blocks.append(buffer)
                    buffer = ""
                continue
            candidate = f"{buffer}\n{stripped}" if buffer else stripped
            if len(candidate) > max_chars and buffer:
                blocks.append(buffer)
                buffer = stripped
            else:
                buffer = candidate
        if buffer:
            blocks.append(buffer)

        for i, block in enumerate(blocks):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{heading or 'root'}#{i}",
                    doc_id=doc_id,
                    title=doc_title,
                    section=heading,
                    text=block,
                    source=source,
                )
            )

    if overlap > 0:
        chunks = _add_overlap(chunks, overlap)
    return chunks


def _add_overlap(chunks: list[Chunk], overlap: int) -> list[Chunk]:
    """相邻 chunk 之间补 overlap 个字符，减少跨块信息割裂。"""
    out: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        text = chunk.text
        if i > 0 and overlap > 0:
            prev = chunks[i - 1].text
            if len(prev) > overlap:
                text = prev[-overlap:] + "\n" + text
        out.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                section=chunk.section,
                text=text,
                source=chunk.source,
            )
        )
    return out


def load_corpus(corpus_dir, *, max_chars: int = 500, overlap: int = 80) -> list[Chunk]:
    """读取 corpus/ 下所有 .md，按文件名作为 doc_id 切分。"""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, _ = parse_front_matter(text)
        doc_id = meta.get("id") or path.stem
        chunks.extend(
            split_markdown(
                text,
                doc_id=doc_id,
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                max_chars=max_chars,
                overlap=overlap,
            )
        )
    return chunks
