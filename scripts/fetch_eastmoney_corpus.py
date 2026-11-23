"""抓取东方财富公开财经新闻，作为 FinRAG 的真实语料。

只抓公开页面，不登录、不绕反爬；每篇保留来源 URL 与抓取时间，仅用于检索评测。
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
LIST_URL = "https://finance.eastmoney.com/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
ARTICLE_RE = re.compile(r"https://finance\.eastmoney\.com/a/\d+\.html")
BODY_START = 'id="ContentBody"'
TAG_RE = re.compile(r"<[^>]+>")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_article(html: str) -> tuple[str, str]:
    title_m = re.search(r"<title>(.*?)</title>", html, re.S)
    title = TAG_RE.sub("", title_m.group(1)).strip() if title_m else ""
    start = html.find(BODY_START)
    if start < 0:
        return title, ""
    snippet = html[start : start + 60000]
    # 正文容器到文末分享/推荐区为止；用常见边界标记，取最靠前的有效位置。
    boundaries = [
        snippet.find("</div>", snippet.find('id="ContentBody"') + 1),
        snippet.find("东方财富网", 200),
        snippet.find("责任编辑", 200),
    ]
    ends = [b for b in boundaries if b > 200]
    text = TAG_RE.sub("\n", snippet[: min(ends) if ends else len(snippet)])
    text = re.sub(r"[ \t　]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 8]
    return title, "\n".join(lines)


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    html = fetch(LIST_URL)
    links = list(dict.fromkeys(ARTICLE_RE.findall(html)))[:40]
    if not links:
        print("列表页没有抓到文章链接（可能改版或被反爬）")
        return 1

    made = 0
    for idx, url in enumerate(links, start=1):
        try:
            title, body = extract_article(fetch(url))
        except Exception as exc:  # 单篇失败不影响整体
            print(f"跳过 {url}: {exc}")
            continue
        if len(body) < 400:
            print(f"跳过 {url}: 正文过短")
            continue
        doc_id = f"em-{idx:03d}"
        meta = f"""---
id: {doc_id}
title: {title}
source: {url}
license: 公开新闻（仅用于检索评测，版权归原作者）
date: 2026-11-23
---
"""
        (CORPUS / f"{doc_id}.md").write_text(meta + body, encoding="utf-8", newline="\n")
        made += 1
        print(f"已保存 {doc_id} {title[:40]}")
    print(f"完成：{made} 篇")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())