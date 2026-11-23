"""生成合成金融语料与评测集。

为什么用合成语料：
- 真实研报/年报有版权与登录限制，不能公开分发；
- 合成语料可以精确控制「哪个 chunk 是答案」，评测集标注无歧义；
- 本项目要证明的是检索/评测工程能力，语料是可控基准，README 里会明确标注 synthetic。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
EVAL = ROOT / "eval" / "qa.jsonl"

SECTORS = ["电力设备", "电子", "医药生物", "机械设备", "基础化工"]
PREFIX = ["华", "远", "新", "联", "恒", "瑞", "中", "东", "金", "安"]
SUFFIX = ["科技", "股份", "智造", "材料", "电子", "医药", "能源", "重工"]


def make_name(rng: random.Random, used: set[str]) -> str:
    while True:
        name = rng.choice(PREFIX) + rng.choice(SUFFIX) + rng.choice(["", "集团", "控股"])
        if name not in used:
            used.add(name)
            return name


def build_doc(idx: int, rng: random.Random, used: set[str]) -> tuple[str, dict]:
    name = make_name(rng, used)
    sector = SECTORS[idx % len(SECTORS)]
    doc_id = f"syn-{idx:03d}"
    revenue = round(rng.uniform(8, 80), 1)
    growth = round(rng.uniform(-5, 30), 1)
    profit = round(revenue * rng.uniform(0.06, 0.18), 2)
    rd_ratio = round(rng.uniform(2.5, 12.0), 1)
    rd = round(revenue * rd_ratio / 100, 2)
    staff = rng.randint(120, 1800)
    gross = round(rng.uniform(18, 55), 1)
    risk = rng.choice(["下游需求波动", "原材料价格上行", "应收账款回收周期拉长", "行业竞争加剧", "汇率波动"])
    risk2 = rng.choice(["客户集中度较高", "新品放量不及预期", "产能爬坡慢于计划", "政策变化", "技术迭代风险"])
    product = rng.choice(["高压变频器", "储能变流器", "车载摄像头", "医用敷料", "工业机器人", "光模块", "锂电隔膜", "精密轴承"])
    region = rng.choice(["华东", "华南", "华北", "西南", "海外"])
    year = 2024

    meta = f"""---
id: {doc_id}
title: 示例：{name} {year} 年度报告要点
source: 合成数据（fictional company, 仅用于检索评测）
license: synthetic
date: {year}-12-31
---
"""
    # 每个 section 都带上公司名，否则「经营情况」段没有唯一标识词，
    # 检索会被其他公司同名指标段淹没——这是合成语料必须有的可判别性。
    body = f"""# 公司概况

{name}是一家主营{sector}业务的公司，核心产品为{product}，主要市场在{region}。{name} {year} 年营业收入 {revenue} 亿元，在职员工 {staff} 人。

# 经营情况

{name} {year} 年实现营业收入 {revenue} 亿元，同比增长 {growth}%；{name} 归母净利润 {profit} 亿元，毛利率 {gross}%。

# 研发投入

{name} {year} 年研发投入 {rd} 亿元，占{name}营业收入比重 {rd_ratio}%，研发人员 {staff // 4} 人。

# 风险提示

{name}面临的主要风险包括：{risk}、{risk2}。

# 行业与竞争

{name}所属{sector}行业，{year} 年行业整体增速约 {round(rng.uniform(1, 18), 1)}%，{name}市场份额约 {round(rng.uniform(1, 15), 1)}%。
"""
    facts = {
        "name": name,
        "revenue": revenue,
        "growth": growth,
        "profit": profit,
        "rd_ratio": rd_ratio,
        "rd": rd,
        "gross": gross,
        "risk": risk,
        "doc_id": doc_id,
    }
    return meta + body, facts


def main() -> int:
    rng = random.Random(20261123)
    CORPUS.mkdir(parents=True, exist_ok=True)
    EVAL.parent.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    docs: list[dict] = []
    for i in range(1, 41):
        text, facts = build_doc(i, rng, used)
        (CORPUS / f"{facts['doc_id']}.md").write_text(text, encoding="utf-8", newline="\n")
        docs.append(facts)

    items: list[dict] = []
    for f in docs:
        base = f"{f['doc_id']}"
        items.append(
            {
                "question": f"{f['name']} 2024 年营业收入是多少？",
                "relevant_chunk_ids": [f"{base}#经营情况#0"],
            }
        )
        items.append(
            {
                "question": f"{f['name']} 2024 年研发投入占营业收入比重是多少？",
                "relevant_chunk_ids": [f"{base}#研发投入#0"],
            }
        )
    EVAL.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"已生成 40 篇合成语料 / {len(items)} 条评测问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
