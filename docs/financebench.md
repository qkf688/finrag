# FinanceBench 页级检索评测

## 为什么做

自建的 120 条问答是脚本从正文生成的、标注只到文档级，面试官一问「这 120 条谁标的」就露。
[FinanceBench](https://github.com/patronus-ai/financebench)（arXiv 2311.11944）是公开基准，
题目、金答案与证据页都可以被第三方核对，因此本项目把它作为「检索能力」的外部参照。

本项目不接生成模型，所以只评测检索的一半：**给定题目，金答案所在的那一页有没有排进 top-k**。
端到端答案准确率不在本表范围内。

## 数据

- 150 题公开子集（全量 10,231 题在 Patronus 平台，未公开）。
- 84 份 10-K / 10-Q / 8-K / Earnings PDF，共 12,013 页。
- 题目自带 `doc_name`（金文档）与 `evidence`（含零基 `evidence_page_num` 和原文）。
- `data/financebench_open_source.jsonl` 已入库（908 KB）；PDF 合计 159 MB，不入库，按需下载。

## 做法

1. 解析：pymupdf 抽取每页文本，折叠空白。
2. 切分：1200 字符窗、200 重叠；`chunk_id = <doc>#p<page>#c<start>`，保留零基页码。
3. 检索：复用仓库内的 `finrag.retrieve.BM25`（纯 Python，无第三方依赖），
   对全部 43,605 个 chunk 一起打分。
4. 指标：
   - **页级命中 page hit@k**：题目自带文档内排序，top-k 的 chunk 是否落在金页上。
   - **证据覆盖 coverage@k**：top-k 文本与金证据的词元重合度是否 ≥ 0.8
     （PDF 抽取有连字符断词与连字，精确子串对截断长度敏感，故用词元集合）。
   - **全库文档命中 doc hit@k**（参考）：全库排序的 top-k 里是否出现金文档，
     用来衡量「没有 oracle 文档时」的检索难度。

## 结果（2026-11-25，150 题 / 84 文档）

| 指标 | @1 | @3 | @5 |
|---|---|---|---|
| 页级命中（oracle 文档内排序） | 0.153 | 0.233 | 0.307 |
| 证据覆盖（词元重合 ≥ 0.8） | 0.067 | 0.160 | 0.220 |
| 全库文档命中（参考） | 0.247 | 0.373 | 0.480 |

| 题型 | 题数 | page hit@3 |
|---|---|---|
| domain-relevant | 50 | 0.200 |
| metrics-generated | 50 | 0.120 |
| novel-generated | 50 | 0.380 |

完整报告（含 115 条未命中明细）由命令写入 `results/financebench.md`（`results/` 不入库）。

## 怎么读

- **不能和新闻语料的 Recall@5 0.775 直接比**：那套指标是文档级、语料 78 篇、
  问题由正文自动生成；这里是页级、84 份真实 10-K，难度差一个量级。
- **不能和论文的 Oracle 85%、长上下文 79% 比**：那是生成模型的答案准确率，
  本表是检索指标，官方仓库也没有公布检索基线。
- `metrics-generated` 最低（0.120）：数值题依赖表格，固定字符窗会把表头与数字拆开，
  需要版面还原或表格解析才能实质改善。
- 小样本探针（12 份文档 / 52 题）曾给出 0.231 / 0.308 / 0.346，
  全量跑出来是 0.153 / 0.233 / 0.307 —— 抽样会高估，最终数字以全量为准。
- oracle 文档是上界；全库 doc hit@5 只有 0.480，跨文档检索是另一道题。

## 局限

1. 只测检索，不测生成：金答案多是 `$1577.00` 这类短数值，端到端判定需要生成模型与 judge。
2. 页级命中采用 oracle 文档设定（题目自带 `doc_name`），不代表真实场景下的跨文档检索。
3. 公开子集只有 150/10,231 题。
4. 固定字符窗切分，无版面还原与表格解析；10-K 的目录页与跨页表格是主要噪声来源。
5. 官方仓库无检索基线，本表数字无法与论文的 Oracle 85% / 长上下文 79% 直接比较。

## 复现

```bash
make bench-setup     # pip install -e ".[bench]"（pymupdf）
make financebench    # 等价：python -m finrag.cli financebench --download
```

或手动指定路径与限额：

```bash
PYTHONPATH=src python -m finrag.cli financebench \
  --questions data/financebench_open_source.jsonl \
  --pdf-dir data/financebench_pdfs \
  --top-k 5 --out results/financebench.md
```

`--limit N` 只跑前 N 题（冒烟用）。PDF 缺失时加 `--download`，已存在且大于 10 KB 的文件会跳过。
本机单进程整跑约 10 分钟量级（含 PDF 解析；43,605 chunk × 150 题，纯 Python BM25）。
