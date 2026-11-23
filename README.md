# FinRAG

金融知识库 RAG 检索层与评测 —— 独立项目，目标是把「LLM + 检索」做出**可复现、可量化**的证据。

对应 JD 要求（`AI 全栈开发工程师`）：req-005 搭建与优化金融 RAG 知识库与检索增强生成系统（must_have）。

## 为什么做这个

- 这类岗位的 JD 写的是「搭建和优化」，落点就是**评测数字**：检索召回率、答案忠实度、延迟。没有数字就只是堆功能。
- 独立仓库叙事更清楚：不挂在别的项目下面，README 一眼能看懂「做了什么、怎么测的、结果如何」。
- 全流程离线可复现：默认用哈希基线做检索评测，模型可替换；没有 GPU、没有 API key 也能把评测表跑出来。

## 架构

```
corpus/*.md ──split.py──> chunk(doc_id, section, text)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        BM25（纯 Python）              embed.py（Hash / 本地模型）
              │                               │
              └────── retrieve.hybrid_search ──┘   RRF 融合 → top-k
                              │
                     answer.py 组装带引用的答案 / 拒答
                              │
                     evaluate.py 出评测表
```

- `src/finrag/split.py`：按标题 + 段落切分，带 overlap。
- `src/finrag/embed.py`：中文按「单字 + 二元组」分词；`HashEmbedding` 是确定性离线基线，`LocalEmbedding` 用本地 bge-small-zh。
- `src/finrag/retrieve.py`：BM25 + 向量索引 + RRF 混合召回。
- `src/finrag/answer.py`：提示词要求「只依据片段回答」，输出 JSON（答案 + 引用）；得分低于阈值直接拒答。
- `src/finrag/evaluate.py`：Recall@K / MRR@K / 引用支撑度 / P50、P95 延迟。

## 快速开始

```bash
# Windows
.\scripts\setup.ps1

# 或通用
make setup
make test          # 跑测试
make check-corpus  # 语料元数据体检
make eval          # 跑评测，产出 results/eval-hash.json
make demo          # 单问检索 + 组装答案（--llm 可调本地 Ollama）
```

## 评测表

基准：40 篇**合成**中文金融文档（fictional company，`license: synthetic`）+ 80 条自动生成问答；每个 section 都带公司名以保证可判别性。

`python -m finrag.cli experiment` 实测（HashEmbedding 512 维基线，k=5，80 条）：

| 方案 | Recall@5 | MRR@10 | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| BM25 | 1.000 | 0.721 | 1.96 | 2.46 |
| 纯向量 | 1.000 | 0.760 | 10.75 | 11.85 |
| 混合召回（RRF） | 1.000 | 0.727 | 13.14 | 15.78 |
| 混合 + 词面重排 | 1.000 | 0.662 | 13.98 | 77.56 |

结论（如实写，不粉饰）：

- 在这个合成基准上四种方案的 **Recall@5 都是 1.0**——语料只有 40 篇、每个 section 都带唯一公司名，检索难度偏低，不能拿它证明「线上效果好」。
- **MRR 上纯向量最好（0.760），混合 0.727，词面重排反而降到 0.662**。原因是词面重排与 BM25 重复计分，把 RRF 已经排好的顺序打乱了。
- 因此下一步不是继续调词面权重，而是接 **cross-encoder 语义重排**，并扩大语料与评测集；同时把「拒答」与「引用忠实度」纳入评测。
- P95 77ms 是重排路径的离群值（单条最慢样本），不是稳态；正式报告会跑多轮取分位。


## 已知局限

- 语料是 40 篇**合成**文档 + 80 条自动生成问答，不是真实研报；它能验证检索链路，但不能代替真实语料上的人工标注评测。正式版需要接入公开真实材料并做 40–60 条人工标注。
- 向量检索是暴力余弦，chunk 上万后要换 ANN。
- 忠实度是词级覆盖近似，不是语义级评审。

## 路线图

- [ ] 语料扩容到 40+ 篇 + 评测集 40–60 条
- [ ] 接入本地 embedding 模型并与哈希基线做对照
- [ ] 加 cross-encoder 重排
- [ ] 出评测表与 README 结论
- [ ] 打 tag v1.0.0

## 开发方式

本仓库按「AI 辅助开发 + 机械门禁」的方式推进：改动前先检索复用点，改完跑测试与静态分析再提交，提交信息里写清验证证据。
