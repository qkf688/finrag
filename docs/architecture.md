# 架构说明

## 分层

| 层 | 模块 | 职责 | 依赖 |
|---|---|---|---|
| 语料 | `corpus/*.md` | 原始文档 + front-matter 元数据（id/title/source/license/date） | 无 |
| 切分 | `split.py` | 标题分段 → 段落打包 → chunk（带 overlap） | 无 |
| 向量化 | `embed.py` | `HashEmbedding`（离线基线）/ `LocalEmbedding`（本地模型） | 可选 sentence-transformers |
| 检索 | `retrieve.py` | BM25 + VectorIndex + RRF 混合召回 | embed |
| 生成 | `answer.py` | 提示词、JSON 解析、引用拼装、拒答阈值 | 无 |
| 评测 | `evaluate.py` | Recall@K / MRR@K / 引用支撑度 / 延迟 | 无 |
| 编排 | `cli.py` | check-corpus / eval / demo | 以上全部 |

## 关键设计取舍

1. **切分点**：先按标题层级，再按空行。标题是语义边界，优先于长度；段落内不硬切，避免把一句话劈开。
2. **中文分词**：不引 jieba，用「单字 + 相邻字二元组」，零依赖且对金融术语够用；要换 jieba 只需改 `embed.tokenize`。
3. **混合召回用 RRF 而不是加权求和**：两路分数量纲不同（BM25 是对数似然、余弦是 [-1,1]），RRF 只看名次，稳。**但 RRF 要求两路候选宽度对齐**：`hybrid_search` 里 BM25 一路是全库排名、向量一路只有 `top_n` 条，宽度不对称会让弱检索器的短名单压过强检索器的头部（实测 Recall@5 0.775 → 0.592，宽度对齐后回到 0.733，仍不及纯 BM25，见 README「为什么混合召回低于 BM25」）。因此 2026-11-28 起 Web 演示与生成侧测量改为 BM25 单路（`serve.py` / `scripts/measure_gen_quality.py` 不传 `vec_scores`），RRF 只保留在 `make report` 的四组对照里。
4. **拒答优先**：检索最高分低于阈值就返回「知识库中没有相关信息」。宁可拒答，不让模型拿不相关的片段编。
5. **引用是硬要求**：答案必须带 chunk_id，评测里的引用支撑度才有意义。
6. **评测与生成分离**：`evaluate.py` 不调 LLM，检索质量与生成质量分开测，避免用一次生成波动掩盖检索问题。

## 扩展点

- **换 embedding**：`embed.py` 实现 `EmbeddingBackend` 协议即可（`encode(texts) -> vectors`）。
- **换重排**：`retrieve.hybrid_search` 后接 cross-encoder 打分再重排。
- **换模型服务**：`llm.py` 的 `chat` 是唯一的生成侧 HTTP 出口（OpenAI 兼容 `/v1/chat/completions`），换端点/模型只改环境变量。
- **索引持久化**：`VectorIndex` 换成 FAISS / sqlite-vec，只需保持 `search()` 返回 `(doc_id, score)`。

## 已知局限（TODO）

- 语料与评测集规模还很小（2 篇 / 6 条），正式版需 40+ 篇、40–60 条标注。
- 向量检索是暴力余弦，chunk 上万后需要 ANN。
- 引用支撑度是词级近似，不是语义评审。
- 没有做增量索引与去重。
