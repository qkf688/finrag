"""FinRAG：金融知识库 RAG 检索层与评测。

模块分层：
    config      全局配置（环境变量可覆盖）
    split       文档切分
    embed       向量化后端（本地模型 / 离线哈希基线）
    retrieve    BM25、向量索引、混合召回
    answer      提示词拼装、带引用回答、拒答阈值
    evaluate    Recall@K / MRR / 忠实度 / 延迟
    corpus_check 语料元数据体检
"""

__version__ = "0.1.0"
