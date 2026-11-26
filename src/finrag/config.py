"""全局配置。默认值指向仓库内路径，环境变量可覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = ROOT / "corpus"
    index_dir: Path = ROOT / "index"
    eval_file: Path = ROOT / "eval" / "qa.jsonl"
    chunk_chars: int = 500
    chunk_overlap: int = 80
    top_k: int = 5
    refuse_score: float = 0.15
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    llm_model: str = "trae-cn/deepseek-v4.1-flash"
    llm_base_url: str = "http://127.0.0.1:3425/v1"
    llm_api_key: str = ""
    llm_max_tokens: int = 512

    @staticmethod
    def from_env() -> Settings:
        base = Settings()
        return Settings(
            corpus_dir=Path(os.environ.get("FINRAG_CORPUS_DIR", base.corpus_dir)),
            index_dir=Path(os.environ.get("FINRAG_INDEX_DIR", base.index_dir)),
            eval_file=Path(os.environ.get("FINRAG_EVAL_FILE", base.eval_file)),
            chunk_chars=int(os.environ.get("FINRAG_CHUNK_CHARS", base.chunk_chars)),
            chunk_overlap=int(os.environ.get("FINRAG_CHUNK_OVERLAP", base.chunk_overlap)),
            top_k=int(os.environ.get("FINRAG_TOP_K", base.top_k)),
            refuse_score=float(os.environ.get("FINRAG_REFUSE_SCORE", base.refuse_score)),
            embedding_model=os.environ.get("FINRAG_EMBEDDING_MODEL", base.embedding_model),
            embedding_dim=int(os.environ.get("FINRAG_EMBEDDING_DIM", base.embedding_dim)),
            llm_model=os.environ.get("FINRAG_LLM_MODEL", base.llm_model),
            llm_base_url=os.environ.get("FINRAG_LLM_BASE_URL", base.llm_base_url),
            llm_api_key=os.environ.get(
                "FINRAG_LLM_API_KEY",
                os.environ.get("ANTHROPIC_AUTH_TOKEN", os.environ.get("OPENAI_API_KEY", "")),
            ),
            llm_max_tokens=int(os.environ.get("FINRAG_LLM_MAX_TOKENS", base.llm_max_tokens)),
        )
