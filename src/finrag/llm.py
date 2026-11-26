"""生成侧 LLM 客户端：只实现 OpenAI 兼容的 /v1/chat/completions。

本地 LLM 网关（magpie）与 Ollama 都提供这个接口，因此一套代码即可覆盖：

    FINRAG_LLM_BASE_URL   默认 http://127.0.0.1:3425/v1（本地网关）
    FINRAG_LLM_MODEL      默认 trae-cn/deepseek-v4.1-flash
    FINRAG_LLM_API_KEY    缺省依次回退 ANTHROPIC_AUTH_TOKEN / OPENAI_API_KEY
    FINRAG_LLM_MAX_TOKENS 默认 512（推理型模型思考也占额度，给足）

Ollama 用法：FINRAG_LLM_BASE_URL=http://localhost:11434/v1，模型名照 Ollama 的 tag。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .answer import SYSTEM_PROMPT
from .config import Settings


class LLMError(RuntimeError):
    """调用生成模型失败（网络、鉴权、空输出）。"""


def build_payload(prompt: str, settings: Settings) -> dict:
    return {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": settings.llm_max_tokens,
    }


def parse_response(data: dict) -> str:
    """取 choices[0].message.content；空内容视为失败，不冒充答案。"""
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"响应结构不符合 OpenAI 兼容格式：{exc}") from exc
    text = text.strip()
    if not text:
        raise LLMError("模型未返回内容（可能 max_tokens 被推理过程占满）")
    return text


def chat(prompt: str, settings: Settings) -> str:
    """同步调用一次，返回模型原始输出（通常是含 answer/citations 的 JSON 文本）。"""
    body = json.dumps(build_payload(prompt, settings), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise LLMError(f"HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(f"请求 {url} 失败：{exc}") from exc
    return parse_response(data)