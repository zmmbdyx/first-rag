"""多轮对话查询改写：规则识别指代/省略 + LLM 规范化，两步式。

目标：把依赖上下文的口语输入（"那转正后呢？"）改写为可独立检索的完整问题
（"转正后的试用期考核标准是什么？"），改写结果同时用于检索与生成。
"""

import re

from .config import LLM_MODEL
from .llm import chat

# 常见指代/省略信号词
_ANAPHORA_RE = re.compile(
    r"(那|那么|这个|该|此|上述|以上|它|他们|它们|前面(说)?的|刚才|还有|另外|呢|呢？|？怎么|怎么样)$"
    r"|^(那|那么|这个|该|此|它|他们)"
)
_PRONOUN_RE = re.compile(r"(它|他们|它们|其|该|此)")


def needs_rewrite(message: str, history: list[dict]) -> bool:
    """规则判断：无历史时不改写；有历史且出现指代信号词/过短省略句时改写。"""
    if not history:
        return False
    msg = message.strip()
    if len(msg) <= 4: # "那社保呢？"类省略句
        return True
    return bool(_ANAPHORA_RE.search(msg))


def rule_rewrite(message: str, history: list[dict]) -> str | None:
    """纯规则改写：把最近一轮用户问题中的主题词拼接进来（低成本兜底）。"""
    if not history:
        return None
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not last_user:
        return None
    topic = re.sub(r"[？?。！]", "", last_user)
    # 去掉上一轮问题的疑问词部分，保留主题（粗粒度：整句前置）
    return f"关于「{topic}」：{message.strip()}"


_REWRITE_PROMPT = """你是查询改写器。把用户当前输入改写成一个**独立、完整、可检索**的问题：
- 结合对话历史完成指代消解（它/该/上述/那…）与省略补全（"那X呢？"→补出X具体指什么）；
- 只改写，不回答；不要编造历史中不存在的信息；
- 若当前输入本就完整独立，原样返回；
- 只输出改写后的问题本身，不要任何解释。

对话历史：
{history}

当前输入：{message}

改写后的问题："""


def llm_rewrite(message: str, history: list[dict], model: str | None = None) -> str:
    turns = "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in history[-6:])
    prompt = _REWRITE_PROMPT.format(history=turns or "（无）", message=message)
    out = chat([{"role": "user", "content": prompt}], model=model or LLM_MODEL,
               temperature=0.0, max_tokens=200)
    return out.strip().strip('"“”') or message


def rewrite_query(message: str, history: list[dict] | None = None,
                  use_llm: bool = True) -> dict:
    """统一改写入口。返回 {query, method}：method ∈ none/rule/llm。

    策略：无历史 → 原样；有历史且规则命中 → 先 LLM（失败降级规则改写）。
    """
    history = history or []
    if not history or not needs_rewrite(message, history):
        return {"query": message, "method": "none"}
    if use_llm:
        try:
            return {"query": llm_rewrite(message, history), "method": "llm"}
        except Exception: # noqa: BLE001 — LLM 不可用时降级到规则
            pass
    rule = rule_rewrite(message, history)
    return {"query": rule or message, "method": "rule" if rule else "none"}
