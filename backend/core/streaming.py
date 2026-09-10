"""流式调用大模型：把 OpenAI 兼容接口的逐块输出转成结构化事件。

这里的逻辑**逐字继承**自 Streamlit 版本 ``app.py::stream_answer``，只把
"边生成边渲染 UI" 换成 "边生成边 yield 事件"，因此：

* 多厂商路由仍走 ``rag.llm.get_client``（支持「模型名@端点别名」）；
* ``extra_body`` 参数仍按 ``enable_thinking`` 能力**逐级降级重试**（有的端点不支持）；
* 计时仍从**发起请求之前**开始，保证首字耗时/总耗时口径与旧版一致；
* ``reasoning_content``（思考过程）与 ``content``（正文）分开推送。

产出的事件：
    ("sources", list[dict])  —— 检索命中的来源（由调用方先发）
    ("reasoning", str)       —— 思考过程增量
    ("content", str)         —— 正文增量
    ("usage", object)        —— token 用量（流末尾，需 stream_options.include_usage）
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from rag.llm import get_client


def stream_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = 2048,
    thinking: bool = False,
    thinking_budget: int | None = None,
    stop_event: threading.Event | None = None,
) -> Iterator[tuple[str, Any]]:
    """流式生成，逐个 yield ``(事件名, 负载)``。

    ``stop_event`` 被 set 时立即停止读取并结束生成（前端"停止生成"按钮走这里，
    底层 HTTP 连接随生成器被垃圾回收而关闭）。
    """
    client, bare_model = get_client(model)

    base: dict[str, Any] = {
        "model": bare_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens:
        base["max_tokens"] = max_tokens

    # 思考模式的参数兼容性矩阵：从"功能最全"到"什么都不传"逐级降级
    if thinking:
        candidates: list[dict | None] = [
            *([{"enable_thinking": True, "thinking_budget": thinking_budget}] if thinking_budget else []),
            {"enable_thinking": True},
            None,
        ]
    else:
        candidates = [{"enable_thinking": False}, None]

    stream = None
    last_err: Exception | None = None
    t0 = time.time()  # 计时点必须在 create() 之前：建连+排队都算进延迟
    for extra in candidates:
        kw = dict(base)
        if extra:
            kw["extra_body"] = extra
        try:
            stream = client.chat.completions.create(**kw)
            break
        except Exception as e:  # noqa: BLE001 — 逐个候选参数降级重试
            last_err = e
    if stream is None:
        raise last_err if last_err else RuntimeError("流式请求初始化失败")

    ttft: float | None = None
    for chunk in stream:
        if stop_event is not None and stop_event.is_set():
            break
        usage = getattr(chunk, "usage", None)
        if usage:
            yield ("usage", usage)
            continue
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if ttft is None:
                ttft = time.time() - t0
            yield ("reasoning", reasoning)
        content = getattr(delta, "content", None)
        if content:
            if ttft is None:
                ttft = time.time() - t0
            yield ("content", content)

    yield ("meta", {"ttft": ttft, "elapsed": time.time() - t0})


def build_messages(
    system_prompt: str,
    history: list[dict],
    question: str,
) -> list[dict]:
    """拼装 OpenAI 风格消息列表：system + 历史 + 本轮提问。"""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": question})
    return messages
