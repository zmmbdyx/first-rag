"""大模型调用：带引用标注的答案生成 + 评测用 LLM 判分。"""

import json
import re
import time

from openai import OpenAI

from .config import API_KEY, BASE_URL, LLM_MODEL
from .retriever import Hit

_client = None

SYSTEM_ANSWER = """你是企业知识库问答助手，必须严格遵守以下规则：
1. 仅根据提供的参考资料回答，禁止使用参考资料以外的知识，禁止编造。
2. 在答案的关键结论后用 [1][2] 标注所引用资料的编号；一句话可引用多条资料。
3. 如果参考资料中没有足够信息，直接回答"根据知识库中的资料，未找到相关内容"，不要猜测。
4. 回答简洁、准确，优先使用资料原文中的关键表述。"""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def chat(messages: list[dict], model: str = LLM_MODEL, temperature: float = 0.2,
         max_tokens: int | None = None) -> str:
    """OpenAI 兼容接口调用，兼容不支持 enable_thinking 参数的端点（逐级降级）。"""
    client = get_client()
    base = dict(model=model, messages=messages, temperature=temperature)
    if max_tokens:
        base["max_tokens"] = max_tokens
    last_err = None
    for extra in ({"enable_thinking": False}, None):
        try:
            kw = dict(base)
            if extra:
                kw["extra_body"] = extra
            resp = client.chat.completions.create(**kw)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  ***REMOVED*** noqa: BLE001
            last_err = e
    raise RuntimeError(f"大模型调用失败: {last_err}") from last_err


***REMOVED*** ---------- 带溯源的答案生成 ----------


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = h.location or "未知来源"
        blocks.append(f"[{i}] 来源：{loc}\n{h.text}")
    return "\n\n".join(blocks)


def answer_question(question: str, hits: list[Hit], model: str = LLM_MODEL,
                    max_tokens: int | None = 1024) -> dict:
    """生成答案并解析引用编号 → 溯源信息。"""
    if not hits:
        return {"answer": "知识库为空，请先调用入库脚本导入文档。", "sources": [], "citations": [],
                "latency": 0.0, "has_citation": False}

    prompt = f"""请根据以下参考资料回答问题。

参考资料：
{build_context(hits)}

问题：{question}"""
    t0 = time.time()
    answer = chat(
        [{"role": "system", "content": SYSTEM_ANSWER}, {"role": "user", "content": prompt}],
        model=model, max_tokens=max_tokens,
    )
    latency = time.time() - t0

    citations = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer) if 1 <= int(n) <= len(hits)})
    sources = [hits[i - 1] for i in citations]
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "latency": latency,
        "has_citation": bool(citations),
    }


***REMOVED*** ---------- 评测判分（LLM as Judge） ----------

_JUDGE_PROMPT = """你是问答质量评审员。根据「标准答案」判断「模型回答」是否正确。

【问题】{question}
【标准答案】{gold}
【模型回答】{pred}

评判标准：
- correct：模型回答中与问题所问直接相关的关键信息与标准答案一致（表述不同也算对）。模型回答额外补充的细节，只要与标准答案不矛盾，不影响判分；不要因为回答比标准答案更简短或更详细而扣分。
- partial：问题明确包含多个子问题而模型只答对了其中一部分，或回答中存在与标准答案不一致的表述。
- wrong：关键信息与标准答案矛盾、无依据地编造了与标准答案冲突的内容、或答非所问。
只输出 JSON：{{"verdict": "correct|partial|wrong", "reason": "一句话理由"}}"""

_JUDGE_UNANSWERABLE = """你是问答质量评审员。这个问题在知识库中【没有】相关内容。

【问题】{question}
【模型回答】{pred}

评判标准：
- correct：模型明确表示资料/知识库中未提及、未找到相关内容（表述可不同）；
- wrong：模型编造了答案，或给出了知识库中不存在的具体信息。
只输出 JSON：{{"verdict": "correct|wrong", "reason": "一句话理由"}}"""


def judge_answer(question: str, gold_answer: str, pred_answer: str,
                 answerable: bool = True, model: str | None = None) -> dict:
    model = model or LLM_MODEL
    template = _JUDGE_PROMPT if answerable else _JUDGE_UNANSWERABLE
    prompt = template.format(question=question, gold=gold_answer, pred=pred_answer)
    text = chat([{"role": "user", "content": prompt}], model=model,
                temperature=0.0, max_tokens=256)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    verdict = data.get("verdict", "wrong")
    if verdict not in ("correct", "partial", "wrong"):
        verdict = "wrong"
    return {"verdict": verdict, "reason": data.get("reason", text[:120])}
