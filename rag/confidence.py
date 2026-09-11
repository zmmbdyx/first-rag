"""置信度分档：把「无答案」从**模型自觉**变成**系统保证**。

背景
----
改造前，检索结果无条件拼进 prompt，拒答与否全靠提示词里一句
"如果所有资料中都没有相关信息，请说我无法回答" —— 拒答正确率因此只有
89.2%，且遇到跨领域提问时会出现"硬答"。

分档依据
--------
优先级：重排分数 > 向量余弦相似度。

* **重排分数**（CrossEncoder logit）语义最清晰，sigmoid 后即"相关概率"，
  是阈值化的理想尺度；
* 关闭重排时退化为**向量余弦相似度**。注意两者尺度不同，因此配置里
  用同一组阈值会偏松或偏紧 —— 这是刻意的取舍：宁可保守（尺度不同时
  更容易落到 caution 档给出提示），也不要静默地给出错误答案。

分档动作
--------
    high    直接作答
    medium  作答，但前端显式提示"检索置信度偏低，建议核对引用"
    low     **不调用大模型**，直接返回拒答话术与改进建议
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .config import ANSWER_THRESHOLD, CAUTION_THRESHOLD

TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"

REFUSAL_TEXT = "根据现有资料，我无法回答这个问题。"

_LOW_HINT = (
    "知识库中没有找到与该问题足够相关的资料。你可以："
    "① 换用文档里出现过的术语或型号再问一次；"
    "② 补充限定条件（部门、年份、产品型号）；"
    "③ 若确认知识库应包含此内容，请上传对应文档后重试。"
)
_MEDIUM_HINT = "检索置信度偏低，回答仅供参考，请核对下方引用来源。"


@dataclass
class Confidence:
    """一次检索的置信度评估结果。"""

    tier: str
    score: float
    # 依据来源：rerank（CrossEncoder 相关概率/Sigmoid）或 cosine（向量余弦）
    basis: str
    threshold_answer: float | None
    threshold_caution: float | None
    hint: str = ""
    # 触发拒答时为 True，调用方据此跳过生成
    refuse: bool = False

    @property
    def warn(self) -> bool:
        return self.tier == TIER_MEDIUM

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "score": round(self.score, 4),
            "basis": self.basis,
            "hint": self.hint,
            "refuse": self.refuse,
        }


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid：避免 x 很大时 exp 溢出。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _text_score(text: str, query: str) -> float:
    """无重排分数时的兜底依据：查询词在片段中的字符重合率（0~1）。

    只作为最后手段 —— 它比余弦相似度更粗，但比"完全不看"要好得多。
    用于两侧都拿不到分数（如关键词模式且未开重排）的极端情况。
    """
    q = set(re.sub(r"[^\w\u4e00-\u9fff]+", "", query or ""))
    if not q:
        return 0.0
    t = set(re.sub(r"[^\w\u4e00-\u9fff]+", "", text or ""))
    return len(q & t) / len(q)


def assess(hits: list, query: str = "") -> Confidence:
    """对检索结果分档。

    ``hits`` 为 ``rag.retriever.Hit`` 列表（按相关性降序）。
    """
    # 门限关闭（阈值为 None）时一律直答，保持旧行为
    if ANSWER_THRESHOLD is None and CAUTION_THRESHOLD is None:
        return Confidence(TIER_HIGH, 0.0, "disabled", None, None)

    if not hits:
        return Confidence(TIER_LOW, 0.0, "empty", ANSWER_THRESHOLD, CAUTION_THRESHOLD,
                          hint=_LOW_HINT, refuse=True)

    top = hits[0]
    rerank_score = getattr(top, "rerank_score", None)
    if rerank_score is not None:
        score = sigmoid(float(rerank_score))
        basis = "rerank"
    else:
        # 融合/向量检索下 score 的含义随 mode 变化，这里统一用"文本重合率"
        # 而非直接拿 RRF 分数当相关性 —— RRF 是排名倒数和，没有相关性语义。
        score = max(_text_score(getattr(h, "text", ""), query) for h in hits[:3])
        basis = "lexical"

    answer_thr = ANSWER_THRESHOLD if ANSWER_THRESHOLD is not None else -1.0
    caution_thr = CAUTION_THRESHOLD if CAUTION_THRESHOLD is not None else -1.0

    if score >= answer_thr:
        return Confidence(TIER_HIGH, score, basis, ANSWER_THRESHOLD, CAUTION_THRESHOLD)
    if score >= caution_thr:
        return Confidence(TIER_MEDIUM, score, basis, ANSWER_THRESHOLD, CAUTION_THRESHOLD,
                          hint=_MEDIUM_HINT)
    return Confidence(TIER_LOW, score, basis, ANSWER_THRESHOLD, CAUTION_THRESHOLD,
                      hint=_LOW_HINT, refuse=True)


def refusal_payload(conf: Confidence) -> dict:
    """低置信度时直接返回的拒答结果（不调用大模型）。"""
    return {
        "answer": REFUSAL_TEXT,
        "citations": [],
        "forged_citations": [],
        "refused_by": "confidence",
        "confidence": conf.to_dict(),
        "hint": conf.hint,
    }
