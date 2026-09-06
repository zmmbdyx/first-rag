"""重排序（Reranker）：用本地 CrossEncoder 对混合检索候选做精排。

说明：langchain-community 0.4.x 已移除 CrossEncoderReranker（社区包进入 sunset），
故此处直接使用 sentence_transformers.CrossEncoder——与 LangChain 压缩器的内部实现一致。
模型 BAAI/bge-reranker-base 首次使用自动下载（约 1.1GB，可设 HF_ENDPOINT 走镜像）。

开关（RERANK_ENABLED）：
- auto（默认）：模型可用则启用，加载失败自动降级关闭并告警一次
- on / off：强制开/关
"""

import logging

from .config import RERANK_CANDIDATES, RERANK_ENABLED, RERANK_MODEL

_model = None
_load_failed = False


def _get_model():
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(RERANK_MODEL)
        except Exception as e:  ***REMOVED*** noqa: BLE001
            _load_failed = True
            logging.warning("Reranker 模型加载失败（已降级为不重排）: %s", e)
    return _model


def rerank_available() -> bool:
    if RERANK_ENABLED == "off":
        return False
    if RERANK_ENABLED == "on":
        return True
    return _get_model() is not None


def rerank(question: str, hits: list, top_n: int | None = None) -> list:
    """按 CrossEncoder 相关性重排 hits（原对象复用，附加 rerank_score）。失败时原序返回。"""
    model = _get_model()
    if model is None or not hits:
        return hits
    top_n = top_n or RERANK_CANDIDATES
    cands = hits[:top_n]
    try:
        scores = model.predict([(question, h.text) for h in cands])
    except Exception:  ***REMOVED*** noqa: BLE001
        return hits
    for h, s in zip(cands, scores):
        h.rerank_score = float(s)
    return sorted(cands, key=lambda h: h.rerank_score, reverse=True) + hits[top_n:]
