"""本地句向量模型封装（sentence-transformers），自动选择 CUDA/CPU。

选择本地模型而非 API 嵌入的原因：
- 检索评测与日常问答零调用成本、可离线复现；
- 中文模型 text2vec-base-chinese 对企业制度类文本效果稳定；
- 模型只加载一次（进程内单例），入库与查询共用同一向量空间。
"""

_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        from .config import EMBED_MODEL

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, normalize: bool = True):
    """批量编码。normalize=True 配合余弦距离的 Chroma collection 使用。"""
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype="float32")
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def embed_query(question: str, normalize: bool = True):
    return embed_texts([question], normalize=normalize)[0]
