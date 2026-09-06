"""句向量模型封装（自动选择 CUDA/CPU）。

此处通过 LangChain 的 HuggingFaceEmbeddings 封装加载（统一走 transformers/sentence-transformers
栈），失败时回退 sentence-transformers 直连——两者底层同一模型，向量完全一致。
选择本地模型而非 API 嵌入的原因：评测与问答零调用成本、可离线复现、CPU 即可实时编码查询。
"""

_model = None


def get_model():
    global _model
    if _model is None:
        import torch

        from .config import EMBED_MODEL

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            ***REMOVED*** LangChain 封装：与项目其他 LangChain 组件保持一致的加载方式
            from langchain_huggingface import HuggingFaceEmbeddings

            wrapper = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
            )
            _model = _LangChainAdapter(wrapper)
        except Exception:  ***REMOVED*** noqa: BLE001 — langchain 包缺失时回退直连
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model


class _LangChainAdapter:
    """把 HuggingFaceEmbeddings 适配成 encode(texts, ...) 接口。"""

    def __init__(self, wrapper):
        self._w = wrapper

    def encode(self, texts, batch_size: int = 64, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, convert_to_numpy: bool = True):
        vecs = self._w.embed_documents(list(texts))
        import numpy as np

        arr = np.asarray(vecs, dtype="float32")
        if normalize_embeddings:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr

    def encode_query(self, text: str):
        return self._w.embed_query(text)


def embed_texts(texts: list[str], batch_size: int = 64, normalize: bool = True):
    """批量编码。normalize=True 配合余弦距离的 Chroma collection 使用。"""
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype="float32")
    model = get_model()
    if hasattr(model, "encode_query") and len(texts) == 1:
        vecs = model.encode_query(texts[0])
        vecs = [vecs]
    else:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    vecs = np.asarray(vecs, dtype="float32")
    if normalize:
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs.astype("float32")


def embed_query(question: str, normalize: bool = True):
    return embed_texts([question], normalize=normalize)[0]
