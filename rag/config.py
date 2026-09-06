"""全局配置：路径、模型、切分与检索参数。密钥只从 .env / 环境变量读取。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

***REMOVED*** ---------- 大模型服务（OpenAI 兼容接口） ----------
***REMOVED*** 端点与密钥只在 .env / 环境变量中配置，代码与仓库中不保留任何真实端点。
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

***REMOVED*** ---------- 嵌入模型（本地运行，无需联网调用） ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "shibing624/text2vec-base-chinese")

***REMOVED*** ---------- 多模型 / 多端点 ----------
***REMOVED*** 默认端点 = 上面的 BASE_URL / API_KEY。
***REMOVED*** 附加端点在 .env 中按「别名(大写)__BASE_URL / 别名(大写)__API_KEY」命名，例如：
***REMOVED***   DEEPSEEK__BASE_URL=https://api.deepseek.com
***REMOVED***   DEEPSEEK__API_KEY=sk-xxx
***REMOVED*** 跨端点模型用「模型名@别名」表示（如 deepseek-chat@deepseek），
***REMOVED*** 未带 @ 的模型走默认端点。UI 下拉清单：
MODEL_OPTIONS = [m.strip() for m in os.getenv(
    "MODEL_OPTIONS", "deepseek-v4-pro").split(",") if m.strip()]


def resolve_model(model: str | None = None) -> tuple[str, str, str]:
    """把模型标识解析为 (真实模型名, base_url, api_key)。

    支持 "模型名@端点别名"（别名对应 .env 中的 <别名>__BASE_URL / <别名>__API_KEY）；
    不带 @ 时使用默认端点。解析失败抛 ValueError。
    """
    m = (model or LLM_MODEL).strip()
    if "@" in m:
        bare, ep = m.rsplit("@", 1)
        ep_u = ep.strip().upper()
        base = os.getenv(f"{ep_u}__BASE_URL", "").strip()
        key = os.getenv(f"{ep_u}__API_KEY", "").strip()
        if not base or not key:
            raise ValueError(
                f"端点别名 “{ep}” 未配置：请在 .env 中添加 {ep_u}__BASE_URL 和 {ep_u}__API_KEY")
        return bare, base, key
    if not BASE_URL or not API_KEY:
        raise ValueError("未配置大模型服务：请在 .env 中设置 BASE_URL 和 API_KEY")
    return m, BASE_URL, API_KEY

***REMOVED*** ---------- 路径 ----------
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")

***REMOVED*** ---------- 切分参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))    ***REMOVED*** 单块最大字符数
MIN_CHUNK_CHARS = 60                                 ***REMOVED*** 过短块并入相邻块
OVERLAP_SENTENCES = 1                                ***REMOVED*** 相邻块重叠句数
NAIVE_CHUNK_SIZE = 450                               ***REMOVED*** 基线：固定窗口大小
NAIVE_STRIDE = 350                                   ***REMOVED*** 基线：窗口步长（含 100 字符重叠）

***REMOVED*** ---------- 检索参数 ----------
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  ***REMOVED*** vector | keyword | hybrid
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))     ***REMOVED*** 召回阶段各路候选数
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))        ***REMOVED*** 送入大模型的最终条数
RRF_K = int(os.getenv("RRF_K", "60"))                   ***REMOVED*** RRF 融合常数
RRF_P = float(os.getenv("RRF_P", "1.0"))                ***REMOVED*** 加权 RRF 指数: score = Σ 1/(k+rank)^p
BM25_VARIANT = os.getenv("BM25_VARIANT", "precise_dict")  ***REMOVED*** precise | precise_dict | search | search_dict
CUSTOM_DICT = DATA_DIR / "user_dict.txt"                ***REMOVED*** jieba 自定义词典（型号/缩写/术语）
EXPAND_QUERY = os.getenv("EXPAND_QUERY", "0") == "1"    ***REMOVED*** 是否用 LLM 做查询扩展

***REMOVED*** ---------- 重排序（Reranker） ----------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "auto")    ***REMOVED*** auto | on | off
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  ***REMOVED*** 送入重排的候选数
