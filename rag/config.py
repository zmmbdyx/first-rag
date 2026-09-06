"""全局配置：路径、模型、切分与检索参数。密钥只从 .env / 环境变量读取。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

***REMOVED*** ---------- 大模型服务（OpenAI 兼容接口） ----------
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv(
    "BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

***REMOVED*** ---------- 嵌入模型（本地运行，无需联网调用） ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "shibing624/text2vec-base-chinese")

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
