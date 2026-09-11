"""全局配置：路径、模型、切分与检索参数。密钥只从 .env / 环境变量读取。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------- 大模型服务（OpenAI 兼容接口） ----------
# 端点与密钥只在 .env / 环境变量中配置，代码与仓库中不保留任何真实端点。
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

# ---------- 嵌入模型（本地运行，无需联网调用） ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "shibing624/text2vec-base-chinese")

# ---------- 多模型 / 多端点 ----------
# 默认端点 = 上面的 BASE_URL / API_KEY。
# 附加端点在 .env 中按「别名(大写)__BASE_URL / 别名(大写)__API_KEY」命名，例如：
#   DEEPSEEK__BASE_URL=https://api.deepseek.com
#   DEEPSEEK__API_KEY=sk-xxx
# 跨端点模型用「模型名@别名」表示（如 deepseek-chat@deepseek），
# 未带 @ 的模型走默认端点。UI 下拉清单：
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

# ---------- 路径 ----------
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")

# ---------- 切分参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420")) # 单块最大字符数
MIN_CHUNK_CHARS = 60 # 过短块并入相邻块
OVERLAP_SENTENCES = 1 # 相邻块重叠句数
NAIVE_CHUNK_SIZE = 450 # 基线：固定窗口大小
NAIVE_STRIDE = 350 # 基线：窗口步长（含 100 字符重叠）

# ---------- 检索参数 ----------
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid") # vector | keyword | hybrid
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10")) # 召回阶段各路候选数
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5")) # 送入大模型的最终条数
RRF_K = int(os.getenv("RRF_K", "60")) # RRF 融合常数
RRF_P = float(os.getenv("RRF_P", "1.0")) # 加权 RRF 指数: score = Σ 1/(k+rank)^p
BM25_VARIANT = os.getenv("BM25_VARIANT", "precise_dict") # precise | precise_dict | search | search_dict
CUSTOM_DICT = DATA_DIR / "user_dict.txt" # jieba 自定义词典（型号/缩写/术语）
EXPAND_QUERY = os.getenv("EXPAND_QUERY", "0") == "1" # 是否用 LLM 做查询扩展

# ---------- 重排序（Reranker） ----------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "auto") # auto | on | off
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10")) # 送入重排的候选数（越小越快）

# ---------- 入库并发 ----------
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "4")) # 解析并发度
ASYNC_INGEST = os.getenv("ASYNC_INGEST", "1") == "1" # 异步解析（asyncio+线程池）

# ---------- 文档级权限（ACL） ----------
# 每个 chunk 入库时带 perm_tags 标签；检索时按调用方的用户组过滤。
#   RAG_ACL_ENABLED=1     检索时启用权限过滤（默认开）
#   RAG_ACL_STRICT=1      严格模式：未打标签的 chunk 不对普通用户可见
#   RAG_ACL_DEFAULT_TAGS  未显式指定标签时使用的默认标签
#
# 默认（宽松）模式下"未打标签"视为公开，好处是升级到带 ACL 的版本后历史索引
# 不会突然全部检索不到；但生产部署应开启 RAG_ACL_STRICT=1 并重新入库，
# 让"没打标签"默认变成拒绝（fail-closed）而不是放行（fail-open）。
ACL_ENABLED = os.getenv("RAG_ACL_ENABLED", "1") == "1"
ACL_STRICT = os.getenv("RAG_ACL_STRICT", "0") == "1"
ACL_DEFAULT_TAGS = [t.strip() for t in os.getenv("RAG_ACL_DEFAULT_TAGS", "public").split(",") if t.strip()]
ACL_PUBLIC_TAG = "public"
# 带该标签的用户可看全部文档（管理员/审计角色）
ACL_ADMIN_TAG = os.getenv("RAG_ACL_ADMIN_TAG", "admin")

# ---------- 相关性阈值与分级拒答 ----------
# 按重排分数（CrossEncoder logit，sigmoid 后为 0~1 的相关概率）分档：
#   >= ANSWER_THRESHOLD   直答
#   >= CAUTION_THRESHOLD  直答但提示"仅供参考"
#   <  CAUTION_THRESHOLD  不生成，直接拒答并给出建议
# 关闭重排时退化为按向量余弦相似度判定（尺度不同，见 core/confidence.py 说明）。
# 置为空串/none 表示不做门限（保持旧行为）。
_thr_answer = os.getenv("RAG_ANSWER_THRESHOLD", "0.35")
_thr_caution = os.getenv("RAG_CAUTION_THRESHOLD", "0.15")
ANSWER_THRESHOLD = float(_thr_answer) if _thr_answer.lower() not in ("", "none") else None
CAUTION_THRESHOLD = float(_thr_caution) if _thr_caution.lower() not in ("", "none") else None

# ---------- 审计日志（PII 与保留期） ----------
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1") == "1"
# 审计落盘前是否对 PII 做脱敏
AUDIT_REDACT = os.getenv("AUDIT_REDACT", "1") == "1"
# 是否保存答案/上下文原文（关闭则只留长度摘要与哈希，隐私最强但排障变弱）
AUDIT_STORE_TEXT = os.getenv("AUDIT_STORE_TEXT", "1") == "1"
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))

# ---------- 重排调优 ----------
# 早停：若第 1 名领先第 2 名超过该分差，则认为顺序已稳定，跳过剩余候选计算。
# 默认 0 表示不早停（保持原行为）。
RERANK_EARLY_STOP_MARGIN = float(os.getenv("RERANK_EARLY_STOP_MARGIN", "0.0"))

# ---------- 服务化（FastAPI） ----------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()] # 逗号分隔；空=不鉴权

# ---------- Redis 问答缓存 ----------
# 详见 rag/cache.py：Redis 不可用时全部降级为 no-op，不影响主流程。
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "auto") # auto | on | off
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800")) # 秒
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "rag:qa")
