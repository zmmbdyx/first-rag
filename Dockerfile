# 企业知识库 RAG 问答系统 —— 生产镜像
# 构建： docker build -t rag-kb .
# 运行： docker run -d -p 8000:8000 --env-file .env rag-kb
#
# 说明：
# - 嵌入模型（text2vec-base-chinese ~400MB）与重排模型在首次运行时下载，
#   通过挂载 HF_CACHE 卷持久化，避免每次重建镜像都重新下载；
# - 镜像内不包含任何密钥：API_KEY 等一律运行时经 --env-file / -e 注入；
# - 采用非 root 用户运行，降低容器逃逸风险。

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    # 容器内默认监听全网卡，端口由 compose/run 映射
    API_HOST=0.0.0.0 \
    API_PORT=8000

# 系统依赖：
#   libgomp1  —— torch / onnxruntime 的 OpenMP 运行时
#   libgl1    —— OpenCV 运行时（rapidocr 链路间接依赖，缺失会 ImportError）
#   curl      —— 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用 Docker 层缓存：只要 requirements 没变就不会重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码（代码改动不会触发依赖重装）
COPY . .

# 非 root 用户运行
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/chroma_db /app/logs /models /app/backend/data /app/backend/uploads \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# 健康检查走 v3.0 后端的 /api/health（旧 api_server 的 /health 仍在，但入口已切换）
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# 默认起 v3.0 后端（backend/main.py）。
# 旧的 api_server:app 与 Streamlit 界面仍可用，分别用：
#   uvicorn api_server:app --host 0.0.0.0 --port 8000
#   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
