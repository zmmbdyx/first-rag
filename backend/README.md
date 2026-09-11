# 后端服务（FastAPI + LangChain 编排）

企业知识库 RAG 问答系统的后端。提供 RESTful + SSE 接口，负责文档解析入库、向量检索、多轮改写、
大模型流式生成与会话持久化。

## 快速启动

```bash
# 在项目根目录执行（backend 需要导入同级的 rag/ 核心包）
uvicorn backend.main:app --reload --port 8000
```

- OpenAPI 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>

也可以用 `python -m backend.main`（读取 `HOST` / `PORT` / `RELOAD` 环境变量）。

## 目录结构

```
backend/
├── main.py              # FastAPI 入口：CORS、路由注册、lifespan 预热
├── config.py            # Pydantic Settings（新增配置；密钥复用 rag/config.py 的 .env 契约）
├── api/
│   ├── chat.py          # POST /api/chat —— SSE 流式问答
│   ├── conversations.py # 会话 CRUD
│   └── upload.py        # POST /api/upload —— 文档上传入库
├── core/
│   ├── rag_chain.py     # RAG 编排：安全校验 → 多轮改写 → 混合检索 → 流式生成
│   ├── streaming.py     # 大模型流式封装（含 extra_body 逐级降级重试）
│   ├── vectorstore.py   # 检索器进程级单例（入库后热重载）
│   └── embeddings.py    # Embedding 配置元信息
├── models/              # SQLAlchemy：database / conversation / message
├── schemas/             # Pydantic：chat / conversation / common / upload
├── services/            # 业务层：conversation_service / upload_service
├── uploads/             # 上传文件落盘目录（已 gitignore）
└── data/                # SQLite 会话库（已 gitignore）
```

> **注意**：本目录**不包含**任何 RAG 核心逻辑。文档切分策略、Embedding 模型、向量库、
 *检索策略（向量/BM25/RRF/重排）、多轮改写与引用校验全部复用项目根目录的 `rag/` 核心包*，
 与 Streamlit（`app.py`）和 CLI（`scripts/ask.py`）共用同一套实现。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | SSE 流式问答（按 `X-User-Groups` 做文档级权限过滤；低置信直接拒答） |
| `POST` | `/api/conversations` | 新建会话 |
| `GET` | `/api/conversations` | 会话列表（**仅当前用户**，带身份时按 owner 过滤） |
| `GET` | `/api/conversations/{id}/messages` | 某会话全部消息（校验归属，越权返回 404） |
| `DELETE` | `/api/conversations/{id}` | 删除会话 |
| `PUT` | `/api/conversations/{id}/title` | 重命名会话 |
| `POST` | `/api/upload` | 上传文档（可带 `?perm_tags=hr` 指定密级；需管理员密钥） |
| `GET` | `/api/documents` | 文档清单 + 密级 + `untagged_count`（未归类数） |
| `PUT` | `/api/documents/{name}/tags` | 修改文档密级（需 `ADMIN_KEYS`） |
| `DELETE` | `/api/documents/{name}` | 删除文档及其全部向量（需 `ADMIN_KEYS`） |
| `POST` | `/api/documents/backfill-tags` | 给无标签的历史文档补密级（默认 dry-run，需 `ADMIN_KEYS`） |
| `POST` | `/api/feedback` | 回答反馈 👍/👎（在线质量信号，与 `request_id` 关联） |
| `GET` | `/api/health` | 服务与知识库状态 **免鉴权** |

### 权限、置信度与可追溯（v3.1）

**ACL**：检索按 `X-User-Groups`（默认头名，可用 `ACL_GROUPS_HEADER` 配置）过滤文档。
`public` 全员可见、`admin` 组可见全部、其余按标签取交集；无标签文档在
`RAG_ACL_STRICT=1` 下仅管理员可见（fail-closed）。**问答缓存键里带用户组**，
避免 A 组检索到的答案被返回给 B 组。

> ⚠️ 这些头必须由可信网关注入并**覆盖**客户端同名头，否则客户端可伪造成任意组。

**置信度**：按重排分数 sigmoid 后的相关概率分三档 —— `high` 直答、
`medium` 直答并在前端提示"仅供参考"、`low` **不调用大模型**直接拒答。
阈值见 `RAG_ANSWER_THRESHOLD` / `RAG_CAUTION_THRESHOLD`。

**可追溯**：每次问答写入 `chat_requests` 明细表（`request_id` / `user_hash` / 置信度分档 /
权限组 / token / 各段耗时），`request_id` 经 SSE `trace` 事件下发前端，用户报障可直接提供。

**审计**：`logs/audit-YYYYMMDD.jsonl`，落盘前做 PII 掩码（手机号/身份证/邮箱/银行卡/
API Key/私钥/IP），支持按 `AUDIT_RETENTION_DAYS` 清理；`AUDIT_STORE_TEXT=0` 可完全不落原文。

**离线验证**：`python scripts/check_governance.py`（判定逻辑）与
`python scripts/verify_acl_e2e.py`（真实 HTTP 端到端）。

### SSE 事件协议

`POST /api/chat` 的响应是 `text/event-stream`，帧格式为 `event: <name>` + 单行 JSON 的 `data:`：

```
event: conversation
data: {"conversation_id":"...","title":"员工手册里试用期是多久？"}

event: sources
data: [{"index":1,"doc_name":"员工手册.pdf","section_path":"第二章 > 2.2 试用期","page":3,"similarity":0.73,...}]

event: status
data: {"stage":"generating"}

event: reasoning
data: {"delta":"…"}      # 思考过程增量（模型支持时才有）

event: content
data: {"delta":"根据"}    # 正文增量，逐 token

event: done
data: {"message_id":12,"latency":1.8,"ttft":1.1,"prompt_tokens":587,...}
```

出错时下发 `event: error`，`data` 为 `{"detail": "...", "code": "..."}`。

**为什么来源先于正文**：`sources` 在检索完成后、生成开始前立即下发，
前端可以在第一个 token 到达之前就把引用卡片渲染出来，观感上"检索与生成分离"。

## 环境变量

后端**新增**的配置见下表；大模型、Embedding、向量库、检索等配置**复用** `rag/config.py`
已有的环境变量（`API_KEY` / `BASE_URL` / `LLM_MODEL` / `MODEL_OPTIONS` / `EMBED_MODEL` /
`INDEX_DIR` / `COLLECTION_NAME` / `REDIS_URL` …），完整清单见项目根目录 `.env.example`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `API_KEYS` | 空 | 服务端密钥，逗号分隔；**空 = 不鉴权**（与旧版同名） |
| `DATABASE_URL` | `sqlite:///./backend/data/conversations.db` | 会话库连接串，可换 PostgreSQL |
| `UPLOAD_DIR` | `backend/uploads` | 上传文件落盘目录 |
| `MAX_UPLOAD_MB` | `50` | 单文件大小上限 |
| `INGEST_ON_UPLOAD` | `true` | 上传后是否立即向量化入库 |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | 允许跨域的前端地址（逗号分隔） |
| `HISTORY_ROUNDS` | `3` | 参与改写与生成的最近对话轮数 |
| `TITLE_MAX_LEN` | `20` | 会话标题取首条消息的前 N 字 |
| `DEFAULT_TOP_K` | `5` | 送入大模型的片段数 |

## 鉴权

`API_KEYS` 留空时不鉴权（本地开发零配置）；配置后，**除 `/api/health` 外**的所有
`/api/*` 端点都要求：

```
Authorization: Bearer <key>      或      X-API-Key: <key>
```

- 逗号分隔支持多把密钥，便于轮换；
- 只比对 SHA-256 摘要，并用 `secrets.compare_digest` 做常量时间比较——
  密钥不会出现在日志或 401 响应里，也避免按字符提前返回的时序侧信道；
- `/api/health` 刻意豁免：容器 HEALTHCHECK、负载均衡探针与前端"知识库是否就绪"
  都要能无凭据访问，它不返回任何业务数据。是否启用鉴权通过响应里的
  `auth_required` 字段如实告知前端；
- 离线验证：`python scripts/check_auth.py`。

## 重新生成（regenerate）

`POST /api/chat` 支持 `regenerate: true`：复用最近一条用户提问重跑，并**删除**该提问之后的回答。

放在后端做是因为一次问答在库里是 (user, assistant) 两条消息，"重新生成"语义上是**替换**这一轮，
若让前端直接重发同一条消息，历史里会出现两条一模一样的提问。此时 `message` 可省略；
会话里没有可重跑的提问时返回 `event: error` 且 `code: nothing_to_regenerate`。

> 实现提示：删除旧回答后必须 `db.expire_all()`。否则紧接着的历史查询会从 Session 的
> identity map 里命中已删除实例，抛 `ObjectDeletedError`（表现为"会话初始化失败"）。

## 设计说明

**为什么自己管理消息历史，而不用 `ConversationBufferMemory`**：
对话要能在刷新页面、换设备后完整还原（含引用来源与耗时统计），必须落库；
LangChain 的内存对象活不过进程重启，且只保留纯文本、会丢掉 sources。
因此历史存进数据库，每轮按"最近 N 轮"取出来喂给模型。

**同步 IO 与事件循环**：检索（嵌入 + Chroma + BM25 + 重排）与生成都是阻塞调用，
统一用 `asyncio.to_thread` 丢进线程池；流式生成在线程里消费同步迭代器，
再通过 `loop.call_soon_threadsafe` 把事件投回事件循环，避免阻塞其他请求。

**停止生成**：客户端 abort 触发 `asyncio.CancelledError`，后端把**已生成的部分回答**保存后再退出，
所以刷新页面能看到被中止的回答，而不是凭空消失。

## 自检

```bash
python scripts/check_backend.py        # 离线自检：配置/建表/CRUD/路由契约
python scripts/verify_backend_e2e.py   # 端到端：会话 + 上传 + SSE + 持久化（需能访问大模型）
```
