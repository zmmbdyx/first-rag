/** 与后端 API 契约一一对应的类型定义。 */

/** 一条引用来源（对应后端 backend/schemas/common.py::SourceItem）。 */
export interface SourceItem {
  /** 引用编号，与答案正文里的 [n] 对应 */
  index: number
  doc_name: string
  /** 章节路径，如「第二章 入职与试用期 > 2.2 试用期」 */
  section_path: string
  /** 页码，-1 表示无页码信息（如 TXT） */
  page: number
  location: string
  /** 融合检索分数 */
  score: number
  /** 重排分数，未开启重排时为 null */
  rerank_score: number | null
  has_table: boolean
  /** 命中路径，如 ['vector', 'bm25'] */
  sources: string[]
  /** 原文片段 */
  snippet: string
  /** 归一化相似度 0~1 */
  similarity: number | null
}

/** 会话概要（GET /api/conversations）。 */
export interface Conversation {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

/** 一条历史消息（GET /api/conversations/{id}/messages）。 */
export interface Message {
  id: number | null
  role: 'user' | 'assistant'
  content: string
  reasoning: string
  sources: SourceItem[]
  created_at: string
  model: string
  latency: number
  retrieval_latency: number
  ttft: number
  cache_hit: boolean
  error: string
}

/** POST /api/chat 请求体。 */
export interface ChatRequest {
  message: string
  conversation_id: string
  model?: string
  mode?: 'vector' | 'keyword' | 'hybrid'
  top_k?: number
  temperature?: number
  thinking?: boolean
  use_cache?: boolean
  /** 重新生成最后一条回答：复用最近提问并覆盖旧回答，不追加重复提问 */
  regenerate?: boolean
}

/** 检索置信度分档（后端统一判定，前端只负责展示）。 */
export interface Confidence {
  tier: 'high' | 'medium' | 'low'
  score: number
  basis: string
  hint: string
  refuse: boolean
}

/** SSE done 事件的负载。 */
export interface ChatDone {
  message_id?: number | null
  conversation_id: string
  title?: string
  latency?: number
  retrieval_latency?: number
  ttft?: number
  prompt_tokens?: number
  completion_tokens?: number
  cache_hit?: boolean
  citations?: number[]
  forged_citations?: number[]
  citation_warning?: string
  query_used?: string
  rewrite_method?: string
  /** 贯穿本次请求的追踪 ID，用户报障时可直接提供 */
  request_id?: string
  /** 检索置信度；low 时后端已直接拒答（未调用大模型） */
  confidence?: Confidence
  /** 值为 "confidence" 表示本次是按阈值拒答，而非模型自己说不知道 */
  refused_by?: string
}

/** POST /api/upload 响应。 */
export interface UploadResponse {
  file_id: string
  filename: string
  status: 'done' | 'processing' | 'failed'
  size: number
  chunks: number
  collection_count: number
  message: string
}

/** GET /api/health 响应。 */
export interface HealthResponse {
  status: 'ok' | 'degraded'
  collection: string
  chunks: number | null
  bm25_ready: boolean
  retriever_ready: boolean
  models: string[]
  cache: Record<string, unknown>
  upload_dir: string
  default_top_k: number
  /** 后端是否启用了 API Key 鉴权（部署时配置 API_KEYS 才为 true） */
  auth_required: boolean
}

/** 前端本地维护的"正在流式生成"的回答。 */
export interface StreamingState {
  /** 正文 */
  content: string
  /** 思考过程 */
  reasoning: string
  /** 当前阶段提示 */
  stage: string
  sources: SourceItem[]
  error: string
}

/** SSE 事件联合类型（由 lib/sse.ts 解析产出）。 */
export type SSEEvent =
  | { type: 'conversation'; data: { conversation_id: string; title: string } }
  | { type: 'trace'; data: { request_id: string } }
  | { type: 'rewrite'; data: { query: string; method: string } }
  | { type: 'sources'; data: SourceItem[] }
  | { type: 'status'; data: { stage: string } }
  | { type: 'reasoning'; data: { delta: string } }
  | { type: 'content'; data: { delta: string } }
  | { type: 'done'; data: ChatDone }
  | { type: 'error'; data: { detail: string; code?: string } }
