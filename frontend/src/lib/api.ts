/** REST 客户端：会话 CRUD 与文档上传。统一相对路径 /api，开发态由 Vite 代理。 */

import type {
  Conversation,
  HealthResponse,
  Message,
  UploadResponse,
} from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

/**
 * 可选的服务端 API Key。
 *
 * 后端 `API_KEYS` 留空时不鉴权（本地开发默认如此），此函数返回空对象即可；
 * 部署时若配置了 `API_KEYS`，前端通过构建期变量注入同一把密钥：
 *     VITE_API_KEY=xxx  npm run build
 * 生产环境更推荐用 Nginx 在反代时注入 `Authorization` 头，
 * 这样密钥不会打进前端产物、也不会出现在浏览器里。
 *
 * 注意：只允许 `VITE_` 前缀的变量进入前端产物 —— 任何被打进产物的值
 * 都等于公开，切勿把大模型的真实密钥（API_KEY / BASE_URL）放到这里。
 */
export function authHeaders(): Record<string, string> {
  const key = import.meta.env.VITE_API_KEY
  return key ? { Authorization: `Bearer ${key}` } : {}
}

/** 把后端的错误响应转成可读的 Error。FastAPI 的 detail 可能是字符串或校验错误数组。 */
async function toError(res: Response): Promise<Error> {
  let detail: unknown
  try {
    const body = await res.json()
    detail = (body as { detail?: unknown }).detail
  } catch {
    detail = await res.text().catch(() => '')
  }
  if (Array.isArray(detail)) {
    const msg = detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string }
        const loc = Array.isArray(item.loc) ? item.loc.join('.') : ''
        return loc ? `${loc}: ${item.msg}` : item.msg
      })
      .filter(Boolean)
      .join('; ')
    return new Error(msg || `请求失败（${res.status}）`)
  }
  return new Error(typeof detail === 'string' && detail ? detail : `请求失败（${res.status}）`)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // 关键：multipart 请求**不能**手动设置 Content-Type —— 必须让浏览器
  // 自己带上 `multipart/form-data; boundary=...`。少写 boundary 时 FastAPI
  // 会把请求体当成 JSON 解析，直接回 422（{"loc":["body","file"],"msg":"Field required"}）。
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...authHeaders(),
      ...init?.headers,
    },
  })
  if (!res.ok) throw await toError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

/** GET /api/health —— 服务与知识库状态。 */
export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

/** GET /api/conversations —— 会话列表（后端按更新时间倒序）。 */
export function fetchConversations(): Promise<Conversation[]> {
  return request<Conversation[]>('/conversations')
}

/** POST /api/conversations —— 新建会话。 */
export function createConversation(payload: {
  title?: string
  model?: string
} = {}): Promise<Conversation> {
  return request<Conversation>('/conversations', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/** GET /api/conversations/{id}/messages —— 某会话的全部消息。 */
export function fetchMessages(conversationId: string): Promise<Message[]> {
  return request<Message[]>(`/conversations/${encodeURIComponent(conversationId)}/messages`)
}

/** DELETE /api/conversations/{id} —— 删除会话。 */
export function deleteConversation(conversationId: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
  })
}

/** PUT /api/conversations/{id}/title —— 重命名会话。 */
export function renameConversation(conversationId: string, title: string): Promise<Conversation> {
  return request<Conversation>(`/conversations/${encodeURIComponent(conversationId)}/title`, {
    method: 'PUT',
    body: JSON.stringify({ title }),
  })
}

/** POST /api/upload —— 上传文档（multipart/form-data）。 */
export function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  // 不设置 Content-Type，交给 request() 的 isFormData 分支处理
  return request<UploadResponse>('/upload', { method: 'POST', body: form })
}
