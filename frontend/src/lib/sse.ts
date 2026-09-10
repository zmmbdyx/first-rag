/**
 * SSE 流式客户端。
 *
 * 后端用 POST + `text/event-stream`，浏览器原生的 EventSource 只支持 GET，
 * 所以这里用 fetch + ReadableStream 手工解析 SSE 帧。
 *
 * 协议（见 backend/api/chat.py）：
 *   event: <name>\ndata: <单行 JSON>\n\n
 */

import type { ChatRequest, SSEEvent } from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

/** 把一段 SSE 文本切成若干帧（一帧以空行结束）。 */
function parseFrame(raw: string): SSEEvent | null {
  let eventName = 'message'
  const dataLines: string[] = []

  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      // 规范允许 "data:" 后跟一个可选空格
      dataLines.push(line.slice(5).replace(/^ /, ''))
    }
    // id: / retry: / 注释行（以 : 开头）一律忽略
  }

  if (dataLines.length === 0) return null

  let payload: unknown
  try {
    payload = JSON.parse(dataLines.join('\n'))
  } catch {
    // 后端保证 data 是单行 JSON；解析失败时不要静默吞掉，交给调用方提示
    return { type: 'error', data: { detail: '无法解析服务端事件数据' } }
  }

  return { type: eventName, data: payload } as SSEEvent
}

/**
 * 发起一次流式问答，逐个 yield 事件。
 *
 * @param req      请求体
 * @param signal   AbortSignal —— 前端"停止生成"时 abort，已收到的内容会保留
 */
export async function* streamChat(
  req: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent, void, undefined> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    let detail = `请求失败（${res.status}）`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* 忽略：保持默认提示 */
    }
    throw new Error(detail)
  }
  if (!res.body) throw new Error('当前浏览器不支持流式响应（ReadableStream 不可用）')

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // SSE 帧之间用空行分隔；\r\n\r\n 兼容部分反向代理
      let boundary = buffer.search(/\r?\n\r?\n/)
      while (boundary !== -1) {
        const raw = buffer.slice(0, boundary)
        const match = /\r?\n\r?\n/.exec(buffer.slice(boundary))
        buffer = buffer.slice(boundary + (match ? match[0].length : 2))
        const evt = parseFrame(raw)
        if (evt) yield evt
        boundary = buffer.search(/\r?\n\r?\n/)
      }
    }

    // 收尾：处理没有以空行结束的最后一帧
    const tail = parseFrame(buffer)
    if (tail) yield tail
  } finally {
    // 无论正常结束、抛错还是 abort，都要释放底层连接
    reader.cancel().catch(() => undefined)
  }
}
