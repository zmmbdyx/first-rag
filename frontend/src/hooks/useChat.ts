/**
 * 聊天编排：把「SSE 流 → store 缓冲 → react-query 缓存失效」串起来。
 *
 * 数据流：
 *   1. 乐观追加用户消息到 react-query 缓存（气泡立刻出现）；
 *   2. startStream()，逐个消费 SSE 事件写入 store.stream（打字机效果）；
 *   3. 收到 done → 失效 messages/conversations 查询，用服务端落库结果替换临时态。
 */

import { useCallback, useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  createConversation,
  fetchConversations,
  fetchMessages,
  fetchHealth,
} from '@/lib/api'
import { streamChat } from '@/lib/sse'
import { useAppStore } from '@/store/useAppStore'
import type { ChatRequest, Conversation, Message } from '@/types'

export const queryKeys = {
  conversations: ['conversations'] as const,
  messages: (id: string) => ['messages', id] as const,
  health: ['health'] as const,
}

/** 会话列表（按更新时间倒序，后端已排好）。 */
export function useConversations() {
  return useQuery({
    queryKey: queryKeys.conversations,
    queryFn: fetchConversations,
    staleTime: 5_000,
  })
}

/** 某会话的历史消息。 */
export function useMessages(conversationId: string | null) {
  return useQuery({
    queryKey: queryKeys.messages(conversationId ?? ''),
    queryFn: () => fetchMessages(conversationId as string),
    enabled: Boolean(conversationId),
    staleTime: 5_000,
  })
}

/** 服务与知识库状态（侧边栏底部展示切片数等）。 */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: fetchHealth,
    staleTime: 60_000,
    retry: 1,
  })
}

export function useChat() {
  const qc = useQueryClient()
  const abortRef = useRef<AbortController | null>(null)
  // 记住最后一轮用户提问，供"重新生成"使用
  const lastPromptRef = useRef<{ conversationId: string; message: string } | null>(null)

  const {
    startStream,
    appendContent,
    appendReasoning,
    setStreamSources,
    setStreamStage,
    setStreamError,
    endStream,
  } = useAppStore.getState()

  // 组件卸载时中断未完成的流，避免内存泄漏与"幽灵写入"
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  const runStream = useCallback(
    async (req: ChatRequest, optimisticUserMessage: string) => {
      const { activeId, setActiveId } = useAppStore.getState()
      const conversationId = req.conversation_id
      lastPromptRef.current = { conversationId, message: req.message }

      // 1) 乐观上屏用户消息：不等服务端，输入立刻有反馈
      const now = new Date().toISOString()
      const optimistic: Message = {
        id: null,
        role: 'user',
        content: optimisticUserMessage,
        reasoning: '',
        sources: [],
        created_at: now,
        model: '',
        latency: 0,
        retrieval_latency: 0,
        ttft: 0,
        cache_hit: false,
        error: '',
      }
      qc.setQueryData<Message[]>(queryKeys.messages(conversationId), (old) => [
        ...(old ?? []),
        optimistic,
      ])

      // 2) 开始流式
      startStream(conversationId)
      const controller = new AbortController()
      abortRef.current = controller
      let failed = false

      try {
        for await (const evt of streamChat(req, controller.signal)) {
          switch (evt.type) {
            case 'conversation':
              // 后端确认了会话 ID 与自动标题
              if (evt.data.title) {
                qc.setQueryData<Conversation[]>(queryKeys.conversations, (old) =>
                  (old ?? []).map((c) =>
                    c.conversation_id === evt.data.conversation_id
                      ? { ...c, title: evt.data.title }
                      : c,
                  ),
                )
              }
              break
            case 'sources':
              setStreamSources(evt.data)
              break
            case 'rewrite':
              setStreamStage(`已结合上下文改写查询：${evt.data.query}`)
              break
            case 'status':
              setStreamStage(evt.data.stage === 'generating' ? '' : evt.data.stage)
              break
            case 'reasoning':
              appendReasoning(evt.data.delta)
              break
            case 'content':
              appendContent(evt.data.delta)
              break
            case 'error':
              failed = true
              setStreamError(evt.data.detail)
              break
            case 'done':
              break
          }
        }
      } catch (err) {
        // abort 是"用户主动停止"，不是错误：保留已生成内容
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          failed = true
          setStreamError(err instanceof Error ? err.message : String(err))
        }
      } finally {
        abortRef.current = null
        endStream()
        // 3) 用服务端落库结果替换临时态（包含被中止时保存的部分回答）
        await Promise.all([
          qc.invalidateQueries({ queryKey: queryKeys.messages(conversationId) }),
          qc.invalidateQueries({ queryKey: queryKeys.conversations }),
        ])
      }

      if (failed && !activeId) setActiveId(conversationId)
    },
    [
      qc,
      startStream,
      appendContent,
      appendReasoning,
      setStreamSources,
      setStreamStage,
      setStreamError,
      endStream,
    ],
  )

  /** 发送一条消息。会话不存在时先建会话再流式提问。 */
  const send = useCallback(
    async (message: string) => {
      const text = message.trim()
      if (!text) return
      const { activeId, setActiveId, setDraft } = useAppStore.getState()

      let conversationId = activeId
      if (!conversationId) {
        const conv = await createConversation({})
        conversationId = conv.conversation_id
        qc.setQueryData<Conversation[]>(queryKeys.conversations, (old) => [conv, ...(old ?? [])])
        setActiveId(conversationId)
      }
      setDraft('')
      await runStream({ message: text, conversation_id: conversationId }, text)
    },
    [qc, runStream],
  )

  /** 停止生成（后端会把已生成的部分落库）。 */
  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  /**
   * 重新生成最后一条回答：重发同一问题。
   *
   * 说明：后端把每次 `/api/chat` 都视为一轮新对话，因此重新生成会在历史里
   * 再留下一条相同的用户提问（这也是多数聊天产品的行为）。若要做到"原地替换"，
   * 需要后端支持按 message_id 覆盖，当前刻意不做，以免动到既有的问答链路。
   */
  const regenerate = useCallback(async () => {
    const last = lastPromptRef.current
    const { activeId } = useAppStore.getState()
    const conversationId = activeId ?? last?.conversationId
    if (!conversationId) return

    // 优先用本会话记住的最后一问；刷新页面后回退到历史里最后一条用户消息
    const cached = qc.getQueryData<Message[]>(queryKeys.messages(conversationId)) ?? []
    const lastUser = [...cached].reverse().find((m) => m.role === 'user')
    const message =
      last?.conversationId === conversationId ? last.message : lastUser?.content
    if (!message) return

    await runStream({ message, conversation_id: conversationId }, message)
  }, [qc, runStream])

  return { send, stop, regenerate }
}
