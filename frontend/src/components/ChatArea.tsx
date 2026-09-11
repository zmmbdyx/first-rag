/**
 * 中央对话区。
 *
 * 空状态：居中 Logo + 欢迎语 + 推荐问题卡片（点击直接提问）。
 * 对话态：消息列表 + 自动滚动到底 + 生成中的"正在思考…"。
 */

import { useEffect, useMemo, useRef } from 'react'
import type { Message, SourceItem } from '@/types'
import { MessageBubble } from '@/components/MessageBubble'
import { useAppStore } from '@/store/useAppStore'
import { IconSparkle, Logo } from '@/components/Icons'

/** 推荐问题：覆盖资料检索、流程类与制度类，方便第一次体验。 */
const SUGGESTIONS = [
  { icon: '📘', text: '员工手册里试用期是多久？' },
  { icon: '🛠️', text: 'IT服务台的报修流程是什么？' },
  { icon: '🔐', text: '信息安全制度对账号密码有什么要求？' },
  { icon: '🌴', text: '年假是怎么规定的？' },
]

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <div className="w-full max-w-chat">
        <div className="mb-10 flex flex-col items-center text-center">
          <Logo size={56} />
          <h1 className="mt-5 text-[26px] font-semibold tracking-tight text-ink">
            知识库智能问答
          </h1>
          <p className="mt-2 text-[15px] text-ink-soft">
            基于企业文档的检索增强问答，回答附带可核对的引用来源
          </p>
        </div>

        <p className="mb-3 flex items-center justify-center gap-1.5 text-[15px] text-ink">
          <IconSparkle size={16} className="text-brand" />
          今天有什么可以帮到你？
        </p>

        <div className="grid gap-2.5 sm:grid-cols-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s.text}
              type="button"
              onClick={() => onPick(s.text)}
              className="group flex items-start gap-2.5 rounded-2xl border border-line bg-elevated px-4 py-3.5 text-left transition-all duration-150 hover:-translate-y-0.5 hover:border-brand/40 hover:shadow-card"
            >
              <span className="text-[16px] leading-6">{s.icon}</span>
              <span className="text-[13.5px] leading-6 text-ink-soft transition-colors group-hover:text-ink">
                {s.text}
              </span>
            </button>
          ))}
        </div>

        <p className="mt-8 text-center text-[12px] text-ink-muted">
          也可以点击左下角回形针上传 PDF / Word / TXT / Markdown 扩充知识库
        </p>
      </div>
    </div>
  )
}

interface ChatAreaProps {
  messages: Message[]
  loading: boolean
  onSuggestion: (q: string) => void
  onRegenerate: () => void
  /** 回答反馈回流（👍/👎） */
  onVote?: (vote: 'up' | 'down', ctx: { requestId: string; conversationId: string }) => void
}

export function ChatArea({
  messages,
  loading,
  onSuggestion,
  onRegenerate,
  onVote,
}: ChatAreaProps) {
  const stream = useAppStore((s) => s.stream)
  const activeId = useAppStore((s) => s.activeId)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 当前会话正在进行流式生成
  const generating = Boolean(stream.conversationId) && stream.conversationId === activeId

  // 是否用户已经手动向上滚动（此时不强制拉到底，避免打断阅读）
  const pinnedRef = useRef(true)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight
      pinnedRef.current = distance < 80
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  // 内容变化时滚动到底（流式输出时跟随）
  const streamLength = stream.content.length + stream.reasoning.length
  useEffect(() => {
    if (pinnedRef.current) {
      bottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
    }
  }, [streamLength, messages.length, generating])

  // 切换会话时强制回到底部
  useEffect(() => {
    pinnedRef.current = true
    bottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
  }, [activeId])

  // 最后一条 AI 回答才显示"重新生成"
  const lastAssistantIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') return i
    }
    return -1
  }, [messages])

  const isEmpty = messages.length === 0 && !generating

  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
      {isEmpty && !loading ? (
        <EmptyState onPick={onSuggestion} />
      ) : (
        <div className="mx-auto flex w-full max-w-chat flex-col gap-6 px-4 py-6">
          {loading && (
            <p className="py-8 text-center text-[13px] text-ink-muted">正在加载对话…</p>
          )}

          {messages.map((m, i) => (
            <MessageBubble
              key={m.id ?? `${m.role}-${i}-${m.created_at}`}
              role={m.role}
              content={m.content}
              reasoning={m.reasoning}
              sources={m.sources as SourceItem[]}
              error={m.error}
              cacheHit={m.cache_hit}
              latency={m.latency}
              retrievalLatency={m.retrieval_latency}
              ttft={m.ttft}
              onRegenerate={i === lastAssistantIndex ? onRegenerate : undefined}
              onVote={
                onVote && m.role === 'assistant'
                  ? (vote) => onVote(vote, { requestId: '', conversationId: activeId ?? '' })
                  : undefined
              }
            />
          ))}

          {/* 正在生成中的临时回答 */}
          {generating && (
            <>
              {stream.stage && !stream.content && (
                <p className="pl-10 text-[12.5px] text-ink-muted">{stream.stage}</p>
              )}
              <MessageBubble
                role="assistant"
                content={stream.content}
                reasoning={stream.reasoning}
                sources={stream.sources}
                error={stream.error}
                confidence={stream.confidence}
                requestId={stream.requestId}
                streaming
                pending={!stream.content && !stream.error}
                onVote={
                  onVote
                    ? (vote) =>
                        onVote(vote, {
                          requestId: stream.requestId,
                          conversationId: activeId ?? '',
                        })
                    : undefined
                }
              />
            </>
          )}

          <div ref={bottomRef} className="h-px" />
        </div>
      )}
    </div>
  )
}
