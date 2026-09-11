/**
 * 消息气泡。
 *
 * 用户消息：右对齐、浅蓝底、大圆角。
 * AI 消息：左对齐、无气泡底色（DeepSeek 风格），顶部有 Logo，底部有操作栏。
 */

import { useState } from 'react'
import type { Confidence, SourceItem } from '@/types'
import { MarkdownRenderer } from '@/components/MarkdownRenderer'
import { SourceCards } from '@/components/SourceCards'
import { useCopy } from '@/hooks/useCopy'
import {
  IconAlert,
  IconBrain,
  IconCheck,
  IconChevron,
  IconCopy,
  IconRefresh,
  IconSparkle,
  IconThumbDown,
  IconThumbUp,
  Logo,
} from '@/components/Icons'

/** 思考过程折叠块（模型返回 reasoning_content 时才出现）。 */
export function ReasoningBlock({
  reasoning,
  streaming = false,
}: {
  reasoning: string
  streaming?: boolean
}) {
  const [open, setOpen] = useState(false)
  if (!reasoning.trim()) return null

  return (
    <div className="mb-3 overflow-hidden rounded-xl border border-line bg-surface">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-[12.5px] text-ink-soft transition-colors hover:bg-hover"
        aria-expanded={open}
      >
        <IconBrain size={14} className="text-brand" />
        <span className="font-medium">{streaming ? '正在思考…' : '已深度思考'}</span>
        <IconChevron
          size={15}
          className={`ml-auto transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {(open || streaming) && (
        <div className="animate-fade-in border-t border-line px-3 py-2.5">
          <p className="whitespace-pre-wrap text-[12.5px] leading-6 text-ink-muted">
            {reasoning}
          </p>
        </div>
      )}
    </div>
  )
}

/** 「正在思考…」三点动画。 */
export function TypingDots({ label = '正在思考' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[14px] text-ink-muted">
      <span className="inline-flex items-center gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-current animate-pulse-dot"
            style={{ animationDelay: `${i * 0.16}s` }}
          />
        ))}
      </span>
      {label && <span>{label}</span>}
    </span>
  )
}

interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sources?: SourceItem[]
  /** 正在流式输出中（显示光标与思考动画） */
  streaming?: boolean
  /** 生成中但还没有任何内容（显示"正在思考…"） */
  pending?: boolean
  error?: string
  cacheHit?: boolean
  latency?: number
  retrievalLatency?: number
  ttft?: number
  citationWarning?: string
  /** 检索置信度：medium 时提示"仅供参考"；low 表示这是系统按阈值拒答 */
  confidence?: Confidence | null
  /** 本次请求的追踪 ID（排障与反馈关联用） */
  requestId?: string
  onRegenerate?: () => void
  /** 赞/踩回流：交给上层落库，成为在线质量信号 */
  onVote?: (vote: 'up' | 'down') => void
}

export function MessageBubble({
  role,
  content,
  reasoning = '',
  sources = [],
  streaming = false,
  pending = false,
  error = '',
  cacheHit = false,
  latency = 0,
  retrievalLatency = 0,
  ttft = 0,
  citationWarning = '',
  confidence = null,
  requestId = '',
  onRegenerate,
  onVote,
}: MessageBubbleProps) {
  const { copied, copy } = useCopy()
  const [vote, setVote] = useState<'up' | 'down' | null>(null)

  const castVote = (v: 'up' | 'down') => {
    const next = vote === v ? null : v
    setVote(next)
    // 只上报"点选"，取消选中的不上报（避免噪声）
    if (next) onVote?.(next)
  }

  // ---------------- 用户消息 ----------------
  if (role === 'user') {
    return (
      <div className="flex animate-fade-in justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-bubble px-4 py-2.5 text-[15px] leading-7 text-ink">
          {content}
        </div>
      </div>
    )
  }

  // ---------------- AI 消息 ----------------
  const meta: string[] = []
  if (retrievalLatency > 0) meta.push(`检索 ${retrievalLatency.toFixed(1)}s`)
  if (ttft > 0) meta.push(`首字 ${ttft.toFixed(1)}s`)
  if (latency > 0) meta.push(`耗时 ${latency.toFixed(1)}s`)
  if (cacheHit) meta.push('缓存命中')

  return (
    <div className="flex animate-fade-in gap-3">
      <div className="mt-0.5 shrink-0">
        <Logo size={28} />
      </div>

      <div className="min-w-0 flex-1">
        <ReasoningBlock reasoning={reasoning} streaming={streaming && !content} />

        {citationWarning && (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-danger/30 bg-danger/5 px-3 py-2 text-[12.5px] text-danger">
            <IconAlert size={14} className="mt-0.5 shrink-0" />
            <span>{citationWarning}</span>
          </div>
        )}

        {/* 置信度提示：low = 系统按阈值拒答（未调用大模型），medium = 仅供参考 */}
        {confidence && confidence.tier !== 'high' && confidence.hint && (
          <div
            className={`mb-2 flex items-start gap-2 rounded-xl border px-3 py-2 text-[12.5px] ${
              confidence.tier === 'low'
                ? 'border-line bg-surface text-ink-soft'
                : 'border-brand/30 bg-brand/5 text-brand'
            }`}
            data-testid={`confidence-${confidence.tier}`}
          >
            <IconSparkle size={14} className="mt-0.5 shrink-0" />
            <span>
              {confidence.hint}
              {confidence.score > 0 && (
                <span className="ml-1 text-ink-muted">
                  （相关度 {Math.round(confidence.score * 100)}%）
                </span>
              )}
            </span>
          </div>
        )}

        {pending && !content ? (
          <div className="py-1">
            <TypingDots />
          </div>
        ) : (
          <MarkdownRenderer
            content={content}
            className={streaming ? 'stream-caret' : undefined}
          />
        )}

        {error && (
          <div className="mt-2 flex items-start gap-2 rounded-xl border border-danger/30 bg-danger/5 px-3 py-2 text-[12.5px] text-danger">
            <IconAlert size={14} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <SourceCards sources={sources} />

        {/* 操作栏：生成中不显示，避免误触 */}
        {!streaming && !pending && content && (
          <div className="mt-2.5 flex items-center gap-0.5 text-ink-muted">
            <button
              type="button"
              className="icon-btn h-7 w-7"
              onClick={() => copy(content)}
              title={copied ? '已复制' : '复制'}
              aria-label="复制回答"
            >
              {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
            </button>

            {onRegenerate && (
              <button
                type="button"
                className="icon-btn h-7 w-7"
                onClick={onRegenerate}
                title="重新生成"
                aria-label="重新生成"
              >
                <IconRefresh size={15} />
              </button>
            )}

            <span className="mx-1 h-3.5 w-px bg-line" />

            <button
              type="button"
              className={`icon-btn h-7 w-7 ${vote === 'up' ? 'text-brand' : ''}`}
              onClick={() => castVote('up')}
              title="回答得不错"
              aria-label="点赞"
              aria-pressed={vote === 'up'}
            >
              <IconThumbUp size={15} />
            </button>
            <button
              type="button"
              className={`icon-btn h-7 w-7 ${vote === 'down' ? 'text-danger' : ''}`}
              onClick={() => castVote('down')}
              title="回答需要改进（会记录以便改进检索）"
              aria-label="点踩"
              aria-pressed={vote === 'down'}
            >
              <IconThumbDown size={15} />
            </button>

            {vote && (
              <span className="ml-1 text-[11px] text-ink-muted">
                {vote === 'up' ? '感谢反馈' : '已记录，会用于改进'}
              </span>
            )}

            {meta.length > 0 && (
              <span className="ml-2 text-[11px] text-ink-muted">{meta.join(' · ')}</span>
            )}
            {requestId && (
              <span
                className="ml-2 cursor-help text-[10px] text-ink-muted/60"
                title={`追踪 ID：${requestId}（报障时请提供）`}
              >
                #{requestId.slice(0, 8)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
