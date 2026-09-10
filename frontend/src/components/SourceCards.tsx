/**
 * 引用来源卡片（RAG 专属）。
 *
 * 默认折叠成一行摘要，点击展开看原文片段。
 * 每张卡片显示：文档名、章节/页码定位、相似度、命中路径，以及原文摘录。
 */

import { useState } from 'react'
import type { SourceItem } from '@/types'
import { IconChevron, IconDoc, IconLink } from '@/components/Icons'

/** 命中路径 -> 中文标签。 */
const VIA_LABEL: Record<string, string> = {
  vector: '向量',
  bm25: '关键词',
  keyword: '关键词',
  rerank: '重排',
  hybrid: '混合',
}

function similarityTone(sim: number | null): string {
  if (sim === null) return 'text-ink-muted'
  if (sim >= 0.7) return 'text-success'
  if (sim >= 0.5) return 'text-brand'
  return 'text-ink-muted'
}

function SourceCard({ source, defaultOpen = false }: { source: SourceItem; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const simText = source.similarity === null ? '—' : `${Math.round(source.similarity * 100)}%`

  return (
    <div className="overflow-hidden rounded-xl border border-line bg-elevated transition-colors hover:border-brand/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2.5 px-3 py-2.5 text-left"
        aria-expanded={open}
      >
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-brand/10 text-[11px] font-semibold text-brand">
          {source.index}
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <IconDoc size={13} className="shrink-0 text-ink-muted" />
            <span className="truncate text-[13px] font-medium text-ink" title={source.doc_name}>
              {source.doc_name || '未知文档'}
            </span>
            {source.has_table && (
              <span className="shrink-0 rounded bg-hover px-1.5 py-px text-[10px] text-ink-muted">
                表格
              </span>
            )}
          </span>

          {!open && (
            <span className="mt-0.5 line-clamp-1 block text-[12px] text-ink-muted">
              {source.snippet}
            </span>
          )}

          <span className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-ink-muted">
            {source.section_path && <span className="truncate">{source.section_path}</span>}
            {source.page >= 0 && <span>第 {source.page} 页</span>}
            <span className={similarityTone(source.similarity)}>相似度 {simText}</span>
            {source.rerank_score !== null && <span>重排 {source.rerank_score.toFixed(2)}</span>}
            {source.sources?.length > 0 && (
              <span className="inline-flex items-center gap-1">
                <IconLink size={11} />
                {source.sources.map((v) => VIA_LABEL[v] ?? v).join(' + ')}
              </span>
            )}
          </span>
        </span>

        <IconChevron
          size={16}
          className={`mt-0.5 shrink-0 text-ink-muted transition-transform duration-200 ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>

      {open && (
        <div className="animate-fade-in border-t border-line bg-surface px-3 py-2.5">
          <p className="whitespace-pre-wrap text-[12.5px] leading-6 text-ink-soft">
            {source.snippet || '（无原文片段）'}
          </p>
        </div>
      )}
    </div>
  )
}

export function SourceCards({ sources }: { sources: SourceItem[] }) {
  const [expanded, setExpanded] = useState(false)
  if (!sources || sources.length === 0) return null

  const shown = expanded ? sources : sources.slice(0, 3)
  const hidden = sources.length - shown.length

  return (
    <div className="mt-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-ink-muted">
        <IconDoc size={13} />
        <span>引用来源 · {sources.length} 条</span>
      </div>
      <div className="grid gap-2">
        {shown.map((s) => (
          <SourceCard key={`${s.index}-${s.doc_name}-${s.section_path}`} source={s} />
        ))}
      </div>
      {sources.length > 3 && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 text-[12px] text-brand hover:text-brand-hover"
        >
          {expanded ? '收起' : `展开剩余 ${hidden} 条`}
        </button>
      )}
    </div>
  )
}
