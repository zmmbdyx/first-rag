/**
 * 底部输入区（固定在对话区底部）。
 *
 * - 多行输入：Enter 发送、Shift+Enter 换行；高度随内容自增，超过上限转为内部滚动；
 * - 左侧回形针：点击选择或直接拖拽文件到输入区上传（PDF/Word/TXT/Markdown）；
 * - 右侧按钮：有内容时是可点的发送键，生成中变成停止键；
 * - 上方模型下拉：模型清单来自后端 /api/health。
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
} from 'react'
import { useMutation } from '@tanstack/react-query'
import { uploadDocument } from '@/lib/api'
import type { UploadResponse } from '@/types'
import { useAppStore } from '@/store/useAppStore'
import {
  IconAlert,
  IconCheck,
  IconChevron,
  IconPaperclip,
  IconSend,
  IconStop,
  IconX,
} from '@/components/Icons'

const MAX_TEXTAREA_HEIGHT = 200

/** 允许上传的扩展名，与后端 rag.parsers.SUPPORTED_EXTS 保持一致。 */
const ACCEPT = '.pdf,.docx,.txt,.md'

/** 下拉框：模型选择。 */
function ModelPicker({
  models,
  value,
  onChange,
}: {
  models: string[]
  value: string
  onChange: (v: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  if (models.length === 0) return null
  const label = (m: string) => (m.includes('@') ? m.replace('@', ' · ') : m)

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12.5px] text-ink-soft transition-colors hover:bg-hover hover:text-ink"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="max-w-[220px] truncate">{label(value || models[0])}</span>
        <IconChevron
          size={14}
          className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 z-30 mb-1.5 min-w-[200px] animate-fade-in overflow-hidden rounded-xl border border-line bg-elevated py-1 shadow-pop"
        >
          {models.map((m) => (
            <button
              key={m}
              type="button"
              role="option"
              aria-selected={m === value}
              onClick={() => {
                onChange(m)
                setOpen(false)
              }}
              className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] transition-colors hover:bg-hover ${
                m === value ? 'text-brand' : 'text-ink'
              }`}
            >
              <span className="flex-1 truncate">{label(m)}</span>
              {m === value && <IconCheck size={14} />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

interface InputBoxProps {
  models: string[]
  model: string
  onModelChange: (m: string) => void
  onSend: (message: string) => void
  onStop: () => void
  generating: boolean
  /** 知识库未就绪等提示 */
  disabled?: boolean
  disabledHint?: string
}

export function InputBox({
  models,
  model,
  onModelChange,
  onSend,
  onStop,
  generating,
  disabled = false,
  disabledHint = '',
}: InputBoxProps) {
  const draft = useAppStore((s) => s.draft)
  const setDraft = useAppStore((s) => s.setDraft)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  // 高度自适应：先归零再按 scrollHeight 撑开，超过上限则内部滚动
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`
  }, [draft])

  const upload = useMutation<UploadResponse, Error, File>({
    mutationFn: uploadDocument,
    onSuccess: (res) => {
      if (res.status === 'failed') {
        setNotice({ kind: 'err', text: res.message || '上传失败' })
      } else {
        setNotice({
          kind: 'ok',
          text: `${res.filename} 已入库（${res.chunks} 个切片），现在就可以提问了`,
        })
      }
      window.setTimeout(() => setNotice(null), 6000)
    },
    onError: (err) => {
      setNotice({ kind: 'err', text: err.message })
      window.setTimeout(() => setNotice(null), 8000)
    },
  })

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (!file) return
      setNotice(null)
      upload.mutate(file)
    },
    [upload],
  )

  const canSend = draft.trim().length > 0 && !generating && !disabled

  const submit = useCallback(() => {
    if (!canSend) return
    onSend(draft.trim())
  }, [canSend, draft, onSend])

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter 发送，Shift+Enter 换行；输入法组合中的回车不触发发送
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="mx-auto w-full max-w-chat px-4 pb-4">
      {/* 上传结果 / 错误提示 */}
      {notice && (
        <div
          className={`mb-2 flex animate-fade-in items-start gap-2 rounded-xl border px-3 py-2 text-[12.5px] ${
            notice.kind === 'ok'
              ? 'border-success/30 bg-success/5 text-success'
              : 'border-danger/30 bg-danger/5 text-danger'
          }`}
        >
          {notice.kind === 'ok' ? (
            <IconCheck size={14} className="mt-0.5 shrink-0" />
          ) : (
            <IconAlert size={14} className="mt-0.5 shrink-0" />
          )}
          <span className="flex-1">{notice.text}</span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            className="icon-btn h-4 w-4 shrink-0"
            aria-label="关闭提示"
          >
            <IconX size={12} />
          </button>
        </div>
      )}

      {disabled && disabledHint && (
        <div className="mb-2 flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2 text-[12.5px] text-ink-muted">
          <IconAlert size={14} className="shrink-0" />
          <span>{disabledHint}</span>
        </div>
      )}

      <div
        data-testid="composer-dropzone"
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`rounded-2xl border bg-elevated shadow-composer transition-colors duration-150 ${
          dragging ? 'border-brand bg-brand/5' : 'border-line'
        }`}
      >
        {/* 模型选择 */}
        <div className="flex items-center justify-between px-2.5 pt-1.5">
          <ModelPicker models={models} value={model} onChange={onModelChange} />
          {upload.isPending && (
            <span className="pr-1 text-[11.5px] text-ink-muted">正在解析并入库…</span>
          )}
        </div>

        <div className="flex items-end gap-1.5 px-2.5 pb-2">
          {/* 附件上传 */}
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              handleFiles(e.target.files)
              e.target.value = '' // 允许连续上传同一个文件
            }}
          />
          <button
            type="button"
            disabled={disabled || upload.isPending}
            onClick={() => fileRef.current?.click()}
            className="icon-btn mb-0.5 h-9 w-9 shrink-0"
            title="上传文档（PDF / Word / TXT / Markdown）"
            aria-label="上传文档"
          >
            <IconPaperclip size={18} />
          </button>

          {/* 输入框 */}
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            disabled={disabled}
            placeholder={dragging ? '松开即上传文档' : '给知识库提问，或拖入文档…'}
            className="max-h-[200px] min-h-[36px] flex-1 resize-none bg-transparent py-2 text-[15px] leading-6 text-ink outline-none placeholder:text-ink-muted disabled:cursor-not-allowed"
          />

          {/* 发送 / 停止 */}
          {generating ? (
            <button
              type="button"
              onClick={onStop}
              className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ink text-canvas transition-transform hover:scale-105"
              title="停止生成"
              aria-label="停止生成"
            >
              <IconStop size={16} />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              className={`mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition-all ${
                canSend
                  ? 'bg-brand text-brand-fg hover:bg-brand-hover'
                  : 'cursor-not-allowed bg-hover text-ink-muted'
              }`}
              title={canSend ? '发送（Enter）' : '请输入内容'}
              aria-label="发送"
            >
              <IconSend size={17} />
            </button>
          )}
        </div>
      </div>

      <p className="mt-2 text-center text-[11.5px] text-ink-muted">
        内容由 AI 基于知识库生成，请核对引用来源 · Enter 发送 / Shift+Enter 换行
      </p>
    </div>
  )
}
