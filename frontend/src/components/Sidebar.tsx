/**
 * 左侧边栏（约 260px，可折叠）。
 *
 * - 顶部：「开启新对话」按钮（品牌蓝高亮）
 * - 中部：搜索框 + 历史对话列表（按 今天/昨天/近7天/更早 分组）
 *           悬停显示重命名、删除
 * - 底部：知识库状态 + 深色/浅色切换
 */

import { useMemo, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { deleteConversation, renameConversation } from '@/lib/api'
import { relativeTime, timeGroupOf, type TimeGroup } from '@/lib/date'
import { queryKeys, useConversations, useHealth } from '@/hooks/useChat'
import { useAppStore } from '@/store/useAppStore'
import type { Conversation } from '@/types'
import {
  IconCheck,
  IconDatabase,
  IconEdit,
  IconMoon,
  IconPanel,
  IconPlus,
  IconSearch,
  IconSun,
  IconTrash,
  IconX,
  Logo,
} from '@/components/Icons'

const GROUP_ORDER: TimeGroup[] = ['今天', '昨天', '近7天', '更早']

interface SidebarProps {
  /** 桌面端折叠态（移动端抽屉始终展开内容） */
  collapsed: boolean
  onClose: () => void
  onNewConversation: () => void
}

export function Sidebar({ collapsed, onClose, onNewConversation }: SidebarProps) {
  const qc = useQueryClient()
  const { data: conversations = [], isLoading } = useConversations()
  const { data: health } = useHealth()

  const activeId = useAppStore((s) => s.activeId)
  const setActiveId = useAppStore((s) => s.setActiveId)
  const theme = useAppStore((s) => s.theme)
  const toggleTheme = useAppStore((s) => s.toggleTheme)
  const toggleSidebarCollapsed = useAppStore((s) => s.toggleSidebarCollapsed)

  const [query, setQuery] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [confirmId, setConfirmId] = useState<string | null>(null)
  // 悬停态用显式 state 管理，而不是 group-hover CSS 类。
  // 原因：Tailwind 把 `.group:hover .group-hover\:hidden` 与 `...:flex` 一起生成时，
  // display:none 排在后面，会盖掉 display:flex —— 结果是悬停后按钮永远不出现、
  // 时间文字被按钮压住重叠。用 state 做互斥渲染，行为确定且可测。
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  // 关键词过滤（大小写不敏感）
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q
      ? conversations.filter((c) => c.title.toLowerCase().includes(q))
      : conversations
    const map = new Map<TimeGroup, Conversation[]>()
    for (const c of filtered) {
      const g = timeGroupOf(c.updated_at)
      const arr = map.get(g) ?? []
      arr.push(c)
      map.set(g, arr)
    }
    return GROUP_ORDER.map((g) => [g, map.get(g) ?? []] as const).filter(([, v]) => v.length > 0)
  }, [conversations, query])

  const rename = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => renameConversation(id, title),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.conversations })
      setEditingId(null)
    },
  })

  const remove = useMutation({
    mutationFn: deleteConversation,
    onSuccess: (_res, id) => {
      qc.invalidateQueries({ queryKey: queryKeys.conversations })
      qc.removeQueries({ queryKey: queryKeys.messages(id) })
      if (activeId === id) setActiveId(null)
      setConfirmId(null)
    },
  })

  const startRename = (c: Conversation) => {
    setEditingId(c.conversation_id)
    setEditValue(c.title)
  }

  const commitRename = () => {
    const title = editValue.trim()
    if (editingId && title) rename.mutate({ id: editingId, title })
    else setEditingId(null)
  }

  const chunks = health?.chunks
  const ready = health?.retriever_ready ?? false

  // 折叠态：只留一条窄栏（桌面端）
  if (collapsed) {
    return (
      <div className="flex h-full w-14 flex-col items-center gap-2 border-r border-line bg-surface py-3">
        <button
          type="button"
          onClick={toggleSidebarCollapsed}
          className="icon-btn h-9 w-9"
          title="展开侧边栏"
          aria-label="展开侧边栏"
        >
          <IconPanel size={18} />
        </button>
        <button
          type="button"
          onClick={onNewConversation}
          className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand text-brand-fg transition-colors hover:bg-brand-hover"
          title="开启新对话"
          aria-label="开启新对话"
        >
          <IconPlus size={18} />
        </button>
        <button
          type="button"
          onClick={toggleTheme}
          className="icon-btn mt-auto h-9 w-9"
          title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
          aria-label="切换主题"
        >
          {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
        </button>
      </div>
    )
  }

  return (
    <aside className="flex h-full w-sidebar flex-col border-r border-line bg-surface">
      {/* 顶部：品牌 + 关闭（移动端） */}
      <div className="flex items-center gap-2 px-3 pb-2 pt-3">
        <Logo size={26} />
        <span className="flex-1 truncate text-[14px] font-semibold text-ink">知识库问答</span>
        <button
          type="button"
          onClick={toggleSidebarCollapsed}
          className="icon-btn hidden h-8 w-8 md:inline-flex"
          title="收起侧边栏"
          aria-label="收起侧边栏"
        >
          <IconPanel size={17} />
        </button>
        <button
          type="button"
          onClick={onClose}
          className="icon-btn h-8 w-8 md:hidden"
          title="关闭"
          aria-label="关闭侧边栏"
        >
          <IconX size={17} />
        </button>
      </div>

      {/* 新对话 */}
      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={onNewConversation}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-3 py-2.5 text-[14px] font-medium text-brand-fg transition-colors hover:bg-brand-hover"
        >
          <IconPlus size={17} />
          开启新对话
        </button>
      </div>

      {/* 搜索 */}
      <div className="px-3 pb-2">
        <div className="flex items-center gap-2 rounded-lg bg-elevated px-2.5 py-1.5 transition-colors focus-within:ring-1 focus-within:ring-brand/40">
          <IconSearch size={15} className="shrink-0 text-ink-muted" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索历史对话"
            className="min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-muted"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="icon-btn h-5 w-5 shrink-0"
              aria-label="清除搜索"
            >
              <IconX size={12} />
            </button>
          )}
        </div>
      </div>

      {/* 历史列表 */}
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {isLoading && (
          <p className="px-2 py-3 text-[12.5px] text-ink-muted">正在加载历史对话…</p>
        )}

        {!isLoading && groups.length === 0 && (
          <p className="px-2 py-3 text-[12.5px] text-ink-muted">
            {query ? '没有匹配的对话' : '还没有对话，去提第一个问题吧'}
          </p>
        )}

        {groups.map(([group, items]) => (
          <div key={group} className="mb-1">
            <div className="px-2.5 py-1.5 text-[11px] font-medium text-ink-muted">{group}</div>
            {items.map((c) => {
              const active = c.conversation_id === activeId
              const editing = editingId === c.conversation_id
              const confirming = confirmId === c.conversation_id
              const hovered = hoveredId === c.conversation_id
              // 悬停时让位给操作按钮；正在确认删除时始终显示按钮
              const showActions = confirming || (hovered && !editing)

              return (
                <div
                  key={c.conversation_id}
                  className={`conv-item ${
                    active ? 'bg-brand/10 text-brand' : 'text-ink-soft hover:bg-hover hover:text-ink'
                  }`}
                  onClick={() => {
                    if (!editing) {
                      setActiveId(c.conversation_id)
                      onClose()
                    }
                  }}
                  onMouseEnter={() => setHoveredId(c.conversation_id)}
                  onMouseLeave={() => setHoveredId((v) => (v === c.conversation_id ? null : v))}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !editing) {
                      setActiveId(c.conversation_id)
                      onClose()
                    }
                  }}
                >
                  {editing ? (
                    <input
                      autoFocus
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        e.stopPropagation()
                        if (e.key === 'Enter') commitRename()
                        if (e.key === 'Escape') setEditingId(null)
                      }}
                      className="min-w-0 flex-1 rounded border border-brand/50 bg-elevated px-1.5 py-0.5 text-[13px] text-ink outline-none"
                    />
                  ) : (
                    <>
                      <span className="min-w-0 flex-1 truncate" title={c.title}>
                        {c.title}
                      </span>

                      {showActions ? (
                        confirming ? (
                          <span
                            className="flex shrink-0 items-center gap-0.5"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={() => remove.mutate(c.conversation_id)}
                              className="icon-btn h-6 w-6 text-danger"
                              title="确认删除"
                              aria-label="确认删除"
                            >
                              <IconCheck size={14} />
                            </button>
                            <button
                              type="button"
                              onClick={() => setConfirmId(null)}
                              className="icon-btn h-6 w-6"
                              title="取消"
                              aria-label="取消删除"
                            >
                              <IconX size={13} />
                            </button>
                          </span>
                        ) : (
                          <span
                            className="flex shrink-0 items-center gap-0.5"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={() => startRename(c)}
                              className="icon-btn h-6 w-6"
                              title="重命名"
                              aria-label="重命名"
                            >
                              <IconEdit size={13} />
                            </button>
                            <button
                              type="button"
                              onClick={() => setConfirmId(c.conversation_id)}
                              className="icon-btn h-6 w-6 hover:text-danger"
                              title="删除"
                              aria-label="删除"
                            >
                              <IconTrash size={13} />
                            </button>
                          </span>
                        )
                      ) : (
                        <span className="shrink-0 text-[10.5px] text-ink-muted">
                          {relativeTime(c.updated_at)}
                        </span>
                      )}
                    </>
                  )}
                </div>
              )
            })}
          </div>
        ))}
      </div>

      {/* 底部：知识库状态 + 主题切换 */}
      <div className="border-t border-line px-3 py-2.5">
        {health && (
          <div className="mb-2 flex items-center gap-2 text-[11.5px] text-ink-muted">
            <IconDatabase size={13} className={ready ? 'text-success' : 'text-danger'} />
            <span className="flex-1 truncate">
              {ready ? `知识库 ${chunks ?? 0} 个切片` : '知识库未就绪'}
            </span>
          </div>
        )}
        <button
          type="button"
          onClick={toggleTheme}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-[13px] text-ink-soft transition-colors hover:bg-hover hover:text-ink"
        >
          {theme === 'dark' ? <IconSun size={16} /> : <IconMoon size={16} />}
          <span className="flex-1 text-left">{theme === 'dark' ? '浅色模式' : '深色模式'}</span>
        </button>
      </div>
    </aside>
  )
}
