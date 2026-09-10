/**
 * 全局 UI 状态（zustand）。
 *
 * 职责边界：
 *   - zustand  —— 只放**客户端状态**：当前会话、侧边栏、主题、输入草稿、流式缓冲；
 *   - react-query —— 放**服务端状态**：会话列表与消息历史（缓存、失效、重取）。
 * 二者刻意不重叠，避免"同一份数据两处维护"的经典 bug。
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { SourceItem } from '@/types'

export type Theme = 'light' | 'dark'

/** 一次正在进行的流式生成。 */
export interface StreamState {
  /** 正在生成回答的会话 ID；null 表示当前没有生成任务 */
  conversationId: string | null
  content: string
  reasoning: string
  sources: SourceItem[]
  /** 阶段提示文案，如「正在检索知识库…」「正在思考…」 */
  stage: string
  error: string
}

const EMPTY_STREAM: StreamState = {
  conversationId: null,
  content: '',
  reasoning: '',
  sources: [],
  stage: '',
  error: '',
}

interface AppState {
  // ---- 会话 ----
  activeId: string | null
  setActiveId: (id: string | null) => void

  // ---- 侧边栏 ----
  sidebarOpen: boolean
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  toggleSidebarCollapsed: () => void

  // ---- 主题 ----
  theme: Theme
  setTheme: (t: Theme) => void
  toggleTheme: () => void

  // ---- 输入草稿（新建对话时把首页推荐问题带进输入框） ----
  draft: string
  setDraft: (v: string) => void

  // ---- 流式生成 ----
  stream: StreamState
  startStream: (conversationId: string) => void
  appendContent: (delta: string) => void
  appendReasoning: (delta: string) => void
  setStreamSources: (sources: SourceItem[]) => void
  setStreamStage: (stage: string) => void
  setStreamError: (error: string) => void
  endStream: () => void
  isGenerating: (conversationId?: string | null) => boolean
}

/** 首次进入时跟随系统主题（也用于拒绝持久化值时兜底）。 */
function systemTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      activeId: null,
      setActiveId: (id) => set({ activeId: id }),

      sidebarOpen: false, // 移动端抽屉默认关闭；桌面端由布局按断点忽略此值
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebarCollapsed: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

      theme: systemTheme(),
      setTheme: (t) => set({ theme: t }),
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),

      draft: '',
      setDraft: (v) => set({ draft: v }),

      stream: { ...EMPTY_STREAM },
      startStream: (conversationId) =>
        set({ stream: { ...EMPTY_STREAM, conversationId, stage: '正在检索知识库…' } }),
      appendContent: (delta) =>
        set((s) =>
          s.stream.conversationId
            ? { stream: { ...s.stream, content: s.stream.content + delta, stage: '' } }
            : {},
        ),
      appendReasoning: (delta) =>
        set((s) =>
          s.stream.conversationId
            ? { stream: { ...s.stream, reasoning: s.stream.reasoning + delta } }
            : {},
        ),
      setStreamSources: (sources) =>
        set((s) =>
          s.stream.conversationId ? { stream: { ...s.stream, sources } } : {},
        ),
      setStreamStage: (stage) =>
        set((s) => (s.stream.conversationId ? { stream: { ...s.stream, stage } } : {})),
      setStreamError: (error) =>
        set((s) => (s.stream.conversationId ? { stream: { ...s.stream, error, stage: '' } } : {})),
      endStream: () => set({ stream: { ...EMPTY_STREAM } }),
      isGenerating: (conversationId) => {
        const id = get().stream.conversationId
        if (!id) return false
        return conversationId === undefined || conversationId === null || conversationId === id
      },
    }),
    {
      name: 'rag-ui',
      // 只持久化用户偏好；会话/消息属于服务端状态，不进 localStorage
      partialize: (s) => ({
        theme: s.theme,
        sidebarCollapsed: s.sidebarCollapsed,
      }),
    },
  ),
)

/** 把主题写到 <html class="dark">，Tailwind 的 darkMode: 'class' 依赖它。 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement
  root.classList.toggle('dark', theme === 'dark')
}
