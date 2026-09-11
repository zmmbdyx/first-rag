/** 三栏式布局：左侧边栏（可折叠/移动端抽屉）+ 中央对话区 + 底部输入区。 */

import { useCallback, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ChatArea } from '@/components/ChatArea'
import { InputBox } from '@/components/InputBox'
import { Sidebar } from '@/components/Sidebar'
import { IconAlert, IconMenu, Logo } from '@/components/Icons'
import { queryKeys, sendFeedback, useChat, useConversations, useHealth, useMessages } from '@/hooks/useChat'
import { applyTheme, useAppStore } from '@/store/useAppStore'

export default function App() {
  const qc = useQueryClient()
  const { send, stop, regenerate } = useChat()
  const { data: conversations = [] } = useConversations()
  const { data: health } = useHealth()

  const activeId = useAppStore((s) => s.activeId)
  const setActiveId = useAppStore((s) => s.setActiveId)
  const sidebarCollapsed = useAppStore((s) => s.sidebarCollapsed)
  const toggleSidebarCollapsed = useAppStore((s) => s.toggleSidebarCollapsed)
  const sidebarOpen = useAppStore((s) => s.sidebarOpen)
  const setSidebarOpen = useAppStore((s) => s.setSidebarOpen)
  const theme = useAppStore((s) => s.theme)

  const { data: messages = [], isLoading: messagesLoading } = useMessages(activeId)
  const stream = useAppStore((s) => s.stream)

  // 模型选择：默认取后端清单第一项。
  // 放在 store 里而不是组件局部 state —— useChat 发送时要读取它，
  // 通过 props 传递曾导致漏传（下拉框点了不生效）。
  const models = health?.models ?? []
  const model = useAppStore((s) => s.model)
  const setModel = useAppStore((s) => s.setModel)
  useEffect(() => {
    if (!model && models.length > 0) setModel(models[0])
  }, [models, model, setModel])

  // 主题落到 <html>（Tailwind darkMode: 'class'）
  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  // 当前会话标题（顶栏展示）
  const activeTitle =
    conversations.find((c) => c.conversation_id === activeId)?.title ?? '新对话'

  const generating = Boolean(stream.conversationId) && stream.conversationId === activeId

  const onNewConversation = useCallback(() => {
    // 不预先建会话：等第一条消息发出时再创建，避免侧边栏堆一堆空会话
    setActiveId(null)
    qc.removeQueries({ queryKey: queryKeys.messages('') })
    setSidebarOpen(false)
  }, [qc, setActiveId, setSidebarOpen])

  // 回答反馈回流：👍/👎 直接落库，成为可用的在线质量信号
  const onVote = useCallback(
    (vote: 'up' | 'down', ctx: { requestId: string; conversationId: string }) => {
      void sendFeedback({ vote, requestId: ctx.requestId, conversationId: ctx.conversationId })
    },
    [],
  )

  const onSuggestion = useCallback(
    (q: string) => {
      void send(q)
    },
    [send],
  )

  const onSend = useCallback(
    (text: string) => {
      void send(text)
    },
    [send],
  )

  const kbReady = health?.retriever_ready ?? false
  const kbHint =
    health && !kbReady
      ? '知识库尚未就绪：请先在左侧上传文档（后端会自动解析并向量化入库）'
      : ''

  return (
    <div className="flex h-full w-full overflow-hidden bg-canvas">
      {/* 移动端抽屉遮罩 */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 backdrop-blur-[1px] md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* 侧边栏：桌面端常驻（可折叠成窄栏），移动端为抽屉 */}
      <div
        className={`fixed inset-y-0 left-0 z-40 transition-transform duration-300 ease-smooth md:static md:z-auto md:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <Sidebar
          collapsed={sidebarCollapsed}
          onClose={() => setSidebarOpen(false)}
          onNewConversation={onNewConversation}
        />
      </div>

      {/* 主区域 */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* 顶栏 */}
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-4">
          <button
            type="button"
            onClick={() => {
              // 桌面端：折叠/展开；移动端：打开抽屉
              if (window.matchMedia('(min-width: 768px)').matches) toggleSidebarCollapsed()
              else setSidebarOpen(true)
            }}
            className="icon-btn h-9 w-9"
            title="切换侧边栏"
            aria-label="切换侧边栏"
          >
            <IconMenu size={19} />
          </button>

          <div className="flex min-w-0 flex-1 items-center gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{activeTitle}</span>
          </div>

          {health && (
            <span
              className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[11.5px] sm:inline-flex ${
                kbReady
                  ? 'bg-success/10 text-success'
                  : 'bg-danger/10 text-danger'
              }`}
              title={
                kbReady
                  ? `向量库 ${health.chunks ?? 0} 个切片 · ${health.collection}`
                  : '知识库未就绪'
              }
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${kbReady ? 'bg-success' : 'bg-danger'}`}
              />
              {kbReady ? `${health.chunks ?? 0} 切片` : '未就绪'}
            </span>
          )}
        </header>

        {/* 生成失败/降级提示条 */}
        {health && !kbReady && (
          <div className="flex items-center gap-2 border-b border-line bg-danger/5 px-4 py-2 text-[12.5px] text-danger">
            <IconAlert size={14} className="shrink-0" />
            <span>{kbHint}</span>
          </div>
        )}

        {/* 对话区（messages 为空时 ChatArea 自行渲染欢迎空状态） */}
        <ChatArea
          messages={messages}
          loading={messagesLoading}
          onSuggestion={onSuggestion}
          onRegenerate={regenerate}
          onVote={onVote}
        />

        {/* 输入区 */}
        <InputBox
          models={models}
          model={model}
          onModelChange={setModel}
          onSend={onSend}
          onStop={stop}
          generating={generating}
          disabled={Boolean(health) && !kbReady}
          disabledHint={kbHint}
        />
      </div>

      {/* 移动端右下角品牌水印（仅在无侧边栏时可见的设计细节） */}
      <span className="pointer-events-none fixed bottom-3 right-4 hidden select-none items-center gap-1.5 text-[11px] text-ink-muted/60 lg:flex">
        <Logo size={14} />
        RAG 知识库
      </span>
    </div>
  )
}
