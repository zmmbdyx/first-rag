import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from '@/App'
import { applyTheme, useAppStore } from '@/store/useAppStore'
import '@/styles/index.css'

// 首帧之前就把主题类挂到 <html>，避免深色模式用户看到一闪而过的白屏
applyTheme(useAppStore.getState().theme)

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5_000,
    },
  },
})

const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('缺少 #root 挂载点')

createRoot(rootEl).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
