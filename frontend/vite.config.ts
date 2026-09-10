import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// 开发时把 /api 代理到后端，前端代码里统一用相对路径 /api/xxx，
// 这样开发态同源（无 CORS 烦恼）、生产态交给 Nginx 同源反代，两边都不用改代码。
const BACKEND = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        // SSE 必须关闭代理层缓冲，否则流式输出会被攒成一坨再吐出
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (String(proxyRes.headers['content-type']).includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        // 把体积最大的高亮语言包拆出去，首屏只加载框架与 UI
        manualChunks: {
          react: ['react', 'react-dom'],
          markdown: ['react-markdown', 'remark-gfm'],
          highlight: ['react-syntax-highlighter'],
        },
      },
    },
  },
})
