# 前端（React 18 + TypeScript + Vite + Tailwind CSS）

企业知识库 RAG 问答系统的前端，界面风格对齐 **DeepSeek 官方网页版**：
三栏布局、低饱和度冰淇淋蓝主色（`#4D6BFE`）、大圆角、宽松留白、极简白底（可切深色）。

## 快速启动

```bash
npm install
npm run dev      # http://127.0.0.1:5173
```

开发服务器已把 `/api` 反向代理到后端（见 `vite.config.ts`），**开发态同源、无需处理 CORS**。
后端默认地址 `http://127.0.0.1:8000`，可用 `VITE_BACKEND_URL` 覆盖。

> 启动前端前请先启动后端：`uvicorn backend.main:app --reload --port 8000`

```bash
npm run build      # 生产构建 → dist/
npm run preview    # 预览构建产物
npm run typecheck  # 只做类型检查
```

## 技术栈

| 依赖 | 用途 |
|---|---|
| React 18 + TypeScript | UI 与类型安全 |
| Vite 6 | 构建与开发服务器（含 `/api` 代理） |
| Tailwind CSS 3 | 原子化样式；设计令牌走 CSS 变量 |
| react-markdown + remark-gfm | Markdown 渲染（标题/列表/表格/引用/任务列表） |
| react-syntax-highlighter | 代码块语法高亮（随主题切换 oneLight / oneDark） |
| zustand | 客户端状态（当前会话、侧边栏、主题、输入草稿、流式缓冲） |
| @tanstack/react-query | 服务端状态（会话列表、消息历史）的缓存与失效 |

## 目录结构

```
src/
├── components/
│   ├── Sidebar.tsx           # 侧边栏：新对话 / 搜索 / 分组历史 / 悬停改名删除 / 主题切换
│   ├── ChatArea.tsx          # 对话区：空状态欢迎页 + 推荐问题 + 消息列表 + 自动滚动
│   ├── MessageBubble.tsx     # 消息气泡：Markdown / 思考过程 / 操作栏（复制·重生成·赞踩）
│   ├── MarkdownRenderer.tsx  # GFM 渲染 + 代码高亮 + 一键复制
│   ├── SourceCards.tsx       # 引用来源卡片（可折叠，显示原文与相似度）
│   ├── InputBox.tsx          # 输入区：多行输入 / 拖拽上传 / 发送·停止 / 模型选择
│   └── Icons.tsx             # 内联 SVG 图标集（不引第三方图标库）
├── hooks/
│   ├── useChat.ts            # 聊天编排：SSE 流 → store 缓冲 → react-query 失效
│   └── useCopy.ts            # 剪贴板（含 execCommand 降级）
├── lib/
│   ├── api.ts                # REST 客户端
│   ├── sse.ts                # SSE 流解析（fetch + ReadableStream）
│   └── date.ts               # 今天/昨天/近7天/更早 分组、相对时间
├── store/useAppStore.ts      # zustand（持久化主题与侧边栏折叠态）
├── types/index.ts            # 与后端契约一一对应的类型
├── styles/index.css          # 设计令牌（CSS 变量）+ Markdown 排版
├── App.tsx                   # 三栏布局与路由级编排
└── main.tsx                  # 入口：主题预应用 + react-query Provider
```

## 状态管理边界

刻意把两类状态分开，避免"同一份数据两处维护"：

- **zustand**：客户端状态 —— 当前会话 ID、侧边栏开合、主题、输入草稿、流式缓冲。
  只用 `partialize` 持久化主题与侧边栏折叠态。
- **react-query**：服务端状态 —— 会话列表与消息历史。
  流式结束后 `invalidateQueries` 用服务端落库结果替换临时态。

## 主题实现

设计令牌集中在 `src/styles/index.css` 的 CSS 变量里，`tailwind.config.js` 把语义色
（`bg-canvas` / `text-ink` / `border-line` / `bg-bubble` …）映射到这些变量，因此**深/浅色切换
只需给 `<html>` 加减一个 `dark` 类**，全站自动生效，无需在每个组件里写 `dark:` 变体。

`index.html` 内联了一小段脚本，在首帧之前读取 localStorage 里的主题并挂类，消除深色模式下的白屏闪烁。

## 交互细节

- **Enter 发送 / Shift+Enter 换行**；输入框高度随内容自增，超过 200px 转为内部滚动；
- 输入法组合输入（中文拼音）中的回车**不会**误触发发送（判断 `isComposing`）；
- **拖拽上传**：把文件拖到输入区即上传；上传成功后提示入库切片数；
- **停止生成**：生成中发送按钮变成停止键，已生成内容保留（后端也会保存部分回答）；
- **重新生成**：调用后端的 `regenerate=true`，由后端复用最近一条提问并**覆盖**旧回答。
  前端刻意不重发消息，否则历史里会出现两条一模一样的提问气泡（这条有浏览器级回归测试）；
- **引用来源**：默认折叠成一行摘要，点击展开原文；显示文档名、章节路径、页码、
  相似度、重排分数与命中路径（向量 / 关键词）；
- **响应式**：< 768px 时侧边栏收起为抽屉，点击汉堡菜单滑出。

## 鉴权（可选）

后端 `API_KEYS` 留空时不鉴权，前端无需任何配置。若后端启用了鉴权，前端可用构建期变量注入密钥：

```bash
VITE_API_KEY=your-service-key npm run build
```

⚠️ **`VITE_` 前缀的变量会被打进前端产物，等于公开可见**。因此它只适合"对前端本身做访问控制"
的场景；若希望密钥完全不出现在浏览器里，请在 Nginx 反代时注入 `Authorization` 头
（`frontend/nginx.conf` 里加一行 `proxy_set_header`），而不要使用该变量。
另：大模型的真实密钥（后端的 `API_KEY`）**绝不能**放在这里。

## 接口契约

前端只依赖以下端点（完整契约见根目录 README 第 3.3 节与后端 `/docs`）：

```
POST   /api/chat                          SSE 流式问答
POST   /api/conversations                 新建会话
GET    /api/conversations                 会话列表
GET    /api/conversations/{id}/messages   某会话消息
DELETE /api/conversations/{id}            删除会话
PUT    /api/conversations/{id}/title      重命名会话
POST   /api/upload                        上传文档
GET    /api/health                        服务与知识库状态
```

## 验收

```bash
python ../scripts/verify_frontend.py    # Playwright 真实交互断言（27 项）
python ../scripts/capture_frontend.py   # 生成界面截图到 screenshots/frontend/
```
