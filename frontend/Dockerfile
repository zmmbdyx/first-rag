# 企业知识库 RAG 问答系统 —— 前端生产镜像
#
# 构建： docker build -t rag-frontend ./frontend
# 运行： docker run -d -p 8080:80 rag-frontend
#
# 两阶段构建：
#   1) node 阶段跑 vite build 产出静态文件；
#   2) nginx 阶段只保留 dist/，镜像里不含 node_modules 与源码。

# ---------- 1) 构建 ----------
FROM node:22-alpine AS build

WORKDIR /app

# 先拷依赖清单，利用层缓存：只要依赖没变就不会重装
COPY package.json package-lock.json ./
RUN npm ci

# 再拷源码
COPY . .

# 后端地址在运行时由 nginx 反代决定，构建期无需知道，
# 因此这里保持默认的相对路径 /api。
RUN npm run build

# ---------- 2) 运行 ----------
FROM nginx:1.27-alpine AS runtime

# 自定义配置：SPA 路由回退 + /api 反代 + 关闭 SSE 缓冲
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
