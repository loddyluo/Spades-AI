# ===========================================================================
# Spades 对战 —— 公网部署单镜像
#
# 一个容器内同时运行：
#   - Caddy        托管前端静态文件 + 反代 /api + 自动 HTTPS
#   - N 个 Python   无状态 AI 后端进程（gui/backend.py）
#
# 构建：  docker build -t spades .
# 运行：  docker run -p 8080:80 -e SITE_ADDRESS=:80 spades   # 本地试跑
# 详见 DEPLOY.md。
# ===========================================================================

# ---- Stage 1：构建前端（React/Vite → 静态文件） ----
FROM node:20-slim AS frontend
WORKDIR /build/gui
# 先拷依赖清单以利用 Docker 层缓存
COPY gui/package.json gui/package-lock.json* ./
RUN npm install
# 再拷前端源码并构建
COPY gui/index.html gui/vite.config.js ./
COPY gui/src ./src
RUN npm run build
# 产物在 /build/gui/dist

# ---- Stage 2：运行镜像（Python + Caddy + native solver） ----
FROM python:3.11-slim AS runtime

# 系统依赖：
#   - libgomp1   native double-dummy solver(.so) 的 OpenMP 运行时（关键！缺它 .so 加载失败）
#   - curl/gpg等 安装 Caddy 官方源所需
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        debian-keyring debian-archive-keyring apt-transport-https gnupg \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends caddy \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖（先装以利用缓存层）
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# 拷入应用代码（.dockerignore 已剔除大体积无关资产）
COPY . /app

# 前端构建产物 → Caddy 静态根目录
COPY --from=frontend /build/gui/dist /srv

# 入口脚本
RUN chmod +x /app/entrypoint.sh

# 默认值（可在 docker run -e 覆盖）
ENV SITE_ADDRESS=:80 \
    PYTHONUNBUFFERED=1

EXPOSE 80 443

ENTRYPOINT ["/app/entrypoint.sh"]
