#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 容器入口：启动 N 个 Python AI 后端进程 + 前台运行 Caddy。
#
# 环境变量：
#   SITE_ADDRESS    Caddy 站点地址。域名→自动 HTTPS；":80"→本地仅 HTTP。默认 :80
#   WORKERS         后端进程数。默认=CPU 核数，上限 8。
#   BASIC_AUTH_USER 设置后启用 HTTP Basic 账号密码保护（配合 BASIC_AUTH_HASH）
#   BASIC_AUTH_HASH 由 `caddy hash-password` 生成的 bcrypt 哈希
# ---------------------------------------------------------------------------
set -euo pipefail

SITE_ADDRESS="${SITE_ADDRESS:-:80}"

# ---- 决定后端进程数：默认取 CPU 核数，封顶 8 ----
DEFAULT_WORKERS="$(nproc 2>/dev/null || echo 2)"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
if [ "$WORKERS" -gt 8 ]; then WORKERS=8; fi

echo "[entrypoint] SITE_ADDRESS=$SITE_ADDRESS  WORKERS=$WORKERS"

# ---- 启动 N 个无状态后端，端口 8001..800N ----
UPSTREAMS=""
for i in $(seq 1 "$WORKERS"); do
	port=$((8000 + i))
	echo "[entrypoint] starting backend on 127.0.0.1:$port"
	python /app/gui/backend.py --host 127.0.0.1 --port "$port" &
	UPSTREAMS="$UPSTREAMS 127.0.0.1:$port"
done

# ---- 生成 Caddy 配置（上游列表与 WORKERS 精确匹配） ----
# 可选 basic_auth 块：仅当同时提供了用户名与哈希时注入。
AUTH_BLOCK=""
if [ -n "${BASIC_AUTH_USER:-}" ] && [ -n "${BASIC_AUTH_HASH:-}" ]; then
	echo "[entrypoint] HTTP Basic auth ENABLED for user '$BASIC_AUTH_USER'"
	AUTH_BLOCK=$(cat <<EOF
	basic_auth {
		${BASIC_AUTH_USER} ${BASIC_AUTH_HASH}
	}
EOF
)
else
	echo "[entrypoint] HTTP Basic auth DISABLED (set BASIC_AUTH_USER + BASIC_AUTH_HASH to enable)"
fi

cat > /etc/caddy/Caddyfile.run <<EOF
${SITE_ADDRESS} {
${AUTH_BLOCK}

	handle /api/* {
		reverse_proxy${UPSTREAMS} {
			lb_policy round_robin
			health_uri /api/health
			health_interval 10s
			health_timeout 3s
		}
	}

	handle {
		root * /srv
		try_files {path} /index.html
		file_server
	}

	encode gzip
}
EOF

echo "[entrypoint] generated Caddyfile:"
cat /etc/caddy/Caddyfile.run

# ---- 前台运行 Caddy（PID 1）。任一后端崩溃不会拖垮容器；Caddy 退出则容器退出 ----
exec caddy run --config /etc/caddy/Caddyfile.run --adapter caddyfile
