# 把 Spades 对战部署到互联网 —— 照抄手册

本手册面向**不熟悉运维**的读者。整个对战已经打包成**一个 Docker 镜像**，
你只要照着复制命令即可。无需改任何代码。

> 架构：一个容器内同时跑 Caddy（网页服务器 + 自动 HTTPS）和多个 Python AI 进程。
> 后端无状态，所以"多人同时玩"靠多开进程解决，对你完全透明。
> 镜像默认使用已选定的 `bid_residual_100k.pt` 负责实际叫牌；旧
> `bid_nsfp.pt` 仅保留为残局 belief bidder 和运行时安全回退。

---

## 准备：安装 Docker

- **Mac（本地试跑）**：装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。
- **云服务器（Ubuntu）**：
  ```bash
  curl -fsSL https://get.docker.com | sh
  ```

---

## A. 先在自己电脑上试跑（强烈建议先做这一步）

在仓库根目录（含 `Dockerfile` 的目录）执行：

```bash
# 1) 构建镜像（第一次较慢，要下载 torch 等，约 5-15 分钟）
docker build -t spades .

# 2) 运行（本地只用 HTTP，映射到 8080 端口）
docker run --rm -p 8080:80 -e SITE_ADDRESS=:80 spades
```

然后：
- 浏览器打开 **http://localhost:8080** → 应能发牌、出牌、AI 应手。
- 健康检查：另开一个终端跑 `curl http://localhost:8080/api/health`。
  返回值中的 `acting_bidder.name` 应为 `residual_q_100k`，且
  `model_id` 应以 `72b9b2fd95da` 开头。

按 `Ctrl+C` 停止。确认这步没问题，再上云。

---

## B. 部署到云服务器（公网访问）

**建议配置**：2~4 核 / 4GB 内存的 Linux VPS（阿里云、腾讯云、Hetzner 等均可）。
torch + solver 每个进程约占 300~600MB 内存。

### B1. 把代码弄到服务器上
```bash
# 在服务器上
git clone <你的仓库地址> spades-ai
cd spades-ai
```

### B2.（推荐）准备一个域名 → 自动 HTTPS
把你的域名（如 `spades.example.com`）的 **A 记录**指向服务器公网 IP。
Caddy 会自动申请并续期 HTTPS 证书，你什么都不用配。

```bash
docker build -t spades .

docker run -d --name spades \
  --restart=always \
  -p 80:80 -p 443:443 \
  -e SITE_ADDRESS=spades.example.com \
  -e WORKERS=4 \
  spades
```

打开 **https://spades.example.com** —— 绿锁 HTTPS 就绪。

> `-d` 后台运行；`--restart=always` 开机/崩溃自动重启；`WORKERS=4` 起 4 个 AI 进程。
> 确保云服务商的**安全组/防火墙放行 80 和 443 端口**。

### B3.（没有域名时的降级方案）只用 IP + HTTP
```bash
docker run -d --name spades --restart=always \
  -p 80:80 -e SITE_ADDRESS=:80 -e WORKERS=4 spades
```
打开 **http://你的服务器IP**。注意：这是明文 HTTP，仅适合临时验证。

---

## C. 加一道访问保护（公网必做）

后端没有任何鉴权，每个请求都消耗 CPU。公开到互联网建议加一个**账号密码**，
防止被陌生人/爬虫刷爆服务器。

```bash
# 1) 生成密码哈希（把 你的密码 换成实际密码）
docker run --rm spades caddy hash-password --plaintext '你的密码'
# 输出形如：$2a$14$xxxxxxxxxxxxxxxxxxxxxx
```

```bash
# 2) 带账号密码重新启动（用上一步的哈希）
docker rm -f spades   # 先删掉旧容器
docker run -d --name spades --restart=always \
  -p 80:80 -p 443:443 \
  -e SITE_ADDRESS=spades.example.com \
  -e WORKERS=4 \
  -e BASIC_AUTH_USER=player \
  -e BASIC_AUTH_HASH='$2a$14$xxxxxxxxxxxxxxxxxxxxxx' \
  spades
```

之后任何人首次打开网页都需输入账号 `player` + 你设的密码。
> 注意：哈希里有 `$` 符号，**务必用单引号**包裹，否则 shell 会吃掉它。

---

## 常用运维命令

```bash
docker logs -f spades          # 看实时日志（排查问题第一步）
docker restart spades          # 重启
docker rm -f spades            # 停止并删除容器
docker stats spades            # 看 CPU / 内存占用
```

更新代码后重新部署：
```bash
git pull
docker build -t spades .
docker rm -f spades
docker run -d --name spades --restart=always -p 80:80 -p 443:443 \
  -e SITE_ADDRESS=spades.example.com -e WORKERS=4 spades
```

---

## 调参与排错

| 现象 | 处理 |
|------|------|
| AI 应手慢、CPU 跑满 | 增大 `WORKERS`（上限=CPU 核数，配置封顶 8），或升级服务器核数 |
| 内存不足被杀 | 减小 `WORKERS`，或加大服务器内存（每进程约 0.3~0.6GB） |
| 日志报 `.so` 加载失败 | 极少见；镜像已装 `libgomp1`。确认是在 x86_64 Linux 上构建/运行 |
| HTTPS 证书申请失败 | 检查域名 A 记录是否已指向本机 IP、80/443 是否放行 |
| 打不开网页 | `docker logs spades` 看 Caddy 是否启动；检查防火墙/安全组 |

---

## 环境变量速查

| 变量 | 说明 | 默认 |
|------|------|------|
| `SITE_ADDRESS` | 域名→自动 HTTPS；`:80`→仅 HTTP | `:80` |
| `WORKERS` | 后端 AI 进程数（1~8） | CPU 核数 |
| `BASIC_AUTH_USER` | Basic 登录用户名（配合 HASH 启用保护） | 不设=无保护 |
| `BASIC_AUTH_HASH` | `caddy hash-password` 生成的哈希 | — |
