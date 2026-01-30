# Spades PvE 对弈使用指南（可视化）

本指南介绍如何使用可视化环境与 Spades AI 对弈。

## 1. 启动后端服务

首次使用请先安装后端依赖：

```bash
cd rlcard-showdown
pip3 install -r requirements.txt
```

### 1.1 Leaderboard（可选）

如果只进行 PvE 对弈，可跳过。

```bash
cd rlcard-showdown/server
python3 manage.py runserver
```

默认地址：
- Leaderboard 后端：`http://127.0.0.1:8000/`

### 1.2 Spades PvE 服务（必需）

```bash
cd rlcard-showdown/pve_server
python3 run_spades.py
```

默认地址：
- Spades PvE 后端：`http://127.0.0.1:5001/`

> 若需修改端口，请同步更新 [src/utils/config.js](../src/utils/config.js) 里的 `spadesDemoUrl`。

---

## 2. 启动前端

```bash
cd rlcard-showdown
npm install
export NODE_OPTIONS=--openssl-legacy-provider
npm start
```

如果你使用的是新版 Node（如 v24+），需要加上 `NODE_OPTIONS=--openssl-legacy-provider`，否则会出现 `ERR_OSSL_EVP_UNSUPPORTED`。

默认地址：
- 前端：`http://127.0.0.1:3000/`

---

## 3. 打开 Spades PvE 页面

浏览器访问：

```
http://127.0.0.1:3000/pve/spades
```

进入页面后会自动调用 `reset` 创建对局。

---

## 4. 操作说明

### 4.1 叫牌阶段（Bidding）

- 可点击按钮进行叫牌：
  - Pass
  - Blind Nil
  - Nil
  - Bid 1-13

按钮是否可点击由后端返回的 `legal_actions` 控制。

### 4.2 出牌阶段（Play）

- 点击手牌中高亮的可出牌牌面完成出牌。
- 不可出牌会自动由 AI 处理轮次。

### 4.3 结算

- 对局结束后，会显示队伍分数。
- 点击 Restart 可重新开始。

---

## 5. 常见问题

### 5.1 页面没有反应

请确认以下服务已启动：
- Spades PvE：`http://127.0.0.1:5001/`
- 前端：`http://127.0.0.1:3000/`

### 5.2 端口冲突

可修改端口：
- 后端：修改 [pve_server/run_spades.py](../pve_server/run_spades.py)
- 前端配置：修改 [src/utils/config.js](../src/utils/config.js)

---

## 6. API 说明（参考）

### `POST /reset`

```json
{
  "game": "spades",
  "seed": 42,
  "human_player": 0
}
```

### `POST /step`

```json
{
  "game_id": "spades-xxxx",
  "action": 12
}
```

### `GET /state?game_id=...`

用于刷新当前状态或断线重连。
