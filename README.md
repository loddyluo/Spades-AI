# Spades-AI

本仓库提供 Spades（黑桃）环境、训练好的 DQN 模型，以及终端 PvE 对弈工具。以下步骤面向新手，按顺序执行即可与高水平 AI 对弈。

---

## 0. 你将获得什么

- 训练好的模型：
  - [experiments/spades_selfplay_dqn/checkpoint_dqn.pt](experiments/spades_selfplay_dqn/checkpoint_dqn.pt)
  - [experiments/spades_selfplay_dqn/checkpoint_opponent.pt](experiments/spades_selfplay_dqn/checkpoint_opponent.pt)
- 终端 PvE：
  - [rlcard-showdown/pve_server/run_spades.py](rlcard-showdown/pve_server/run_spades.py)
  - [rlcard-showdown/pve_server/cli_spades.py](rlcard-showdown/pve_server/cli_spades.py)

---

## 1. 创建并激活 Python 虚拟环境

```bash
cd "Spades-AI"
python3 -m venv .venv
source .venv/bin/activate
```

> 看到命令行前面出现 `(.venv)` 即表示激活成功。

---

## 2. 安装依赖

```bash
pip install -r rlcard-showdown/requirements.txt
```

---

## 3. 启动 PvE 后端

```bash
cd rlcard-showdown/pve_server
python3 run_spades.py
```

保持这个终端运行不关闭。默认端口为 `5001`。

---

## 4. 在新终端启动 PvE 客户端（与 AI 对弈）

打开一个**新的终端**，并再次激活虚拟环境：

```bash
cd "Spades-AI"
source .venv/bin/activate
```

运行终端 PvE 客户端（必须指定训练得到的 checkpoint）：

```bash
python3 rlcard-showdown/pve_server/cli_spades.py \
  --server http://127.0.0.1:5001 \
  --ai-checkpoint "experiments/spades_selfplay_dqn/checkpoint_dqn.pt"
```

> 说明：PVE 不再使用随机 AI。三个对手全部使用你提供的 `checkpoint_dqn.pt`。

---

## 5. 终端玩法说明

### 叫牌阶段

- 输入编号或动作名：
  - `pass` / `blind_nil` / `nil` / `bid_1`~`bid_13`

### 出牌阶段

- 直接输入牌面，例如：`SA`（黑桃 A）、`H5`（红桃 5）

### 结算

- 游戏结束会显示：
  - 每位玩家叫牌与赢墩
  - 队伍得分
- 输入 `y` 可重开

---

## 6. 常见问题

### 6.1 Flask/Jinja2 报错

请确保已安装：

```bash
pip install -r rlcard-showdown/requirements.txt
```

### 6.2 无法连接服务器

确认 PvE 后端在运行：

```bash
cd rlcard-showdown/pve_server
python3 run_spades.py
```

---

## 7. 训练脚本（可选）

如果你要重新训练：

- 脚本位置：
  - [rlcard/train_spades_selfplay.py](rlcard/train_spades_selfplay.py)

训练完成后会生成新的 checkpoint 到：

- [experiments/spades_selfplay_dqn](experiments/spades_selfplay_dqn)

---

如需可视化对弈或网页界面，请使用 [rlcard-showdown](rlcard-showdown) 目录下的前端与服务（文档见 [rlcard-showdown/docs](rlcard-showdown/docs)）。
