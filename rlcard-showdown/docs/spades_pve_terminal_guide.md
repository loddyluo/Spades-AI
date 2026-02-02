# Spades Terminal PvE 使用指南

本指南介绍如何在终端中与 Spades AI 对弈（功能与可视化 PvE 对齐）。

---

## 1. 启动 PvE 后端

```bash
cd rlcard-showdown
pip3 install -r requirements.txt
cd pve_server
python3 run_spades.py
```

默认地址：
- `http://127.0.0.1:5001/`

> 若修改端口，请在 CLI 启动时用 `--server` 指定。

---

## 2. 启动终端 PvE 客户端

```bash
cd rlcard-showdown/pve_server
python3 cli_spades.py --server http://127.0.0.1:5001 \
	--ai-checkpoint "../../Spades-AI/experiments/spades_selfplay_dqn/checkpoint_dqn.pt"
```

### 可选参数

- `--human 0-3`：选择人类玩家位置（默认 0）
- `--seed 42`：固定随机种子
- `--auto 5`：自动走 N 步（用于冒烟测试）
- `--auto-exit`：自动步数完成后退出
- `--delay 0.2`：自动步之间的间隔
- `--ai-checkpoint PATH`：指定 DQN checkpoint（AI 使用）
- `--opponent-checkpoint PATH`：指定对手团队 DQN checkpoint（默认与 AI 相同）

示例（冒烟测试）：

```bash
python3 cli_spades.py --server http://127.0.0.1:5001 --auto 5 --auto-exit
```

---

## 3. 交互说明

### 叫牌阶段

- 屏幕会列出合法叫牌选项
- 输入选项编号或动作名（`pass` / `blind_nil` / `nil` / `bid_1`~`bid_13`）

### 出牌阶段

- 会显示合法可出牌牌面
- 直接输入牌面（如 `SA`）即可出牌

---

## 4. 结束与重开

- 对局结束会显示队伍分数
- 输入 `y` 可重新开始

---

## 5. 常见问题

### 5.1 无法连接服务器

确保 PvE 后端已启动：

```bash
cd rlcard-showdown/pve_server
python3 run_spades.py
```

如果你改了端口，请更新 CLI 的 `--server` 参数。
