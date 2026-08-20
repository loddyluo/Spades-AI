# 当前 Spades AI vs DeepSeek V4 Flash 队式赛

`evaluate/deepseek_team_match.py` 提供一个独立测试接口：当前生产 AI 作为一对搭档，对战由 DeepSeek V4 Flash 控制的另一对搭档。

## 对局设置

- 当前 AI：`RuleExactFirst4NilPlayer` 出牌链路、`residual_q_100k` 部署叫牌器、`configs/8.yaml`、默认 `exact_threshold=36`。
- DeepSeek：默认座位 1/3；每次决策只接收自己的手牌和公开叫牌、出牌、墩数，不接收其他三家的手牌。
- 当前 AI：默认座位 0/2。
- 叫牌：支持 Nil 和 1-13；与现有 GUI 一致，不启用 Blind Nil 的两阶段 `pass` 流程。
- 非法响应：DeepSeek 最多按 `--protocol-attempts` 重答；仍然非法就立即中止牌局，不自动代打。
- 当前 AI fallback：立即中止牌局并显示原因，不替换动作。

`--swap-sides` 会让每副牌使用完全相同的种子换边再赛一次，从而降低 0/2 与 1/3 座位差异。
已有第一桌记录时，可用 `--current-ai-seats 1,3` 只补跑当前 AI 换到
1/3 的第二桌，避免重复调用第一桌。

## DeepSeek 透传协议

实现采用随仓库 PDF 的透传格式：

- `POST http://trpc-gpt-eval.production.polaris:8080/v1/chat/completions`
- 请求体为 OpenAI Chat Completions JSON，模型为 `deepseek-v4-flash`
- `Authorization` 为 `Bearer APP_ID:APP_KEY?provider=deepseek&model=deepseek-v4-flash&timeout=60`
- 网关超时在 Authorization 查询串中；HTTP 客户端超时由 `--request-timeout` 单独控制
- 默认开启 DeepSeek 思考模式，请求中不发送 `thinking.disabled`
- 默认 `max_tokens=328000`（328k）；需要关闭思考时显式传 `--no-thinking`
- 发送 `x-should-retry: false`，由本测试接口统一执行有界重试

## 凭据

接口只读取以下环境变量，不接受包含凭据的命令行参数，也不会把它们写入日志或赛果：

```bash
export DEEPSEEK_APP_ID
export DEEPSEEK_APP_KEY
```

你稍后提供 APP_ID 和 APP_KEY 后，再安全设置这两个变量即可。没有凭据时，真实接口会在加载当前 AI 模型前直接报错。

## 运行

单副牌：

```bash
python -m evaluate.deepseek_team_match --games 1 --seed 20260803
```

同牌换边的 10 副评测：

```bash
python -m evaluate.deepseek_team_match \
  --games 10 \
  --seed 20260803 \
  --swap-sides \
  --num-workers 7
```

指定输出文件：

```bash
python -m evaluate.deepseek_team_match \
  --games 1 \
  --seed 20260803 \
  --output output/deepseek_team_match.json
```

默认输出到 `output/deepseek_team_match_<时间>_seed<种子>.json`。记录包含发牌、座位、叫牌、每一步合法动作、实际动作、每墩赢家、最终分数、AI 决策模式和 DeepSeek token 用量，但不包含凭据或 DeepSeek 的思维链。

已有同名输出文件时默认拒绝覆盖；明确需要覆盖时传 `--overwrite`。

## GUI 复盘导入

GUI 模式菜单提供“导入复盘”。支持以下 JSON：

- GUI 自己导出的 `spades-ai-replay`；
- 本脚本生成的 `spades-ai-deepseek-team-match` 单局或多局结果；
- 带 `replay_records` 的汇总文件，以及 `spades-ai-replay-bundle` 多局文件。

选择文件后，可以从下拉框选择牌局和 0-3 任一复盘视角。导入器会重新验证标准 52 张牌、每家初始手牌、逐步持牌与跟牌规则、黑桃是否已破、出牌轮转、每墩赢家、墩数和计分；任何不一致都会在菜单中明确报错，不会静默修正或 fallback。

## 双桌队式汇总

已分别生成 A 桌（当前 AI 坐 0/2）和 B 桌（当前 AI 坐 1/3）时，
`evaluate/deepseek_duplicate_report.py` 会按 seed 配对，确认两桌四手初始牌完全相同，
并从叫牌和墩数独立重算分数。每副牌的队式分差为两桌“当前 AI 视角分差”
之和。

```bash
python -m evaluate.deepseek_duplicate_report \
  --table-a-dir output/table_a \
  --table-b-dir output/table_b \
  --seed 20260804 \
  --games 8 \
  --output output/duplicate_summary.json \
  --replay-bundle output/gui_replay_bundle.json
```

汇总文件和复盘包都嵌入 A/B 两桌的可移植复盘记录，可直接从 GUI
的“导入复盘”打开。

## Python 接口

需要从其他测试程序调用时，使用：

```python
from evaluate.deepseek_team_match import (
    DeepSeekPassThroughClient,
    DeepSeekPassThroughConfig,
    load_current_ai_components,
    run_team_match,
)

client = DeepSeekPassThroughClient(DeepSeekPassThroughConfig.from_env())
components = load_current_ai_components(num_workers=7)
report = run_team_match(
    client=client,
    components=components,
    games=1,
    seed=20260803,
    swap_sides=True,
)
```

## 无真实 API 的验证

单元测试使用注入式 mock 传输层，不会访问网络：

```bash
python -m pytest -q tests/test_deepseek_team_match.py
```

测试会核对透传 Header/Body、凭据脱敏、隐藏手牌隔离、非法动作 fail-fast、完整 52 步 mock 对局和原子赛果写入。
