#!/usr/bin/env python3
"""调试 RuleExactFirst4Player：输入牌局状态，输出玩家的动作和详细调试信息。

用法：
  python debug_rule_exact_first4.py <state.txt>       # 文本格式（推荐）
  python debug_rule_exact_first4.py <state.json>      # JSON 格式（兼容旧版）
  python debug_rule_exact_first4.py --example         # 打印示例

文本格式（推荐）:
  position: 0
  hand: SA, SK, SQ, SJ, ST, S9, S8, S7, S6, S5, S4, S3, S2
  bids: bid_4, bid_2, bid_3, bid_2
  plays:
  0 S2
  1 H2
  2 D2
  3 C2
  0 S3
  1 H3
  2 D3
  3 C3
  0 S4
  1 H4
  2 D4

说明：
  - position: 你的座位 (0=南, 1=西, 2=北, 3=东)
  - hand: 你的全部 13 张初始手牌（程序自动扣除已出牌）
  - bids: 四个玩家的叫牌，按座位 0,1,2,3 顺序
  - plays: 所有已出的牌，按时间顺序，每行 "<座位> <牌>"
  - 程序自动将每 4 张分一组作为已完成墩，最后不足 4 张的是当前桌面
  - 对手的未知牌用随机牌填充（exact solver 会通过采样处理不确定性）

选项:
  --exact        强制走 exact solver（threshold=52）
  --threshold N  设置切换阈值（默认 36）
  --fast         用快速参数（减小 IS pool）
  --config YAML  加载指定的超参数 YAML（默认 configs/8.yaml）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trick_taking.card import Card, Suit, Rank
from trick_taking.card import _STANDARD_CARDS as STANDARD_52
from trick_taking.game_state import GameState, Phase, TrickRecord, Bid
from trick_taking.games.spades import SpadesRules
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from strategy.hyperparam_config import HyperparamConfig
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


# ─── 快速调试用的 config ──────────────────────────────────────────────────────
FAST_EXACT_CONFIG = HyperparamConfig(
    num_proposals=200, num_proposals_limit=500, min_pool_size=30,
    bad_action_weight="x", trick_num_threshold=8,
    swap_is_fill=False, multiplier_clip=40.0, multiplier_clip_factor=1.0,
)


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def parse_card(s: str) -> Card:
    """'SA' → Card(SPADES, ACE), 'H2' → Card(HEARTS, TWO), 等等。"""
    return Card.from_str(s.strip())


def card_to_str(c: Card) -> str:
    return f"{c.suit.short}{c.rank.short}"


def _all_52_ids() -> set[int]:
    return {c.card_id for c in STANDARD_52}


# ─── 文本格式解析 ─────────────────────────────────────────────────────────────

def parse_text_state(path: str | Path) -> dict[str, Any]:
    """解析简洁文本格式的牌局状态文件。

    格式:
        position: <0-3>
        hand: <card>, <card>, ...
        bids: <bid>, <bid>, <bid>, <bid>
        plays:
        <seat> <card>
        <seat> <card>
        ...

    支持 # 开头的注释行和空行。
    """
    text = Path(path).read_text(encoding="utf-8")

    position: int | None = None
    hand_cards: list[str] = []
    bids: list[str] = []
    plays: list[tuple[int, str]] = []

    section: str | None = None  # "plays" or None (header)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # 检测 section 切换
        if line == "plays:":
            section = "plays"
            continue

        if section == "plays":
            # 格式: "<seat> <card>"
            parts = line.split()
            if len(parts) >= 2:
                seat = int(parts[0])
                card_str = parts[1].rstrip(",")
                plays.append((seat, card_str))
            continue

        # header 行: key: value
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "position":
                position = int(value)
            elif key == "hand":
                hand_cards = [s.strip() for s in value.split(",") if s.strip()]
            elif key == "bids":
                bids = [s.strip() for s in value.split(",") if s.strip()]

    if position is None:
        raise ValueError("缺少 'position:' 字段")
    if not hand_cards:
        raise ValueError("缺少 'hand:' 字段")
    if len(bids) != 4:
        raise ValueError(f"'bids:' 需要恰好 4 个叫牌，实际 {len(bids)} 个")

    return {
        "position": position,
        "hand": hand_cards,
        "bids": bids,
        "plays": plays,
    }


def build_state_from_text(data: dict[str, Any]) -> GameState:
    """从文本格式数据构造 GameState。

    hand 是当前玩家的全部 13 张初始手牌。对手的牌用：
      已出牌 + 随机未见牌填充
    exact solver 会通过 IS 采样自然地处理不确定性。
    """
    position: int = data["position"]
    hand_strs: list[str] = data["hand"]
    bid_strs: list[str] = data["bids"]
    plays_raw: list[tuple[int, str]] = data["plays"]

    # 玩家的完整初始手牌（13 张）
    my_full_hand = [parse_card(s) for s in hand_strs]

    # 解析所有出牌
    all_plays: list[tuple[int, Card]] = [(p, parse_card(c)) for p, c in plays_raw]

    # 计算每人已出的牌
    played_by: dict[int, list[Card]] = {i: [] for i in range(4)}
    for p, c in all_plays:
        played_by[p].append(c)

    # 收集所有已知牌（我的完整手牌 + 所有人已出的牌）
    known_ids: set[int] = set(c.card_id for c in my_full_hand)
    for p in range(4):
        if p != position:
            for c in played_by[p]:
                known_ids.add(c.card_id)

    # 未见牌池：52 张中除去已知牌
    unseen = [c for c in STANDARD_52 if c.card_id not in known_ids]
    rng = random.Random(42)  # 固定种子保证可复现
    rng.shuffle(unseen)

    # 对手每人还需填充的牌数
    needs: dict[int, int] = {}
    for p in range(4):
        if p != position:
            needs[p] = 13 - len(played_by[p])

    # 构造 4 手初始牌
    hands: list[list[Card]] = []
    pos = 0
    for p in range(4):
        if p == position:
            hands.append(list(my_full_hand))
        else:
            n = needs[p]
            fill = unseen[pos:pos + n]
            pos += n
            hands.append(list(played_by[p]) + fill)

    # 分组出牌为墩
    trick_history, table_cards = _group_plays_into_tricks(all_plays)

    # 构建 GameState
    rule = SpadesRules()
    state = GameState()
    state.init_for_deal(4, [list(h) for h in hands], [], list(STANDARD_52))
    state.phase = Phase.PLAYING

    state.max_bid = list(bid_strs)
    state.bids = []
    for seat, bv in enumerate(bid_strs):
        state.bids.append(Bid(player_id=seat, value=bv, is_pass=(bv == "pass")))

    state.teams = rule.set_team(state)

    for cards, winner, leader in trick_history:
        record = TrickRecord(cards=cards, winner=winner, leader=leader)
        state.trick_history.append(record)

    state.tricks_played = len(trick_history)

    for pid, c in table_cards:
        state.table_cards.append((pid, c))

    state.turn = position
    first_on_table = table_cards[0][0] if table_cards else position
    state.trick_leader = first_on_table

    # 黑桃打破状态
    spades_broken = False
    for trick_cards, _, _ in trick_history:
        for _, c in trick_cards:
            if c.suit == Suit.SPADES:
                spades_broken = True
    for _, c in table_cards:
        if c.suit == Suit.SPADES:
            spades_broken = True
    state.spades_broken = spades_broken
    state.trump_broken = spades_broken

    # played_bitset
    pbitset = 0
    for pid, c in all_plays:
        pbitset |= c.bit
    state.played_bitset = pbitset

    # tricks_won
    tricks_won = [0] * 4
    for _, winner, _ in trick_history:
        tricks_won[winner] += 1
    state.tricks_won = tricks_won

    # hand_bitsets
    for pid in range(4):
        bit = 0
        for c in state.hands[pid]:
            bit |= c.bit
        state.hand_bitsets[pid] = bit

    return state


def _group_plays_into_tricks(
    all_plays: list[tuple[int, Card]],
) -> tuple[list[tuple[list[tuple[int, Card]], int, int]], list[tuple[int, Card]]]:
    """将出牌序列分组为已完成墩和当前桌面。

    Returns:
        trick_history: list of (cards, winner, leader)
        table_cards: 当前不完整墩的牌
    """
    rule = SpadesRules()
    trick_history: list[tuple[list[tuple[int, Card]], int, int]] = []
    table_cards: list[tuple[int, Card]] = []

    # 每 4 张一组
    full_tricks = len(all_plays) // 4
    remainder = len(all_plays) % 4

    for ti in range(full_tricks):
        start = ti * 4
        cards = all_plays[start:start + 4]
        leader = cards[0][0]

        # 用 SpadesRules 判定谁赢
        # 需要临时构造 state
        state = GameState()
        state.num_players = 4
        state.trump_broken = True  # 保守：设为已打破
        state.spades_broken = True
        state.table_cards = list(cards)
        winner = rule.winner_trick(state)

        trick_history.append((cards, winner, leader))

    if remainder > 0:
        table_cards = all_plays[full_tricks * 4:]

    return trick_history, table_cards


# ─── JSON 格式解析（兼容旧版）──────────────────────────────────────────────────

def parse_json_state(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_state_from_json(data: dict[str, Any]) -> GameState:
    """从 JSON 构造 GameState（需要提供所有 4 手的初始 13 张）。"""
    rule = SpadesRules()

    hands_raw: dict[str, list[str]] = data["hands"]
    hands: list[list[Card]] = []
    for seat in range(4):
        hands.append([parse_card(s) for s in hands_raw[str(seat)]])

    state = GameState()
    state.init_for_deal(4, [list(h) for h in hands], [], list(STANDARD_52))
    state.phase = Phase.PLAYING

    state.dealer_seat = data.get("dealer", 0)

    bids_raw: list[str] = data.get("bids", ["bid_1"] * 4)
    state.max_bid = list(bids_raw)
    state.bids = []
    for seat, bv in enumerate(bids_raw):
        state.bids.append(Bid(player_id=seat, value=bv, is_pass=(bv == "pass")))

    state.teams = rule.set_team(state)

    for th_raw in data.get("trick_history", []):
        cards: list[tuple[int, Card]] = [
            (int(pc[0]), parse_card(pc[1])) for pc in th_raw["cards"]
        ]
        winner = int(th_raw["winner"])
        leader = int(th_raw["leader"])
        state.trick_history.append(
            TrickRecord(cards=cards, winner=winner, leader=leader)
        )

    state.tricks_played = data.get("tricks_played", len(state.trick_history))

    for pid_s, cs in data.get("current_table", []):
        state.table_cards.append((int(pid_s), parse_card(cs)))

    state.turn = data.get("turn", 0)
    first = state.table_cards[0][0] if state.table_cards else state.turn
    state.trick_leader = data.get("trick_leader", first)

    spades_broken = data.get("spades_broken", False) or data.get("trump_broken", False)
    if not spades_broken:
        for rec in state.trick_history:
            for _, c in rec.cards:
                if c.suit == Suit.SPADES:
                    spades_broken = True
        for _, c in state.table_cards:
            if c.suit == Suit.SPADES:
                spades_broken = True
    state.spades_broken = spades_broken
    state.trump_broken = spades_broken

    pbitset = 0
    for rec in state.trick_history:
        for _, c in rec.cards:
            pbitset |= c.bit
    for _, c in state.table_cards:
        pbitset |= c.bit
    state.played_bitset = pbitset

    tricks_won = [0] * 4
    for rec in state.trick_history:
        tricks_won[rec.winner] += 1
    state.tricks_won = tricks_won

    for pid in range(4):
        state.hand_bitsets[pid] = sum(1 << c.card_id for c in state.hands[pid])

    return state


# ─── 统一入口 ─────────────────────────────────────────────────────────────────

def load_state(path: str | Path) -> tuple[dict[str, Any], GameState, bool]:
    """自动检测格式 (JSON 或文本)，返回 (raw_data, GameState, is_json)。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()

    if raw.startswith("{"):
        # JSON 格式
        data = parse_json_state(path)
        state = build_state_from_json(data)
        return data, state, True
    else:
        # 文本格式
        data = parse_text_state(path)
        state = build_state_from_text(data)
        return data, state, False


# ─── 回放 & 扣除 ──────────────────────────────────────────────────────────────

def replay_history(state: GameState, player: RuleExactFirst4Player) -> None:
    """回放游戏历史，使玩家内部状态达到当前状态。"""
    position = player.position

    player.start_game(position, list(state.hands[position]), 4)

    for bid_record in state.bids:
        player.bid_placed(bid_record.player_id, bid_record.value)

    player.set_teams(list(state.teams), list(state.max_bid))

    all_played: list[tuple[int, Card]] = []
    for record in state.trick_history:
        for pid, c in record.cards:
            all_played.append((pid, c))
    for pid, c in state.table_cards:
        all_played.append((pid, c))

    for pid, c in all_played:
        player.card_played(pid, c)

    # 从 state.hands 扣减已出牌（对所有格式都需要）


def _deduct_played_cards(state: GameState) -> None:
    played_by: dict[int, set[int]] = {i: set() for i in range(4)}
    for record in state.trick_history:
        for pid, c in record.cards:
            played_by[pid].add(c.card_id)
    for pid, c in state.table_cards:
        played_by[pid].add(c.card_id)

    for pid in range(4):
        state.hands[pid] = [
            c for c in state.hands[pid]
            if c.card_id not in played_by[pid]
        ]
        state.hand_bitsets[pid] = sum(1 << c.card_id for c in state.hands[pid])


# ─── 格式化输出 ────────────────────────────────────────────────────────────────

POS_NAMES = ["South (0)", "West (1)", "North (2)", "East (3)"]

MODE_DESCRIPTIONS = {
    "rule_first4": "规则式出牌 (前 4 墩, 盲眼规则)",
    "exact_is_determinized": "精确求解器 (IS 采样 + Q 值加权)",
    "nil_policy_first4": "Nil 策略网络",
    "no_state_fallback": "无 state (fallback)",
    "no_exact_solver_fallback": "无精确求解器 (fallback)",
    "exact_no_match_fallback": "精确求解器无匹配 (fallback)",
}


def print_state_summary(raw_data: dict[str, Any], state: GameState,
                        player: RuleExactFirst4Player) -> None:
    """打印牌局状态概览。"""
    position = raw_data.get("position", player.position)
    remaining = sum(len(h) for h in state.hands)
    print("=" * 70)
    print("  牌局状态概览")
    print("=" * 70)
    print(f"  当前玩家:     {POS_NAMES[position]}")
    print(f"  当前 turn:    {state.turn}")
    print(f"  叫牌:         {state.max_bid}")
    print(f"  队伍:         {state.teams}  (0&2 vs 1&3)")
    print(f"  已打墩数:     {state.tricks_played}")
    print(f"  黑桃已打破:   {state.spades_broken}")
    print(f"  剩余牌数:     {remaining}")
    print()

    hand_cards = sorted(state.hands[position], key=lambda c: (c.suit.value, -c.rank.value))
    print(f"  自己手牌 ({len(hand_cards)} 张):")
    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in hand_cards:
        by_suit[c.suit].append(c)
    for suit in Suit:
        cards = sorted(by_suit[suit], key=lambda c: -c.rank.value)
        if cards:
            print(f"    {suit.symbol}: {' '.join(card_to_str(c) for c in cards)}")
    print()

    if state.table_cards:
        print("  当前桌面:")
        for pid, c in state.table_cards:
            marker = " <-- 你" if pid == position else ""
            print(f"    seat {pid}: {card_to_str(c)}{marker}")
        print()

    if state.trick_history:
        print(f"  已完成墩 ({len(state.trick_history)} 墩):")
        offset = max(0, len(state.trick_history) - 5)
        for i, record in enumerate(state.trick_history[-5:]):
            cards_str = " | ".join(
                f"P{pid}:{card_to_str(c)}" for pid, c in record.cards
            )
            print(f"    墩 {offset + i + 1}: [{cards_str}] → winner=P{record.winner}")
        if len(state.trick_history) > 5:
            print(f"    ... (前 {offset} 墩省略)")
        print()


def format_q_table(action_scores: list[dict], my_team: int,
                   legal_cards: list[Card]) -> str:
    legal_set = {c for c in legal_cards}
    lines = [
        f"  {'Action':>8s}  {'AggQ':>10s}  {'Legal':>6s}",
        f"  {'-' * 8}  {'-' * 10}  {'-' * 6}",
        f"  (队伍 {my_team}, 选 {'max' if my_team == 0 else 'min'} Q)",
    ]
    for entry in action_scores:
        card = entry["action"]
        q = entry["value"]
        legal = "✓" if card in legal_set else ""
        lines.append(f"  {card_to_str(card):>8s}  {q:>10.4f}  {legal:>6s}")
    return "\n".join(lines)


def print_debug_exact(info: dict[str, Any], player: RuleExactFirst4Player,
                      legal_cards: list[Card]) -> None:
    debug = info.get("debug", {})
    if not debug:
        print("  (无调试信息)")
        return

    pool = debug.get("pool", {})
    print("-" * 70)
    print("  【IS Pool 统计】")
    pool_size = pool.get("pool_size", 0)
    print(f"    Raw proposals:   {pool_size}")
    w_min = pool.get("pool_weights_min")
    w_max = pool.get("pool_weights_max")
    w_mean = pool.get("pool_weights_mean")
    if w_min is not None and w_max is not None:
        print(f"    Weight range:    [{w_min:.6f}, {w_max:.6f}]")
    else:
        print("    Weight range:    N/A (pool empty)")
    if w_mean is not None:
        print(f"    Weight mean:     {w_mean:.6f}")
    if pool_size == 0:
        print("    (Pool empty — uniform determinization fallback)")

    unique = debug.get("unique_proposals")
    if unique is not None and len(unique) > 0:
        print(f"  去重后 proposals: {len(unique)} 个")
        print("    Top-5 权重:")
        for i, up in enumerate(unique[:5]):
            print(f"      #{i + 1}: w={up['weight_raw']:.6f}")
        if len(unique) > 5:
            print(f"      ... ({len(unique) - 5} more)")

    samples = debug.get("samples")
    n_used = info.get("samples", len(samples) if samples else 0)
    print(f"\n  【采样求解: {n_used} 个】")

    if samples and len(samples) > 0:
        n_fill = debug.get("n_fill", 0)
        n_is = debug.get("n_is", len(samples) - n_fill)

        # ── 所有采样的 norm_w ──
        print(f"    采样明细 (fill={n_fill}, IS={n_is}):")

        if n_fill > 0:
            print(f"    ┌─ 花色多样性补全 (fill) ─ {n_fill} 个:")
            for idx in range(n_fill):
                s = samples[idx]
                nw = s["norm_weight"]
                q_vals = s.get("action_q_values", {})
                q_str = " | ".join(f"{c}={q:.1f}" for c, q in sorted(q_vals.items(), key=lambda x: x[1], reverse=True))
                print(f"    │  #{idx + 1}  norm_w={nw:.6f}  Q: {q_str}")
            print(f"    └{'─' * 50}")

        if n_is > 0:
            print(f"    ┌─ 重要性采样 (IS) ─ {n_is} 个:")
            for idx in range(n_fill, n_fill + n_is):
                s = samples[idx]
                nw = s["norm_weight"]
                q_vals = s.get("action_q_values", {})
                q_str = " | ".join(f"{c}={q:.1f}" for c, q in sorted(q_vals.items(), key=lambda x: x[1], reverse=True))
                print(f"    │  #{idx + 1}  norm_w={nw:.6f}  Q: {q_str}")
            print(f"    └{'─' * 50}")

        # ── Q 值分布汇总 ──
        def _q_sig(s: dict) -> str:
            qv = s.get("action_q_values", {})
            if not qv:
                return "(none)"
            return " | ".join(
                f"{c}={q:.1f}" for c, q in
                sorted(qv.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        sig_counter: dict[str, list[dict]] = {}
        for s in samples:
            sig_counter.setdefault(_q_sig(s), []).append(s)
        sorted_sigs = sorted(sig_counter.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"    Q 值分布:")
        for sig, group in sorted_sigs:
            cnt = len(group)
            print(f"      [{cnt}个] {sig}")

        # ── 前 5 个采样的全部手牌 ──
        show_n = min(5, len(samples))
        print(f"\n    前 {show_n} 个采样的全部手牌:")
        for idx in range(show_n):
            s = samples[idx]
            nw = s["norm_weight"]
            q_vals = s.get("action_q_values", {})
            q_str = " | ".join(f"{c}={q:.1f}" for c, q in sorted(q_vals.items(), key=lambda x: x[1], reverse=True))
            print(f"    ┌─ Sample #{idx + 1}  norm_w={nw:.6f}")
            print(f"    │  Q: {q_str if q_str else '(none)'}")
            all_h = s.get("all_hands", s.get("opponent_hands", {}))
            for ps in sorted(all_h, key=int):
                cards = all_h[ps]
                marker = " ← 自己" if int(ps) == player.position else ""
                print(f"    │  P{ps}: {' '.join(cards)}{marker}")
            print(f"    └{'─' * 50}")

    agg_q_raw = debug.get("agg_q_raw")
    if agg_q_raw and len(agg_q_raw) > 0:
        print("\n  【聚合 Q (raw)】")
        for cs, qv in sorted(agg_q_raw.items(), key=lambda x: x[1], reverse=True):
            print(f"    {cs}: {qv:.4f}")

    my_team = debug.get("my_team", 0)
    print(f"\n  【最终 Action Q】 my_team={my_team}, "
          f"remaining={debug.get('remaining_cards', '?')}")
    action_scores = info.get("action_scores", [])
    if action_scores:
        print(format_q_table(action_scores, my_team, legal_cards))
    else:
        print("  (无)")
    print()


def print_result(player: RuleExactFirst4Player, legal_cards: list[Card],
                 raw_data: dict[str, Any]) -> None:
    info = player.last_play_info
    print("=" * 70)
    print("  结果")
    print("=" * 70)

    mode = info.get("mode", "unknown")
    print(f"  模式: {mode}  ({MODE_DESCRIPTIONS.get(mode, '未知')})")

    if mode == "exact_is_determinized":
        best_value = info.get("best_value", 0.0)
        n_samples = info.get("samples", 0)
        remaining = sum(len(h) for h in raw_data.get("_state_hands", [[]]))
        print(f"  采样数:   {n_samples}")
        print(f"  剩余牌数: {remaining}")
        print(f"  Best Q:   {best_value:.4f}")
        print()
        print_debug_exact(info, player, legal_cards)
    elif mode == "rule_first4":
        print("  (规则式出牌，无采样信息)")

    print("-" * 70)
    print(f"  合法动作 ({len(legal_cards)} 个):")
    for c in sorted(legal_cards, key=lambda c: (c.suit.value, -c.rank.value)):
        print(f"    {card_to_str(c)}")
    print()


def print_example() -> None:
    example = """\
# 示例：seat 3 (东) 初始 13 张全是梅花，前 3 墩打完，第 4 墩 seat 3 首攻
position: 3
hand: CA, CK, CQ, CJ, CT, C9, C8, C7, C6, C5, C4, C3, C2
bids: bid_4, bid_2, bid_3, bid_2
plays:
0 S2
1 H2
2 D2
3 C2
0 S3
1 H3
2 D3
3 C3
0 S4
1 H4
2 D4
3 C4
"""
    print(example)
    print("说明:")
    print("  position: 你的座位 (0=南, 1=西, 2=北, 3=东)")
    print("  hand: 你的全部 13 张初始手牌（程序自动扣除已出牌）")
    print("  bids: 四个玩家的叫牌 (按 seat 0,1,2,3)")
    print("  plays: 所有已出牌，按时间顺序，每行 '座位 牌'")
    print("         每 4 张自动分为一墩，最后不足 4 张为当前桌面")
    print()
    print("用法:")
    print("  python debug_rule_exact_first4.py state.txt        # 文本格式（推荐）")
    print("  python debug_rule_exact_first4.py state.txt --exact # 强制 exact solver")
    print("  python debug_rule_exact_first4.py state.txt --fast  # 快速调试")


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调试 RuleExactFirst4Player — 输入牌局，查看动作和 Q 值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("state_file", nargs="?", default="",
                        help="牌局文件 (.txt 推荐 或 .json)")
    parser.add_argument("--example", action="store_true", help="打印示例")
    parser.add_argument("--exact", action="store_true", help="强制 exact solver")
    parser.add_argument("--threshold", type=int, default=36, help="exact 阈值 (默认 36)")
    parser.add_argument("--fast", action="store_true", help="快速参数 (减小 IS pool)")
    parser.add_argument("--config", type=str, default="", help="超参数 YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.example:
        print_example()
        return

    if not args.state_file:
        print("错误: 请指定牌局文件。")
        print("用法: python debug_rule_exact_first4.py <state.txt>")
        print("      python debug_rule_exact_first4.py --example")
        sys.exit(1)

    # 加载状态
    raw_data, state, _ = load_state(args.state_file)
    position = raw_data.get("position", 0)
    threshold = 52 if args.exact else args.threshold

    # 选择配置
    if args.config:
        config = HyperparamConfig.from_yaml(args.config)
    elif args.fast:
        config = FAST_EXACT_CONFIG
    else:
        default_config = REPO_ROOT / "configs" / "8.yaml"
        config = HyperparamConfig.from_yaml(str(default_config)) if default_config.exists() \
            else HyperparamConfig.default()

    # 创建 player
    solver = ExactDoubleDummyCppFastestSolver()
    player = RuleExactFirst4Player(
        exact_solver=solver,
        exact_threshold=threshold,
        debug=True,
        hyperparam_config=config,
    )

    # 回放历史（含扣减已出牌）
    replay_history(state, player)
    _deduct_played_cards(state)

    remaining = sum(len(h) for h in state.hands)
    top_k, max_samples = config.budget.lookup(max(remaining, 1))
    print(f"  [配置] threshold={threshold}, num_proposals={config.num_proposals}, "
          f"min_pool={config.min_pool_size}, "
          f"remaining_in={remaining}, budget=({top_k}, {max_samples})")
    print()

    # 计算合法动作
    rule = SpadesRules()
    legal_cards = rule.playable(state, state.hands[position], position)

    # state_view
    state_view: dict[str, Any] = {
        "state": state,
        "player_id": position,
        "hand": list(state.hands[position]),
        "hand_size": [len(h) for h in state.hands],
        "phase": state.phase,
        "table_cards": list(state.table_cards),
        "lead_suit": state.lead_suit,
        "trump_suit": state.trump_suit,
        "trump_broken": state.trump_broken,
        "bids": [(b.player_id, b.value, b.is_pass) for b in state.bids],
        "tricks_won": list(state.tricks_won),
        "tricks_played": state.tricks_played,
        "teams": list(state.teams),
        "played_bitset": state.played_bitset,
        "dealer_seat": state.dealer_seat,
        "declaration": state.declaration,
        "trick_leader": state.trick_leader,
    }

    print_state_summary(raw_data, state, player)

    print("  计算中...")
    raw_data["_state_hands"] = [list(h) for h in state.hands]
    chosen = player.play_card(legal_cards, state_view)

    print_result(player, legal_cards, raw_data)
    print(f"  ★ 玩家选择: {card_to_str(chosen)}  ★\n")


if __name__ == "__main__":
    main()
