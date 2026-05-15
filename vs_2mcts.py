#!/usr/bin/env python3
"""vs_2mcts.py: 双 MCTS 玩家 vs 人类玩家的黑桃王交互程序。

两个 MCTS 玩家在同一队（要么 seats 0&2 = team 0，要么 seats 1&3 = team 1）。
两个 MCTS 玩家彼此看不到对方手牌，各自独立决策（各自使用自己的 TruncatedMCTSStrategy）。

交互格式:
    Line 1:   "02" (seats 0&2 为 MCTS) 或 "13" (seats 1&3 为 MCTS)
    Line 2:   13 cards for MCTS 座位 A (空格分隔)
    Line 3:   13 cards for MCTS 座位 B (空格分隔)

    Then 叫牌阶段 (座位顺序 0→1→2→3):
      - MCTS 玩家: 程序用 hand_strength 计算并输出叫牌
      - 人类玩家: 你输入他们的叫牌

    Then 出牌阶段 (每墩):
      - MCTS 玩家: 程序用 our_mcts 计算并输出
      - 人类玩家: 你输入他们的出牌

Card format (each 2 chars):
    Suit: C/D/H/S, Rank: 2-9/T=10/J/Q/K/A,  e.g. DA=方块A, CT=梅花10

Bid format:
    number (0=nil, 1-13=叫牌数) 或 "nil"/"blind_nil"
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from trick_taking.card import Card, Suit, _STANDARD_CARDS as STANDARD_52
from trick_taking.game_state import GameState, Phase, Bid
from trick_taking.games.spades import SpadesRules
from strategy.truncated_mcts_strategy import TruncatedMCTSStrategy, TruncatedMCTSConfig
from strategy.hand_strength import _hand_strength


def card_to_str(card: Card) -> str:
    return f"{card.suit.short}{card.rank.short}"


def normalize_bid(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("nil", "blind_nil"):
        return raw
    try:
        n = int(raw)
        return "nil" if n == 0 else f"bid_{n}"
    except ValueError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="双 MCTS 玩家 vs 人类")
    parser.add_argument("--checkpoint", type=str, default="result/mlp_test_5.pth")
    parser.add_argument("--simulations-per-action", type=int, default=32)
    parser.add_argument("--mcts-determinization-count", type=int, default=8)
    parser.add_argument("--exact-determinization-count", type=int, default=128)
    args = parser.parse_args()

    def prompt(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # ── Step 1: MCTS seat config ──
    prompt("=== 双 MCTS 黑桃王 ===")
    prompt("MCTS 配置: 输入 02 (座位 0&2 为 MCTS) 或 13 (座位 1&3 为 MCTS)")
    line = sys.stdin.readline().strip()
    if line == "02":
        mcts_seats = [0, 2]
        non_mcts_seats = [1, 3]
    else:
        mcts_seats = [1, 3]
        non_mcts_seats = [0, 2]
    prompt(f"MCTS: 座位 {mcts_seats[0]} & {mcts_seats[1]}")
    prompt(f"人类: 座位 {non_mcts_seats[0]} & {non_mcts_seats[1]}")

    # ── Step 2: Read both MCTS hand cards ──
    prompt(f"输入 MCTS 座位 {mcts_seats[0]} 的手牌 (13张, 空格分隔):")
    hand_a = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]
    prompt(f"输入 MCTS 座位 {mcts_seats[1]} 的手牌 (13张, 空格分隔):")
    hand_b = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]

    # ── Step 3: Build full-information GameState (kept internally) ──
    all_cards = list(STANDARD_52)
    state = GameState()
    state.init_for_deal(4, [[] for _ in range(4)], [], all_cards)
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

    state.hands[mcts_seats[0]] = list(hand_a)
    state.hands[mcts_seats[1]] = list(hand_b)
    known_ids = {c.card_id for c in hand_a} | {c.card_id for c in hand_b}
    remaining = [c for c in all_cards if c.card_id not in known_ids]
    random.Random().shuffle(remaining)
    oidx = 0
    for p in non_mcts_seats:
        state.hands[p] = remaining[oidx: oidx + 13]
        oidx += 13
    state.hand_bitsets = [
        sum(1 << c.card_id for c in state.hands[p]) for p in range(4)
    ]

    # ── Step 4: Bidding (interactive) ──
    prompt("=== 叫牌阶段 ===")
    all_bids: dict[int, str] = {}
    for p in range(4):
        if p in mcts_seats:
            hand_fmt = [(c.suit.short, c.rank.short) for c in state.hands[p]]
            bid_val, _ = _hand_strength(hand_fmt)
            bid_str = "nil" if bid_val == 0 else f"bid_{bid_val}"
            all_bids[p] = bid_str
            print(str(bid_val), flush=True)
            prompt(f"MCTS 座位 {p} 叫 {bid_val}")
        else:
            prompt(f"人类 座位 {p} 的叫牌:")
            bid_raw = sys.stdin.readline().strip()
            all_bids[p] = normalize_bid(bid_raw)
            prompt(f"→ 座位 {p} 叫 {bid_raw}")

    for pid in range(4):
        state.bids.append(Bid(player_id=pid, value=all_bids[pid]))
        state.max_bid.append(all_bids[pid])

    # ── Step 5: Strategy setup (separate instances — each has its own
    #    leaf-value cache, policy-prior cache, and model reference) ──
    shared_config = TruncatedMCTSConfig(
        checkpoint_path=args.checkpoint,
        simulations_per_action=args.simulations_per_action,
        mcts_determinization_count=args.mcts_determinization_count,
        determinization_count=args.exact_determinization_count,
    )
    strategies = {seat: TruncatedMCTSStrategy(shared_config) for seat in mcts_seats}
    rules = SpadesRules()

    # ── Helper functions ──

    def refill_for_observer(obs: int) -> None:
        """Fill non-MCTS hands from remaining card pool.

        BOTH MCTS players' hands are preserved (both are known to the
        program).  Only the human players' hands get random cards from
        the unseen pool, sized to match how many cards each human player
        still holds.  The IS determinization inside the strategy will
        resample them from scratch, so the exact content here is irrelevant.
        """
        preserved = set(mcts_seats)  # Both MCTS players' hands stay intact
        used: set[int] = set()
        for p in preserved:
            for c in state.hands[p]:
                used.add(c.card_id)
        for _, c in state.table_cards:
            used.add(c.card_id)
        for rec in state.trick_history:
            for _, c in rec.cards:
                used.add(c.card_id)
        pool = [c for c in all_cards if c.card_id not in used]
        random.Random().shuffle(pool)
        i = 0
        for p in range(4):
            if p in preserved:
                continue
            played = sum(1 for rec in state.trick_history for pid, _ in rec.cards if pid == p)
            played += sum(1 for pid, _ in state.table_cards if pid == p)
            n = 13 - played
            state.hands[p] = pool[i: i + n]
            i += n

    def ensure_in_hand(card: Card, pid: int) -> None:
        """Make *card* present in *state.hands[pid]* (move from another hand if needed)."""
        if card in state.hands[pid]:
            return
        for q in range(4):
            if q != pid and card in state.hands[q]:
                state.hands[q].remove(card)
                break
        state.hands[pid].append(card)

    def apply_play(card: Card, pid: int) -> None:
        """Play *card* as *pid* and update *state* in-place."""
        # ensure_in_hand may temporarily inflate the hand by moving the card
        # from another placeholder hand; the next refill_for_observer resets
        # all non-observer hands to the correct size.
        ensure_in_hand(card, pid)
        state.play_card_to_table(pid, card)
        if card.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        state.turn = (pid + 1) % 4
        if state.trick_complete:
            winner = rules.winner_trick(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner

    # ── Step 6: Play phase ──
    prompt("=== 出牌阶段 ===")
    for _ in range(13):
        if rules.end_trickgame(state):
            break
        leader = state.trick_leader
        prompt(f"--- 第 {state.tricks_played + 1} 墩, 庄家={leader} ---")
        for pos_in_trick in range(4):
            cur = (leader + pos_in_trick) % 4
            if cur in mcts_seats:
                prompt(f"MCTS (座位 {cur}) 思考中...")
                refill_for_observer(cur)
                # Debug: verify state consistency before MCTS decision
                legal_before = rules.playable(state, state.hands[cur], cur)
                prompt(f"  [debug] 座位 {cur} 手牌: {' '.join(card_to_str(c) for c in state.hands[cur])}")
                prompt(f"  [debug] 合法出牌: {' '.join(card_to_str(c) for c in legal_before)}")
                action = strategies[cur].choose_action(state)
                if action is None:
                    break
                card_str = card_to_str(action)
                # Verify MCTS returned a legal card
                legal_check = rules.playable(state, state.hands[cur], cur)
                if action not in legal_check:
                    prompt(f"  [BUG] MCTS 返回非法牌 {card_str}！合法={[card_to_str(c) for c in legal_check]}")
                    prompt(f"  [BUG] 手牌={[card_to_str(c) for c in state.hands[cur]]}")
                    action = legal_check[0]  # fallback
                    card_str = card_to_str(action)
                print(card_str, flush=True)
                prompt(f"→ MCTS 座位 {cur} 出 {card_str}")
            else:
                prompt(f"人类 座位 {cur} 出牌:")
                line = sys.stdin.readline()
                if not line:
                    return
                action = Card.from_str(line.strip())
                prompt(f"→ 座位 {cur} 出 {card_to_str(action)}")
            apply_play(action, cur)


if __name__ == "__main__":
    main()
