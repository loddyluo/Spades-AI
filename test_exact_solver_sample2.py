#!/usr/bin/env python3
"""Test exact solver Q values for Sample #2 from seed=8900051 seat=3 before play [40].

直接对 Sample #2 的初始手牌，回放到 before play [40]，调用精确求解器看 Q 值。
"""
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from trick_taking.game_state import GameState, TrickRecord, Phase
from trick_taking.card import Card, Suit
from trick_taking.games.spades import SpadesRules
from trick_taking.card import _STANDARD_CARDS as STANDARD_52
from trick_taking.solvers.exact_double_dummy_cpp_fastest import ExactDoubleDummyCppFastestSolver

# ── Sample #2 初始手牌 ──────────────────────────────────────────
sample_hands_str = [
    # P0
    "DT H4 C5 HQ C7 D7 SA H5 SK S5 HT ST SQ",
    # P1
    "DK HK C3 H7 CJ D3 CT D9 D2 S2 S9 C9 S4",
    # P2
    "D4 HA CA H9 C2 DJ CQ H8 D5 S8 S6 H3 H6",
    # P3
    "D8 S3 DQ S7 DA SJ C4 C6 H2 C8 CK HJ D6",
]

def parse_hand(s: str) -> list[Card]:
    """Parse space-separated suit+rank cards."""
    return [Card.from_str(c.strip()) for c in s.split()]

# ── 出牌序列 (39 张，来自实际对局) ──────────────────────────────
# 格式: (seat, card_str)
PLAY_SEQUENCE = [
    (0, "DT"), (1, "DK"), (2, "D4"), (3, "D6"),   # 墩1: P0 T♦ P1 K♦ P2 4♦ P3 6♦ → P1 win
    (1, "HK"), (2, "HA"), (3, "H2"), (0, "H4"),   # 墩2: P1 K♥ P2 A♥ P3 2♥ P0 4♥ → P2 win
    (2, "CA"), (3, "C4"), (0, "C5"), (1, "C3"),   # 墩3: P2 A♣ P3 4♣ P0 5♣ P1 3♣ → P2 win
    (2, "H9"), (3, "HJ"), (0, "HQ"), (1, "H7"),   # 墩4: P2 9♥ P3 J♥ P0 Q♥ P1 7♥ → P0 win
    (0, "C7"), (1, "CJ"), (2, "C2"), (3, "C6"),   # 墩5: P0 7♣ P1 J♣ P2 2♣ P3 6♣ → P1 win
    (1, "D3"), (2, "DJ"), (3, "DA"), (0, "D7"),   # 墩6: P1 3♦ P2 J♦ P3 A♦ P0 7♦ → P3 win
    (3, "C8"), (0, "SA"), (1, "CT"), (2, "CQ"),   # 墩7: P3 8♣ P0 A♠ P1 T♣ P2 Q♣ → P0 win(trump)
    (0, "H5"), (1, "D9"), (2, "H8"), (3, "CK"),   # 墩8: P0 5♥ P1 9♦ P2 8♥ P3 K♣ → P2 win
    (2, "D5"), (3, "DQ"), (0, "SK"), (1, "D2"),   # 墩9: P2 5♦ P3 Q♦ P0 K♠ P1 2♦ → P0 win(trump)
    (0, "S5"), (1, "S2"), (2, "S8"),               # 墩10(进行中): P0 S5 P1 S2 P2 S8 → 等 P3
]

def build_state(hands: list[list[Card]], play_seq: list[tuple[int, str]]) -> GameState:
    """构建 GameState 并回放到 before play [40]."""
    id_to_card = {c.card_id: c for c in STANDARD_52}

    state = GameState()
    state.init_for_deal(4, [list(h) for h in hands], [], list(STANDARD_52))
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    state.teams = [0, 1, 0, 1]  # 0&2 vs 1&3
    state.max_bid = ["bid_5", "bid_1", "bid_2", "bid_3"]
    state.tricks_played = 9  # 已完成 9 墩

    # 按墩来回放
    rule = SpadesRules()
    tricks_raw = []
    for i in range(0, 36, 4):
        tricks_raw.append(play_seq[i:i+4])
    if len(play_seq) > 36:
        tricks_raw.append(play_seq[36:])  # 当前不完整墩

    for ti, trick_cards in enumerate(tricks_raw):
        if ti < 9:  # 完整墩
            state.turn = trick_cards[0][0]
            state.trick_leader = trick_cards[0][0]
            state.table_cards = []

            for p, cs in trick_cards:
                card = Card.from_str(cs)
                # 检查 spades_broken
                if card.suit == Suit.SPADES:
                    state.spades_broken = True
                    state.trump_broken = True
                state.play_card_to_table(p, card)

            # 判定赢家
            winner = rule.winner_trick(state)
            state.tricks_won[winner] += 1
            state.trick_history.append(TrickRecord(
                list(state.table_cards), winner, state.trick_leader,
            ))
            state.table_cards = []
        else:
            # 当前不完整墩
            for p, cs in trick_cards:
                card = Card.from_str(cs)
                if card.suit == Suit.SPADES:
                    state.spades_broken = True
                    state.trump_broken = True
                state.play_card_to_table(p, card)
            state.turn = (state.trick_leader + len(trick_cards)) % 4

    state.turn = 3  # seat 3's turn
    return state


def main():
    hands = [parse_hand(s) for s in sample_hands_str]
    state = build_state(hands, PLAY_SEQUENCE)

    print("=== GameState Before Play [40] ===")
    print(f"Turn: {state.turn}")
    print(f"Spades broken: {state.spades_broken}")
    print(f"Table: {[(p, str(c)) for p, c in state.table_cards]}")
    print(f"Tricks played: {state.tricks_played}")
    print(f"Tricks won: {state.tricks_won}")
    for p in range(4):
        cards = [str(c) for c in state.hands[p]]
        print(f"  P{p} hand ({len(cards)}): {', '.join(cards)}")

    # 合法动作
    rule = SpadesRules()
    legal = rule.playable(state, state.hands[3], 3)
    print(f"\nLegal for P3: {[str(c) for c in legal]}")

    # ── 调用精确求解器 ──
    solver = ExactDoubleDummyCppFastestSolver()
    result = solver.solve_with_q(state)
    print(f"\n=== Exact Solver Result ===")
    print(f"Score: {result.get('score')}")
    print(f"Declarer tricks: {result.get('declarer_tricks')}")

    action_q = result.get("action_q_values", {})
    if action_q:
        print(f"\nAction Q values:")
        for card, q in sorted(action_q.items(), key=lambda x: -x[1]):
            marker = " <-- legal" if card in legal else ""
            print(f"  {card}: {q:+8.6f}{marker}")
    else:
        print("No action_q_values returned")


if __name__ == "__main__":
    main()
