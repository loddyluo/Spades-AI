#!/usr/bin/env python3
"""vs2rl.py: 双 RL-Exact 玩家 vs 人类玩家的黑桃王交互程序。

两个 RL-Exact 玩家在同一队（要么 seats 0&2 = team 0，要么 seats 1&3 = team 1）。
两个 RL-Exact 玩家彼此看不到对方手牌，各自独立决策（各自使用自己的 RLExactPlayer）。

与前 4 墩使用 MLP 策略网络（55_2.pt / 55_2nil.pt），后 9 墩使用 IS 确定化 + 精确求解。
叫牌使用 bid_nsfp.pt MLP 模型。

交互格式:
    Line 1:   "02" (seats 0&2 为 RL) 或 "13" (seats 1&3 为 RL)
    Line 2:   13 cards for RL 座位 A (空格分隔)
    Line 3:   13 cards for RL 座位 B (空格分隔)

    Then 叫牌阶段 (座位顺序 0→1→2→3):
      - RL 玩家: 程序用 bid_nsfp.pt 计算并输出叫牌
      - 人类玩家: 你输入他们的叫牌

    Then 出牌阶段 (每墩):
      - RL 玩家: 前 4 墩用 MLP 策略网络，后 9 墩使用 IS 确定化 + 精确求解
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
sys.path.insert(0, str(REPO_ROOT / "evaluate" / "GO-MCTS"))
sys.path.insert(0, str(REPO_ROOT / "Spades_AI_GO-MCTS"))

from trick_taking.card import Card, Suit, _STANDARD_CARDS as STANDARD_52
from trick_taking.game_state import GameState, Phase, Bid
from trick_taking.games.spades import SpadesRules

from bridge import to_go_state
from models import load_bid_mlp_model, MLPBidPlayer

from rl.policy_network import PolicyMLP
from rl.rl_exact_player import RLExactPlayer
from rl.rl_feature_encoder import RLFeatureEncoder
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)

MODEL_OUTPUT_DIM = 55
MODEL_HIDDEN_DIMS = [1024, 512, 512]


def _load_policy(checkpoint_path: str, device: str) -> PolicyMLP | None:
    """Load a policy network checkpoint."""
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    try:
        net = PolicyMLP(
            input_dim=264, hidden_dims=MODEL_HIDDEN_DIMS,
            output_dim=MODEL_OUTPUT_DIM,
        ).to(device)
        net.eval()
        net.load(str(cp.resolve()), device=device)
        net.eval()
        return net
    except Exception as e:
        print(f"  [WARN] Failed to load policy {cp}: {e}", file=sys.stderr, flush=True)
        return None


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
    parser = argparse.ArgumentParser(
        description="双 RL-Exact 玩家 vs 人类 (MLP 叫牌 + 前4墩策略网络 + 后9墩精确求解)",
    )
    parser.add_argument("--bid-checkpoint", type=str,
                        default="Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt")
    parser.add_argument("--checkpoint-nonil", type=str, default="./55_2.pt",
                        help="RL policy for games where no one bids nil")
    parser.add_argument("--checkpoint-nil", type=str, default="./55_2nil.pt",
                        help="RL policy for games where someone bids nil")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="Remaining cards <= threshold → use exact solver")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")

    args = parser.parse_args()

    # ── Load models ──
    bid_mlp = load_bid_mlp_model(args.bid_checkpoint, args.device)
    bid_player = MLPBidPlayer(bid_mlp, args.device)

    policy_nonil = _load_policy(args.checkpoint_nonil, args.device)
    policy_nil = _load_policy(args.checkpoint_nil, args.device)

    if policy_nonil is None:
        print("  [WARN] Nonil policy not loaded; using random weights",
              file=sys.stderr, flush=True)
        policy_nonil = PolicyMLP(input_dim=264, hidden_dims=MODEL_HIDDEN_DIMS,
                                 output_dim=MODEL_OUTPUT_DIM).to(args.device)
        policy_nonil.eval()
    if policy_nil is None:
        print("  [WARN] Nil policy not loaded; using random weights",
              file=sys.stderr, flush=True)
        policy_nil = PolicyMLP(input_dim=264, hidden_dims=MODEL_HIDDEN_DIMS,
                               output_dim=MODEL_OUTPUT_DIM).to(args.device)
        policy_nil.eval()

    exact_solver = ExactDoubleDummyCppFastestSolver()
    encoder = RLFeatureEncoder()

    def prompt(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # ── Step 1: RL seat config ──
    prompt("=== 双 RL-Exact 黑桃王 ===")
    prompt("RL 配置: 输入 02 (座位 0&2 为 RL) 或 13 (座位 1&3 为 RL)")
    line = sys.stdin.readline().strip()
    if line == "02":
        rl_seats = [0, 2]
        human_seats = [1, 3]
    else:
        rl_seats = [1, 3]
        human_seats = [0, 2]
    prompt(f"RL-Exact: 座位 {rl_seats[0]} & {rl_seats[1]}")
    prompt(f"人类: 座位 {human_seats[0]} & {human_seats[1]}")

    # ── Step 2: Read both RL hand cards ──
    prompt(f"输入 RL 座位 {rl_seats[0]} 的手牌 (13张, 空格分隔):")
    hand_a = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]
    prompt(f"输入 RL 座位 {rl_seats[1]} 的手牌 (13张, 空格分隔):")
    hand_b = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]

    # ── Step 3: Build full-information GameState (kept internally) ──
    all_cards = list(STANDARD_52)
    state = GameState()
    state.init_for_deal(4, [[] for _ in range(4)], [], all_cards)
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

    state.hands[rl_seats[0]] = list(hand_a)
    state.hands[rl_seats[1]] = list(hand_b)
    original_rl_hands = {
        rl_seats[0]: list(hand_a),
        rl_seats[1]: list(hand_b),
    }
    known_ids = {c.card_id for c in hand_a} | {c.card_id for c in hand_b}
    remaining = [c for c in all_cards if c.card_id not in known_ids]
    random.Random().shuffle(remaining)
    oidx = 0
    for p in human_seats:
        state.hands[p] = remaining[oidx: oidx + 13]
        oidx += 13
    state.hand_bitsets = [
        sum(1 << c.card_id for c in state.hands[p]) for p in range(4)
    ]

    # ── Step 4: Create RL-Exact players ──
    rl_players: dict[int, RLExactPlayer] = {}
    for seat in rl_seats:
        player = RLExactPlayer(
            policy_nets=[policy_nonil],
            exact_solver=exact_solver,
            encoder=encoder,
            exact_threshold=args.exact_threshold,
            is_training=False,
            bid_model=bid_mlp,
            bid_device=args.device,
        )
        player.start_game(seat, list(state.hands[seat]), 4)
        rl_players[seat] = player

    rules = SpadesRules()

    # ── Step 5: Bidding (interactive) ──
    prompt("=== 叫牌阶段 ===")
    for p in range(4):
        if p in rl_seats:
            state.turn = p
            go_state = to_go_state(state)
            bid_result = bid_player.choose_bid(go_state)
            bid_type_name = bid_result.bid_type.name
            if bid_type_name == "NIL":
                bid_val = 0
                bid_str = "nil"
            else:
                bid_val = bid_result.value
                if bid_val == 0:
                    bid_val = 1
                bid_str = f"bid_{bid_val}"
            print(str(bid_val), flush=True)
            prompt(f"RL-Exact 座位 {p} 叫 {bid_val}")
        else:
            prompt(f"人类 座位 {p} 的叫牌:")
            bid_raw = sys.stdin.readline().strip()
            bid_str = normalize_bid(bid_raw)
            prompt(f"→ 座位 {p} 叫 {bid_raw}")

        state.bids.append(Bid(player_id=p, value=bid_str))
        state.max_bid.append(bid_str)

    # ── After bidding: check for nil bids and set appropriate policy ──
    nil_bid = any(
        isinstance(bv, str) and bv in ("nil", "blind_nil")
        for bv in state.max_bid
    )
    if nil_bid:
        prompt("检测到 nil 叫牌，切换 nil 策略网络")
        for seat in rl_seats:
            rl_players[seat].policy_nets = [policy_nil]
    else:
        for seat in rl_seats:
            rl_players[seat].policy_nets = [policy_nonil]

    # ── Helper functions ──

    def refill_for_observer(obs: int) -> None:
        """Fill non-observer hands from remaining card pool.

        Only the observer's hand is preserved. All other hands get random
        cards from the unseen pool, sized to match how many cards each
        player still holds.
        """
        for seat in rl_seats:
            played_ids: set[int] = set()
            for rec in state.trick_history:
                for pid, c in rec.cards:
                    if pid == seat:
                        played_ids.add(c.card_id)
            for pid, c in state.table_cards:
                if pid == seat:
                    played_ids.add(c.card_id)
            state.hands[seat] = [
                c for c in original_rl_hands[seat] if c.card_id not in played_ids
            ]

        preserved = {obs}
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
        state.hand_bitsets = [
            sum(1 << c.card_id for c in state.hands[p]) for p in range(4)
        ]

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
            if cur in rl_seats:
                prompt(f"RL-Exact (座位 {cur}) 思考中...")
                refill_for_observer(cur)
                state.turn = cur
                legal_before = rules.playable(state, state.hands[cur], cur)
                prompt(f"  [debug] 座位 {cur} 手牌: {' '.join(card_to_str(c) for c in state.hands[cur])}")
                prompt(f"  [debug] 合法出牌: {' '.join(card_to_str(c) for c in legal_before)}")

                # RLExactPlayer 的 play_card 会内部根据剩余牌数
                # 自动选择 policy_play (前16张) 或 exact_play (后36张)
                action = rl_players[cur].play_card(
                    legal_before, {"state": state},
                )
                if action is None:
                    break
                card_str = card_to_str(action)
                if action not in legal_before:
                    prompt(f"  [BUG] RL 返回非法牌 {card_str}！合法={[card_to_str(c) for c in legal_before]}")
                    action = legal_before[0]
                    card_str = card_to_str(action)
                print(card_str, flush=True)
                prompt(f"→ RL-Exact 座位 {cur} 出 {card_str}")
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
