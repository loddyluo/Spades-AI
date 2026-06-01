"""Smoke test for the GUI-to-Python AI bridge.

This script exercises the same payload format that the GUI sends to the
backend: current seat's remaining hand plus public bidding / trick history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLAB_ROOT = REPO_ROOT / "Spades_AI_GO-MCTS"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLAB_ROOT))

from gui.backend import AiProvider
from spades_ai.game.deck import deal_hands
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.game.scoring import BidType, compute_hand_scores, PlayerResult
from spades_ai.game.state import Bid, GameState, Phase


_RANK_TO_CODE = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
_SUIT_TO_CODE = {0: "C", 1: "D", 2: "H", 3: "S"}
_CODE_TO_RANK = {value: key for key, value in _RANK_TO_CODE.items()}
_CODE_TO_SUIT = {value: key for key, value in _SUIT_TO_CODE.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the Python AI bridge")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ai", default="rule_v2")
    parser.add_argument("--auto-human", action="store_true", help="Auto-play the human seat too")
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1, 2, 3])
    return parser.parse_args()


def card_code(card) -> str:
    return f"{_RANK_TO_CODE[int(card.rank.value)]}{_SUIT_TO_CODE[int(card.suit.value)]}"


def code_to_card(code: str):
    from spades_ai.game.card import Card, Rank, Suit

    return Card(Rank(_CODE_TO_RANK[code[:-1]]), Suit(_CODE_TO_SUIT[code[-1]]))


def serialize_bid(bid: Bid | None) -> dict | None:
    if bid is None:
        return None
    if bid.bid_type == BidType.NIL:
        bid_type = "nil"
    elif bid.bid_type == BidType.BLIND_NIL:
        bid_type = "blind_nil"
    else:
        bid_type = "normal"
    return {"value": int(bid.value), "type": bid_type}


def serialize_trick(trick) -> dict:
    return {
        "cards": [{"seat": entry.player, "card": card_code(entry.card)} for entry in trick.cards],
    }


def build_payload(state: GameState, human_seat: int) -> dict:
    return {
        "phase": state.phase.name.lower(),
        "currentPlayer": state.current_player,
        "leader": state.leader,
        "trickNumber": state.trick_number,
        "spadesBroken": state.spades_broken,
        "humanSeat": human_seat,
        "remainingHand": [card_code(card) for card in state.hands[state.current_player]],
        "remainingCounts": [len(hand) for hand in state.hands],
        "bids": [serialize_bid(bid) for bid in state.bids],
        "completedTricks": [serialize_trick(trick) for trick in state.completed_tricks],
        "currentTrick": [{"seat": entry.player, "card": card_code(entry.card)} for entry in state.current_trick_cards],
    }


def apply_ai_choice(state: GameState, provider: AiProvider) -> GameState:
    choice = provider.choose_action(build_payload(state, human_seat=99))
    if choice.kind == "bid":
        bid_type = BidType.NIL if choice.bid_type == "nil" else BidType.BLIND_NIL if choice.bid_type == "blind_nil" else BidType.NORMAL
        return state.place_bid(Bid(value=int(choice.value or 0), bid_type=bid_type))
    return state.play_card(next(card for card in state.hands[state.current_player] if card_code(card) == choice.card))


def main() -> None:
    args = parse_args()
    provider = AiProvider(args.ai)
    hands = deal_hands(args.seed)
    state = GameState.new_game(hands)

    while state.phase == Phase.BIDDING:
        seat = state.current_player
        if seat == args.human_seat and not args.auto_human:
            legal = ["nil"] + [str(index) for index in range(1, 14)]
            chosen = legal[0]
            bid = Bid(value=0, bid_type=BidType.NIL) if chosen == "nil" else Bid(value=int(chosen), bid_type=BidType.NORMAL)
            state = state.place_bid(bid)
        else:
            state = apply_ai_choice(state, provider)

    while state.phase == Phase.PLAYING:
        seat = state.current_player
        if seat == args.human_seat and not args.auto_human:
            legal = get_legal_moves(
                hand=state.hands[seat],
                led_suit=state.led_suit,
                spades_broken=state.spades_broken,
                is_leading=len(state.current_trick_cards) == 0,
            )
            state = state.play_card(sorted(legal, key=lambda card: card.index)[0])
        else:
            state = apply_ai_choice(state, provider)

    team_ns, team_ew = compute_hand_scores(
        [
            PlayerResult(bid=state.bids[0].value, bid_type=state.bids[0].bid_type, tricks_won=state.tricks_won[0]),
            PlayerResult(bid=state.bids[1].value, bid_type=state.bids[1].bid_type, tricks_won=state.tricks_won[1]),
            PlayerResult(bid=state.bids[2].value, bid_type=state.bids[2].bid_type, tricks_won=state.tricks_won[2]),
            PlayerResult(bid=state.bids[3].value, bid_type=state.bids[3].bid_type, tricks_won=state.tricks_won[3]),
        ]
    )

    print(f"Bridge smoke test passed: NS={team_ns}, EW={team_ew}, ai={args.ai}")


if __name__ == "__main__":
    main()