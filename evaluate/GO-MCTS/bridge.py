"""State and action conversion helpers for cross-repo evaluation.

File purpose:
- Convert local `trick_taking` cards and game states into the collaborator
  repository's `spades_ai` equivalents.
- Normalize bid outputs so the local runner can accept bids from either side
  of the codebase.

Function input/output summary:
- to_go_card(card: LocalCard) -> GoCard
    Input: a local `trick_taking.card.Card`.
    Output: the corresponding collaborator `spades_ai.game.card.Card`.
- to_local_card(card: GoCard) -> LocalCard
    Input: a collaborator card.
    Output: the matching local card object.
- to_go_state(state: LocalGameState) -> GoGameState
    Input: a local `trick_taking.game_state.GameState` snapshot.
    Output: a reconstructed collaborator `spades_ai.game.state.GameState`.
- normalize_bid_for_legal_options(raw_bid: Any, legal_bids: Sequence[Any]) -> Any
    Input: a bid object/value from either codebase and the local legal bids.
    Output: one value from `legal_bids` that the local runner can accept.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COLLAB_ROOT = _REPO_ROOT / "Spades_AI_GO-MCTS"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_COLLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_COLLAB_ROOT))

from trick_taking.card import Card as LocalCard, Rank as LocalRank, Suit as LocalSuit
from trick_taking.game_state import GameState as LocalGameState, Phase as LocalPhase

from spades_ai.game.card import Card as GoCard, Rank as GoRank, Suit as GoSuit
from spades_ai.game.state import Bid as GoBid, GameState as GoGameState, Phase as GoPhase
from spades_ai.game.trick import Trick as GoTrick, TrickCard as GoTrickCard
from spades_ai.game.scoring import BidType as GoBidType


def to_go_card(card: LocalCard) -> GoCard:
    """Convert a local card object to the collaborator card class.

    Input:
    - card: local `trick_taking.card.Card`.

    Output:
    - The corresponding collaborator `spades_ai.game.card.Card`.
    """
    return GoCard(GoRank(card.rank.value), GoSuit[card.suit.name])


def to_local_card(card: GoCard) -> LocalCard:
    """Convert a collaborator card object to the local card class.

    Input:
    - card: collaborator `spades_ai.game.card.Card`.

    Output:
    - The corresponding local `trick_taking.card.Card`.
    """
    return LocalCard(LocalSuit[card.suit.name], LocalRank(card.rank.value))


def _to_go_bid_value(raw_bid: Any) -> GoBid | None:
    """Convert a local bid value into the collaborator bid dataclass.

    Input:
    - raw_bid: one of the local bidding values, for example an `int`,
      `"nil"`, `"blind_nil"`, or `"bid_4"`.

    Output:
    - A collaborator `Bid` instance, or `None` when the local value is empty.
    """
    if raw_bid is None:
        return None
    if isinstance(raw_bid, GoBid):
        return raw_bid
    if isinstance(raw_bid, int):
        return GoBid(value=int(raw_bid), bid_type=GoBidType.NORMAL)
    if isinstance(raw_bid, str):
        if raw_bid == "nil":
            return GoBid(value=0, bid_type=GoBidType.NIL)
        if raw_bid == "blind_nil":
            return GoBid(value=0, bid_type=GoBidType.BLIND_NIL)
        if raw_bid.startswith("bid_"):
            return GoBid(value=int(raw_bid.split("_", 1)[1]), bid_type=GoBidType.NORMAL)
    return None


def _convert_completed_trick(record_cards: list[tuple[int, LocalCard]]) -> GoTrick:
    """Convert one local completed trick into the collaborator trick class.

    Input:
    - record_cards: completed trick cards in play order, each as
      `(player_id, local_card)`.

    Output:
    - The corresponding collaborator `Trick` instance.
    """
    go_cards = tuple(GoTrickCard(player=pid, card=to_go_card(card)) for pid, card in record_cards)
    led_suit = go_cards[0].card.suit
    return GoTrick(cards=go_cards, led_suit=led_suit)


def _infer_void_shown(state: LocalGameState) -> tuple[frozenset[GoSuit], frozenset[GoSuit], frozenset[GoSuit], frozenset[GoSuit]]:
    """Infer the collaborator void-tracking tuple from the local trick history.

    Input:
    - state: local game state containing completed tricks and the current table.

    Output:
    - A 4-tuple of frozensets, one per player, describing the suits each
      player has been revealed void in.
    """
    voids: list[set[GoSuit]] = [set(), set(), set(), set()]

    def _consume_trick(cards: list[tuple[int, LocalCard]]) -> None:
        if not cards:
            return
        led_suit = cards[0][1].suit
        for pid, card in cards[1:]:
            if card.suit != led_suit:
                voids[pid].add(GoSuit[led_suit.name])

    for record in state.trick_history:
        _consume_trick(list(record.cards))
    if state.table_cards:
        _consume_trick(list(state.table_cards))

    return tuple(frozenset(suits) for suits in voids)  # type: ignore[return-value]


def to_go_state(state: LocalGameState) -> GoGameState:
    """Reconstruct the collaborator immutable GameState from a local snapshot.

    Input:
    - state: the local mutable `trick_taking.game_state.GameState` snapshot.

    Output:
    - A collaborator `spades_ai.game.state.GameState` with equivalent visible
      information and trick history.
    """
    go_hands = tuple(frozenset(to_go_card(card) for card in hand) for hand in state.hands)

    go_bids = []
    for bid_value in state.max_bid:
        converted = _to_go_bid_value(bid_value)
        if converted is not None:
            go_bids.append(converted)

    completed_tricks = tuple(_convert_completed_trick(list(record.cards)) for record in state.trick_history)
    current_trick_cards = tuple(
        GoTrickCard(player=pid, card=to_go_card(card)) for pid, card in state.table_cards
    )
    go_phase = {
        LocalPhase.BIDDING: GoPhase.BIDDING,
        LocalPhase.PLAYING: GoPhase.PLAYING,
        LocalPhase.SCORING: GoPhase.FINISHED,
        LocalPhase.DEALING: GoPhase.BIDDING,
    }[state.phase]
    trick_number = 0 if state.phase == LocalPhase.BIDDING else state.tricks_played + 1
    current_player = state.current_bidder if state.phase == LocalPhase.BIDDING else state.turn

    return GoGameState(
        hands=go_hands,
        bids=tuple(go_bids),
        completed_tricks=completed_tricks,
        current_trick_cards=current_trick_cards,
        current_player=current_player,
        leader=state.trick_leader,
        trick_number=trick_number,
        tricks_won=tuple(state.tricks_won),
        spades_broken=bool(state.spades_broken or state.trump_broken),
        phase=go_phase,
        void_shown=_infer_void_shown(state),
    )


def normalize_bid_for_legal_options(raw_bid: Any, legal_bids: Sequence[Any]) -> Any:
    """Normalize a bid to one of the local legal bid objects.

    Input:
    - raw_bid: collaborator bid object, integer hand-strength prediction, or
      already-local bid value.
    - legal_bids: the exact legal bid list for the current local state.

    Output:
    - A value that is guaranteed to be a member of `legal_bids` whenever the
      legal bid list is non-empty.
    """
    legal_list = list(legal_bids)
    if not legal_list:
        return None

    def _normal_candidates() -> dict[int, Any]:
        candidates: dict[int, Any] = {}
        for bid in legal_list:
            if isinstance(bid, int):
                candidates[int(bid)] = bid
            elif isinstance(bid, str) and bid.startswith("bid_"):
                candidates[int(bid.split("_", 1)[1])] = bid
        return candidates

    normal_candidates = _normal_candidates()

    if raw_bid is None:
        return legal_list[0]
    if raw_bid in legal_list:
        return raw_bid
    if isinstance(raw_bid, GoBid):
        if raw_bid.bid_type == GoBidType.NIL:
            if "nil" in legal_list:
                return "nil"
            raw_bid = 0
        elif raw_bid.bid_type == GoBidType.BLIND_NIL:
            if "blind_nil" in legal_list:
                return "blind_nil"
            raw_bid = 0
        else:
            raw_bid = raw_bid.value
    if isinstance(raw_bid, str):
        if raw_bid == "nil" and "nil" in legal_list:
            return "nil"
        if raw_bid == "blind_nil" and "blind_nil" in legal_list:
            return "blind_nil"
        if raw_bid.startswith("bid_"):
            raw_bid = int(raw_bid.split("_", 1)[1])
        else:
            return legal_list[0]

    if isinstance(raw_bid, (int, float)):
        target = int(round(raw_bid))
        if target <= 0 and "nil" in legal_list:
            return "nil"
        if target in normal_candidates:
            return normal_candidates[target]
        if normal_candidates:
            nearest = min(normal_candidates, key=lambda value: (abs(value - target), value))
            return normal_candidates[nearest]
        return legal_list[0]

    return legal_list[0]
