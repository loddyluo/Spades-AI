"""Local Python AI backend for the Spades GUI.

The frontend sends only public history plus the current player's remaining
hand.  The backend reconstructs a partial GameState, lets the configured AI
encode that observation however it wants, and returns one move.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLAB_ROOT = REPO_ROOT / "Spades_AI_GO-MCTS"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLAB_ROOT))

from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.scoring import BidType
from spades_ai.game.state import Bid, GameState, Phase
from spades_ai.game.trick import Trick, TrickCard
from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as RuleBasedPlayerV2


@dataclass(frozen=True)
class AiChoice:
    kind: str
    ai_name: str
    value: int | None = None
    bid_type: str | None = None
    card: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AI backend for the Spades GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ai", default="rule_v2", help="AI spec, e.g. rule_v2 or module:Class")
    return parser.parse_args()


def _rank_from_code(code: str) -> Rank:
    lookup = {
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "T": Rank.TEN,
        "J": Rank.JACK,
        "Q": Rank.QUEEN,
        "K": Rank.KING,
        "A": Rank.ACE,
    }
    return lookup[code]


def _suit_from_code(code: str) -> Suit:
    lookup = {"C": Suit.CLUBS, "D": Suit.DIAMONDS, "H": Suit.HEARTS, "S": Suit.SPADES}
    return lookup[code]


def parse_card_code(code: str) -> Card:
    rank_code, suit_code = code[:-1], code[-1]
    return Card(rank=_rank_from_code(rank_code), suit=_suit_from_code(suit_code))


def card_to_code(card: Card) -> str:
    rank_code = {
        Rank.TWO: "2",
        Rank.THREE: "3",
        Rank.FOUR: "4",
        Rank.FIVE: "5",
        Rank.SIX: "6",
        Rank.SEVEN: "7",
        Rank.EIGHT: "8",
        Rank.NINE: "9",
        Rank.TEN: "T",
        Rank.JACK: "J",
        Rank.QUEEN: "Q",
        Rank.KING: "K",
        Rank.ACE: "A",
    }[card.rank]
    suit_code = {Suit.CLUBS: "C", Suit.DIAMONDS: "D", Suit.HEARTS: "H", Suit.SPADES: "S"}[card.suit]
    return f"{rank_code}{suit_code}"


def parse_bid(payload: dict[str, Any] | None) -> Bid | None:
    if not payload:
        return None
    bid_type = str(payload.get("type", "normal")).lower()
    value = int(payload.get("value", 0))
    if bid_type == "nil":
        return Bid(value=0, bid_type=BidType.NIL)
    if bid_type in {"blind_nil", "bnil", "blind-nil"}:
        return Bid(value=0, bid_type=BidType.BLIND_NIL)
    return Bid(value=value, bid_type=BidType.NORMAL)


def parse_trick_entry(entry: dict[str, Any]) -> TrickCard:
    return TrickCard(player=int(entry["seat"]), card=parse_card_code(str(entry["card"])))


def parse_completed_trick(entry: dict[str, Any]) -> Trick:
    cards = tuple(parse_trick_entry(card_entry) for card_entry in entry.get("cards", []))
    if not cards:
        raise ValueError("completedTricks entry is missing cards")
    led_suit = cards[0].card.suit
    return Trick(cards=cards, led_suit=led_suit)


def compute_void_shown(completed_tricks: list[Trick], current_trick_cards: tuple[TrickCard, ...]) -> tuple[frozenset[Suit], ...]:
    voids = [set() for _ in range(4)]

    def mark_from_trick(trick_cards: tuple[TrickCard, ...]) -> None:
        if len(trick_cards) < 2:
            return
        led_suit = trick_cards[0].card.suit
        for entry in trick_cards[1:]:
            if entry.card.suit != led_suit:
                voids[entry.player].add(led_suit)

    for trick in completed_tricks:
        mark_from_trick(trick.cards)
    mark_from_trick(current_trick_cards)
    return tuple(frozenset(suits) for suits in voids)


def count_tricks_won(completed_tricks: list[Trick]) -> tuple[int, int, int, int]:
    tricks = [0, 0, 0, 0]
    for trick in completed_tricks:
        winner = trick.winner()
        tricks[winner] += 1
    return tuple(tricks)  # type: ignore[return-value]


def build_partial_state(payload: dict[str, Any]) -> tuple[GameState, int]:
    current_player = int(payload["currentPlayer"])
    phase_name = str(payload["phase"])
    phase = Phase.BIDDING if phase_name == "bidding" else Phase.PLAYING if phase_name == "playing" else Phase.FINISHED

    hand_codes = payload.get("remainingHand", [])
    current_hand = frozenset(parse_card_code(str(code)) for code in hand_codes)

    bids = tuple(parse_bid(entry) for entry in payload.get("bids", []) if entry is not None)
    completed_tricks = [parse_completed_trick(entry) for entry in payload.get("completedTricks", [])]
    current_trick_cards = tuple(parse_trick_entry(entry) for entry in payload.get("currentTrick", []))

    tricks_won = count_tricks_won(completed_tricks)
    void_shown = compute_void_shown(completed_tricks, current_trick_cards)

    hands = tuple(
        current_hand if seat == current_player else frozenset()
        for seat in range(4)
    )

    state = GameState(
        hands=hands,
        bids=bids,
        completed_tricks=tuple(completed_tricks),
        current_trick_cards=current_trick_cards,
        current_player=current_player,
        leader=int(payload.get("leader", current_player)),
        trick_number=int(payload.get("trickNumber", 0 if phase == Phase.BIDDING else 1)),
        tricks_won=tricks_won,
        spades_broken=bool(payload.get("spadesBroken", False)),
        phase=phase,
        void_shown=void_shown,
    )
    return state, current_player


class AiProvider:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.player = self._load_player(spec)

    def _load_player(self, spec: str) -> Any:
        if spec in {"rule_v2", "rule_based_v2"}:
            return RuleBasedPlayerV2()
        if ":" in spec:
            module_name, class_name = spec.split(":", 1)
            module = import_module(module_name)
            return getattr(module, class_name)()
        module = import_module(spec)
        if hasattr(module, "create_player"):
            return module.create_player()
        if hasattr(module, "Player"):
            return module.Player()
        raise ValueError(f"Unsupported AI spec: {spec}")

    def choose_action(self, payload: dict[str, Any]) -> AiChoice:
        state, seat = build_partial_state(payload)
        if state.phase == Phase.BIDDING:
            bid = self.player.choose_bid(state)
            bid_kind = "nil" if bid.bid_type == BidType.NIL else "blind_nil" if bid.bid_type == BidType.BLIND_NIL else "normal"
            return AiChoice(kind="bid", ai_name=self.spec, value=int(bid.value), bid_type=bid_kind)
        if state.phase == Phase.PLAYING:
            card = self.player.choose_card(state)
            return AiChoice(kind="play", ai_name=self.spec, card=card_to_code(card))
        raise ValueError(f"AI requested in invalid phase: {state.phase}")


def choice_to_payload(choice: AiChoice) -> dict[str, Any]:
    if choice.kind == "bid":
        return {
            "kind": "bid",
            "ai": choice.ai_name,
            "bid": {"value": choice.value, "type": choice.bid_type},
            "label": "nil" if choice.bid_type == "nil" else str(choice.value),
        }
    return {
        "kind": "play",
        "ai": choice.ai_name,
        "card": choice.card,
        "label": choice.card,
    }


def build_response_handler(provider: AiProvider):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send_json(204, {"ok": True})

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/health", "/api/health"}:
                self._send_json(200, {"ok": True, "ai": provider.spec})
                return
            self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/api/choose-action", "/choose-action"}:
                self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                choice = provider.choose_action(payload)
                self._send_json(200, {"ok": True, **choice_to_payload(choice)})
            except Exception as exc:  # pragma: no cover - surfaced to browser/test output
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    args = parse_args()
    provider = AiProvider(args.ai)
    server = ThreadingHTTPServer((args.host, args.port), build_response_handler(provider))
    print(f"AI backend listening on http://{args.host}:{args.port} using {args.ai}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()