"""Local Python AI backend for the Spades GUI — powered by rule_exact_first4.

The frontend (gui/src/game.js) sends ONLY public history plus the current
player's remaining hand.  This backend reconstructs a partial
`trick_taking.game_state.GameState` and drives the **rule_exact_first4** player:

- First 4 tricks (remaining > exact_threshold): RuleBasedFirst4Player
  (rule-based heuristics, blind to opponents' hands).
- Last 36 cards (remaining <= exact_threshold): the exact double-dummy
  solver with importance-sampling determinization.  It RECONSTRUCTS the
  opponents' hidden hands from public history — it never peeks at the
  human's real cards.
- When someone bids nil/blind_nil: the first 4 tricks switch to a rule-based
  nil strategy (RuleExactFirst4NilPlayer → RuleBasedFirst4NilPlayer) instead
  of the rule-based player.

Bidding uses the GO-MCTS MLP bid model (bid_nsfp.pt) via the bridge, exactly
like DDSPlayer / RLExactPlayer in evaluation.

Hyperparameter config defaults to configs/8.yaml.

The HTTP layer is stateless: every request rebuilds the GameState from the
posted payload and replays the full trick history into the rule-based player,
so there is no cross-request memory to keep in sync.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ── Import paths ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
for _p in (str(REPO_ROOT), str(GO_MCTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS, cards_to_bitset  # noqa: E402
from trick_taking.game_state import GameState, Phase, TrickRecord  # noqa: E402
from trick_taking.forced_outcome import (  # noqa: E402
    ShowdownStateError,
    check_for_showdown,
    validate_showdown_state,
)
from trick_taking.games.spades import SpadesRules  # noqa: E402
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (  # noqa: E402
    ExactDoubleDummyCppFastestSolver,
)

from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer  # noqa: E402
from strategy.hyperparam_config import HyperparamConfig  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# Card / bid parsing (frontend "code" strings ↔ trick_taking objects)
# ────────────────────────────────────────────────────────────────────────
def parse_card_code(code: str) -> Card:
    """Parse a frontend card code such as "AS", "TH", "2C" into a local Card.

    Frontend format: rank chars (2-9,T,J,Q,K,A) followed by suit char (S/H/D/C).
    """
    rank_code, suit_code = code[:-1], code[-1]
    return Card(suit=Suit.from_short(suit_code), rank=Rank.from_short(rank_code))


def card_to_code(card: Card) -> str:
    """Serialize a local Card back to the frontend code string (e.g. "AS")."""
    return f"{card.rank.short}{card.suit.short}"


def numeric_bid_to_str(value: int) -> str:
    """Map a numeric contract (1..13) to the local bid string "bid_k"."""
    return f"bid_{int(value)}"


def frontend_bid_to_local(entry: dict[str, Any] | None) -> Any:
    """Convert a frontend bid {value,type} into a local max_bid value.

    - {type:"nil"}          → "nil"
    - {type:"blind_nil"}    → "blind_nil"
    - {type:"normal",value} → "bid_<value>"
    - None / missing        → None (not yet bid)
    """
    if not entry:
        return None
    bid_type = str(entry.get("type", "normal")).lower()
    if bid_type == "nil":
        return "nil"
    if bid_type in ("blind_nil", "bnil", "blind-nil"):
        return "blind_nil"
    value = int(entry.get("value", 0))
    return numeric_bid_to_str(value)



# ────────────────────────────────────────────────────────────────────────
# payload → trick_taking.GameState  (the human's hidden cards are NOT sent;
# opponents' hands stay empty — the exact solver re-derives them via IS)
# ────────────────────────────────────────────────────────────────────────
def _spades_trick_winner(cards: list[tuple[int, Card]], leader: int) -> int:
    """Winner of a completed trick under Spades rules (spades trump)."""
    lead_suit = cards[0][1].suit
    best_seat = cards[0][0]
    best_card = cards[0][1]
    for seat, card in cards[1:]:
        if card.suit == Suit.SPADES:
            if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                best_seat, best_card = seat, card
        elif card.suit == lead_suit and best_card.suit != Suit.SPADES:
            if card.rank.value > best_card.rank.value:
                best_seat, best_card = seat, card
    return best_seat


def build_local_state(payload: dict[str, Any]) -> tuple[GameState, int]:
    """Reconstruct a partial trick_taking GameState from the frontend payload.

    Returns (state, seat) where `seat` is the AI player to act.

    Only the AI's own hand is *known*; the other three seats are filled with
    PLACEHOLDER cards drawn from the unseen pool, with the correct count each.
    Why placeholders rather than empty hands:

    - RLExactPlayer.play_card decides "policy vs exact" via
      `remaining = sum(len(h) for h in state.hands)`.  With empty opponents
      that sum collapses to ~13 and the player wrongly enters the exact branch
      on trick 1, where IS determinization thrashes.  Correct per-seat counts
      restore the right phase split.
    - The exact branch's main path (`_build_is_pool`/`_generate_proposal`)
      ignores opponents' current hands entirely — it re-derives them from the
      observer's hand + the public play history — so placeholder *identities*
      never affect the decision.  Only the *counts* matter (the uniform
      determinization fallback reads `len(state.hands[pid])`), and those are
      exact.  No hidden human card is ever read.
    """
    seat = int(payload["currentPlayer"])
    phase_name = str(payload.get("phase", "playing"))
    phase = {
        "bidding": Phase.BIDDING,
        "playing": Phase.PLAYING,
        "finished": Phase.SCORING,
    }.get(phase_name, Phase.PLAYING)

    # AI's own remaining hand (the only hand we know)
    hand_codes = payload.get("remainingHand", []) or []
    own_hand = [parse_card_code(str(code)) for code in hand_codes]

    # bids → max_bid[4]
    raw_bids = payload.get("bids", []) or []
    max_bid: list[Any] = [None, None, None, None]
    for i in range(4):
        entry = raw_bids[i] if i < len(raw_bids) else None
        max_bid[i] = frontend_bid_to_local(entry)

    # completed tricks → trick_history + played_bitset + tricks_won
    played_bitset = 0
    public_spade_seen = False
    tricks_won = [0, 0, 0, 0]
    cards_played_by_seat = [0, 0, 0, 0]  # how many cards each seat has shown
    trick_history: list[TrickRecord] = []
    for trick in payload.get("completedTricks", []) or []:
        entry_cards: list[tuple[int, Card]] = []
        for c in trick.get("cards", []):
            cseat = int(c["seat"])
            card = parse_card_code(str(c["card"]))
            entry_cards.append((cseat, card))
            played_bitset |= card.bit
            public_spade_seen = public_spade_seen or card.suit == Suit.SPADES
            cards_played_by_seat[cseat] += 1
        if not entry_cards:
            continue
        leader = entry_cards[0][0]
        winner = _spades_trick_winner(entry_cards, leader)
        tricks_won[winner] += 1
        trick_history.append(
            TrickRecord(cards=entry_cards, winner=winner, leader=leader)
        )

    # current (in-progress) trick → table_cards
    table_cards: list[tuple[int, Card]] = []
    for c in payload.get("currentTrick", []) or []:
        cseat = int(c["seat"])
        card = parse_card_code(str(c["card"]))
        table_cards.append((cseat, card))
        played_bitset |= card.bit
        public_spade_seen = public_spade_seen or card.suit == Suit.SPADES
        cards_played_by_seat[cseat] += 1

    # ── Fill opponents with placeholder cards from the unseen pool ──────────
    # Each non-AI seat should hold (cards_per_hand - cards_it_played) cards.
    # The AI's own count is authoritative from remainingHand.
    seen_ids: set[int] = set(c.card_id for c in own_hand)
    for cid in range(52):
        if played_bitset & (1 << cid):
            seen_ids.add(cid)
    unseen_pool = [c for c in _STANDARD_CARDS if c.card_id not in seen_ids]

    hands: list[list[Card]] = [[] for _ in range(4)]
    hands[seat] = own_hand
    pool_idx = 0
    for pid in range(4):
        if pid == seat:
            continue
        remaining_count = 13 - cards_played_by_seat[pid]
        remaining_count = max(0, remaining_count)
        hands[pid] = unseen_pool[pool_idx: pool_idx + remaining_count]
        pool_idx += remaining_count

    spades_broken = bool(payload.get("spadesBroken", False)) or public_spade_seen
    leader = int(payload.get("leader", seat))

    state = GameState()
    state.num_players = 4
    state.phase = phase
    state.hands = hands
    state.hand_bitsets = [cards_to_bitset(h) for h in hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = max_bid
    state.bids = []  # not needed by encoder / solver; max_bid is authoritative
    state.teams = [0, 1, 0, 1]
    state.turn = seat
    state.current_bidder = seat
    state.trick_leader = leader
    state.table_cards = table_cards
    state.trump_suit = Suit.SPADES
    state.trump_broken = spades_broken
    state.spades_broken = spades_broken
    state.tricks_won = tricks_won
    state.trick_history = trick_history
    state.played_bitset = played_bitset
    state.tricks_played = len(trick_history)
    return state, seat


def build_full_showdown_state(payload: dict[str, Any]) -> GameState:
    """Build and validate the full-information state used only for showdown.

    Unlike :func:`build_local_state`, this function deliberately consumes all
    four real remaining hands.  Its result must only be passed to the exact
    forced-outcome checker, never to an acting bidder or card player.
    """
    raw_hands = payload.get("remainingHands")
    if not isinstance(raw_hands, list) or len(raw_hands) != 4:
        raise ShowdownStateError("remainingHands must contain four hands")
    try:
        hands = [
            [parse_card_code(str(code)) for code in hand]
            for hand in raw_hands
        ]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ShowdownStateError("remainingHands contains an invalid card") from exc

    raw_bids = payload.get("bids", []) or []
    max_bid = [
        frontend_bid_to_local(raw_bids[seat] if seat < len(raw_bids) else None)
        for seat in range(4)
    ]

    played_bitset = 0
    public_spade_seen = False
    tricks_won = [0, 0, 0, 0]
    cards_won: list[list[Card]] = [[] for _ in range(4)]
    trick_history: list[TrickRecord] = []
    for raw_trick in payload.get("completedTricks", []) or []:
        entry_cards: list[tuple[int, Card]] = []
        for entry in raw_trick.get("cards", []) or []:
            seat = int(entry["seat"])
            card = parse_card_code(str(entry["card"]))
            entry_cards.append((seat, card))
            played_bitset |= card.bit
            public_spade_seen = public_spade_seen or card.suit == Suit.SPADES
        if not entry_cards:
            raise ShowdownStateError("completed trick cannot be empty")
        leader = entry_cards[0][0]
        winner = _spades_trick_winner(entry_cards, leader)
        tricks_won[winner] += 1
        cards_won[winner].extend(card for _, card in entry_cards)
        trick_history.append(
            TrickRecord(cards=entry_cards, winner=winner, leader=leader)
        )

    table_cards: list[tuple[int, Card]] = []
    for entry in payload.get("currentTrick", []) or []:
        seat = int(entry["seat"])
        card = parse_card_code(str(entry["card"]))
        table_cards.append((seat, card))
        played_bitset |= card.bit
        public_spade_seen = public_spade_seen or card.suit == Suit.SPADES

    claimed_tricks = payload.get("tricksWon")
    if (
        not isinstance(claimed_tricks, list)
        or len(claimed_tricks) != 4
        or [int(value) for value in claimed_tricks] != tricks_won
    ):
        raise ShowdownStateError("payload tricksWon does not match completed history")

    leader = int(payload.get("leader", payload.get("currentPlayer", 0)))
    turn = int(payload.get("currentPlayer", leader))
    phase_name = str(payload.get("phase", "playing"))
    phase = {
        "bidding": Phase.BIDDING,
        "playing": Phase.PLAYING,
        "finished": Phase.SCORING,
    }.get(phase_name)
    if phase is None:
        raise ShowdownStateError(f"unknown game phase: {phase_name}")
    spades_broken = bool(payload.get("spadesBroken", False)) or public_spade_seen

    state = GameState()
    state.num_players = 4
    state.phase = phase
    state.hands = hands
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = max_bid
    state.bids = []
    state.teams = [0, 1, 0, 1]
    state.turn = turn
    state.current_bidder = turn
    state.trick_leader = leader
    state.table_cards = table_cards
    state.trump_suit = Suit.SPADES
    state.trump_broken = spades_broken
    state.spades_broken = spades_broken
    state.tricks_won = tricks_won
    state.cards_won = cards_won
    state.trick_history = trick_history
    state.played_bitset = played_bitset
    state.tricks_played = len(trick_history)
    validate_showdown_state(state)
    return state


# ────────────────────────────────────────────────────────────────────────
# rule_exact provider — one player instance per seat
# ────────────────────────────────────────────────────────────────────────
@dataclass
class AiChoice:
    kind: str          # "bid" | "play"
    value: int | None = None
    bid_type: str | None = None
    card: str | None = None
    detail: str = ""


def _load_bid_model(path: str, device: str):
    """Load the GO-MCTS MLP bid model; None → heuristic fallback."""
    cp = Path(path)
    if not cp.exists():
        print(f"  [WARN] bid checkpoint not found: {cp} — bidding falls back to heuristic",
              flush=True)
        return None
    try:
        from models import load_bid_mlp_model
        model = load_bid_mlp_model(str(cp.resolve()), device)
        print(f"  [OK] loaded bid model: {cp}", flush=True)
        return model
    except Exception as exc:  # pragma: no cover
        print(f"  [WARN] failed to load bid model {cp}: {exc} — heuristic fallback",
              flush=True)
        return None


class RuleExactProvider:
    """Holds the shared models and one rule_exact_first4 player per seat.

    Stateless across HTTP requests: each request rebuilds the GameState and
    replays the full trick history into the rule-based player's internal state.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        device = args.device
        self.device = device
        self.exact_threshold = int(args.exact_threshold)
        self.seed = args.seed

        print("Loading rule_exact_first4 models ...", flush=True)

        # Load hyperparam config
        self.hyperparam_config = HyperparamConfig.from_yaml(args.config)
        print(f"  [OK] loaded config: {args.config}", flush=True)

        self.bid_model = _load_bid_model(args.bid_checkpoint, device)

        # The wrapper serializes entry to the native process-global caches, so
        # this instance can be shared safely across seats.
        self.exact_solver = ExactDoubleDummyCppFastestSolver()
        self.rules = SpadesRules()
        # RuleExact players keep mutable replay history.  The HTTP server is
        # threaded, so reset/replay/action selection must be one transaction.
        self._decision_lock = threading.Lock()

        # One player instance per seat (keeps position / trajectory isolated).
        # RuleExactFirst4NilPlayer: rule-based nil strategy for first 4 tricks
        # when someone bids nil; non-nil → parent's RuleBasedFirst4Player;
        # remaining <= threshold → IS pool exact solver.
        self.players: list[RuleExactFirst4NilPlayer] = [
            RuleExactFirst4NilPlayer(
                exact_solver=self.exact_solver,
                exact_threshold=self.exact_threshold,
                bid_model=self.bid_model,
                bid_device=device,
                hyperparam_config=self.hyperparam_config,
                num_workers=args.num_workers,
            )
            for _ in range(4)
        ]
        self.ai_name = "rule_exact"

    # ── core dispatch ────────────────────────────────────────────────
    def choose_action(self, payload: dict[str, Any]) -> AiChoice:
        with self._decision_lock:
            return self._choose_action_serialized(payload)

    def check_showdown(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Check a complete authoritative state without touching AI players."""
        state = build_full_showdown_state(payload)
        return check_for_showdown(
            state,
            self.exact_solver,
            time_budget_seconds=1.0,
        ).to_payload()

    def _choose_action_serialized(self, payload: dict[str, Any]) -> AiChoice:
        state, seat = build_local_state(payload)
        player = self.players[seat]

        # Reconstruct the AI's original 13-card hand (current hand + already
        # played cards) so the rule-based player can compute its preference
        # order correctly from the full starting hand.
        ai_played: list[Card] = []
        for trick in state.trick_history:
            for pid, card in trick.cards:
                if pid == seat:
                    ai_played.append(card)
        for pid, card in state.table_cards:
            if pid == seat:
                ai_played.append(card)
        original_hand = list(state.hands[seat]) + ai_played

        # Reset the player and replay the full public card-play history so the
        # internal rule-based player (and nil rule player) have up-to-date
        # tracked state (_history, _opp_led_suits, _our_first_led_suit, etc.).
        player.start_game(seat, original_hand, 4)
        player.set_teams(state.teams, state.max_bid)

        for trick in state.trick_history:
            for pid, card in trick.cards:
                player.card_played(pid, card)
        for pid, card in state.table_cards:
            player.card_played(pid, card)

        view = state.get_player_view(seat)
        view["state"] = state  # the contract play_card expects

        if state.phase == Phase.BIDDING:
            return self._choose_bid(player, view)
        if state.phase == Phase.PLAYING:
            return self._choose_play(player, state, seat, view)
        raise ValueError(f"AI invoked in invalid phase: {state.phase}")

    def _choose_bid(self, player: RuleExactFirst4NilPlayer, view: dict[str, Any]) -> AiChoice:
        # Present a normal single-round bid menu (no blind_nil prompt — the GUI
        # has a flat one-shot bidding flow).  place_bid routes through the MLP
        # bid model via the bridge.
        legal_bids = ["nil"] + [numeric_bid_to_str(i) for i in range(1, 14)]
        raw = player.place_bid(legal_bids, view)
        if raw == "nil":
            return AiChoice(kind="bid", value=0, bid_type="nil", detail="mlp_bid")
        if isinstance(raw, str) and raw.startswith("bid_"):
            return AiChoice(kind="bid", value=int(raw.split("_")[1]),
                            bid_type="normal", detail="mlp_bid")
        # Defensive fallback
        return AiChoice(kind="bid", value=1, bid_type="normal", detail="fallback")

    def _choose_play(self, player: RuleExactFirst4NilPlayer, state: GameState, seat: int,
                     view: dict[str, Any]) -> AiChoice:
        # Nil detection and rule-based nil setup already done by set_teams() in
        # choose_action().  RuleExactFirst4NilPlayer.play_card internally routes:
        #   - nil game + first 4 tricks → RuleBasedFirst4NilPlayer (rule-based)
        #   - non-nil + first 4 tricks  → RuleBasedFirst4Player (rule-based)
        #   - remaining <= exact_threshold → IS pool exact solver

        legal_cards = self.rules.playable(state, state.hands[seat], seat)
        if not legal_cards:
            raise ValueError(f"seat {seat} has no legal cards to play")

        card = player.play_card(legal_cards, view)
        if card not in legal_cards:
            card = legal_cards[0]
        mode = ""
        if isinstance(player.last_play_info, dict):
            mode = str(player.last_play_info.get("mode", ""))
        return AiChoice(kind="play", card=card_to_code(card), detail=mode)


def choice_to_payload(choice: AiChoice, ai_name: str) -> dict[str, Any]:
    if choice.kind == "bid":
        return {
            "kind": "bid",
            "ai": ai_name,
            "bid": {"value": choice.value, "type": choice.bid_type},
            "label": "Nil" if choice.bid_type == "nil" else str(choice.value),
            "detail": choice.detail,
        }
    return {
        "kind": "play",
        "ai": ai_name,
        "card": choice.card,
        "label": choice.card,
        "detail": choice.detail,
    }


# ────────────────────────────────────────────────────────────────────────
# HTTP server
# ────────────────────────────────────────────────────────────────────────
def build_response_handler(provider: RuleExactProvider):
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
                self._send_json(200, {"ok": True, "ai": provider.ai_name, "seed": provider.seed})
                return
            self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            action_paths = {"/api/choose-action", "/choose-action"}
            showdown_paths = {"/api/check-showdown", "/check-showdown"}
            if self.path not in action_paths | showdown_paths:
                self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if self.path in showdown_paths:
                try:
                    result = provider.check_showdown(payload)
                    self._send_json(200, {"ok": True, **result})
                except (ShowdownStateError, KeyError, TypeError, ValueError) as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:  # pragma: no cover - surfaced to browser
                    import traceback
                    traceback.print_exc()
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return

            try:
                choice = provider.choose_action(payload)
                self._send_json(
                    200,
                    {"ok": True, **choice_to_payload(choice, provider.ai_name)},
                )
            except Exception as exc:  # pragma: no cover - surfaced to browser
                import traceback
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rule_exact_first4 AI backend for the Spades GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ai", default="rule_exact",
                        help="Kept for compatibility; only rule_exact is served.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="remaining cards <= this → exact solver (first "
                             "52-threshold cards use rule-based / nil policy)")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "configs" / "8.yaml"),
                        help="path to hyperparam YAML config for RuleExactFirst4NilPlayer")
    parser.add_argument("--checkpoint-nil", type=str,
                        default=str(REPO_ROOT / "55_2nil.pt"),
                        help="[deprecated] nil strategy now uses RuleBasedFirst4NilPlayer; "
                             "this arg is ignored")
    parser.add_argument("--bid-checkpoint", type=str,
                        default=str(REPO_ROOT / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt"))
    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of parallel solver workers (0=auto, 1=sequential)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible dealing/determinization")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Set RNG seeds for reproducible behavior across common libs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    provider = RuleExactProvider(args)
    server = ThreadingHTTPServer((args.host, args.port), build_response_handler(provider))
    print(f"rule_exact backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"  exact_threshold={provider.exact_threshold} "
          f"(first {52 - provider.exact_threshold} cards use rule-based / nil policy)", flush=True)
    print(f"  solver_workers={provider.players[0]._num_workers}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
