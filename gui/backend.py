"""Local Python AI backend for the Spades GUI — powered by rl_exact.

The frontend (gui/src/game.js) sends ONLY public history plus the current
player's remaining hand.  This backend reconstructs a partial
`trick_taking.game_state.GameState` and drives the **rl_exact** player:

- First 4 tricks (16 cards, remaining > exact_threshold): a policy MLP
  (55-dim head) chooses the card from self-hand + public info only.
- Last 36 cards (remaining <= exact_threshold): the exact double-dummy
  solver with importance-sampling determinization.  It RECONSTRUCTS the
  opponents' hidden hands from public history — it never peeks at the
  human's real cards.

Checkpoint selection mirrors evaluate/evaluate_dds_vs_rl.py BothPlayer:
- Someone bids nil/blind_nil → 55_2nil.pt
- No one bids nil            → 55_2.pt

Bidding uses the GO-MCTS MLP bid model (bid_nsfp.pt) via the bridge, exactly
like DDSPlayer / RLExactPlayer in evaluation.

The HTTP layer is stateless: every request rebuilds the GameState from the
posted payload, so there is no cross-request memory to keep in sync.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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
from trick_taking.games.spades import SpadesRules  # noqa: E402
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (  # noqa: E402
    ExactDoubleDummyCppFastestSolver,
)

from rl.policy_network import PolicyMLP  # noqa: E402
from rl.rl_exact_player import RLExactPlayer  # noqa: E402
from rl.rl_feature_encoder import RLFeatureEncoder  # noqa: E402

MODEL_INPUT_DIM = 264
MODEL_HIDDEN_DIMS = [1024, 512, 512]
MODEL_OUTPUT_DIM = 55


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


def _has_nil_bid(bids: list[Any]) -> bool:
    """True if any seat has a nil / blind_nil bid (drives checkpoint switch)."""
    return any(isinstance(b, str) and b in ("nil", "blind_nil") for b in bids)


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

    spades_broken = bool(payload.get("spadesBroken", False))
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


# ────────────────────────────────────────────────────────────────────────
# rl_exact provider — one player instance per seat
# ────────────────────────────────────────────────────────────────────────
@dataclass
class AiChoice:
    kind: str          # "bid" | "play"
    value: int | None = None
    bid_type: str | None = None
    card: str | None = None
    detail: str = ""


def _load_policy(path: str, device: str) -> PolicyMLP:
    """Load a 55-dim policy net; fall back to random weights if missing."""
    cp = Path(path)
    net = PolicyMLP(MODEL_INPUT_DIM, list(MODEL_HIDDEN_DIMS), MODEL_OUTPUT_DIM).to(device)
    net.eval()
    if not cp.exists():
        print(f"  [WARN] policy checkpoint not found: {cp} — using random weights",
              flush=True)
        return net
    try:
        net.load(str(cp.resolve()), device=device)
        net.eval()
        print(f"  [OK] loaded policy: {cp}", flush=True)
        return net
    except Exception as exc:  # pragma: no cover
        print(f"  [WARN] failed to load policy {cp}: {exc} — using random weights",
              flush=True)
        return net


def _load_bid_model(path: str, device: str):
    """Load the GO-MCTS MLP bid model; None → RLExactPlayer heuristic fallback."""
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


class RlExactProvider:
    """Holds the shared models and one rl_exact player per seat.

    Stateless across HTTP requests: each request rebuilds the GameState and the
    nil-checkpoint choice from the posted payload alone.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        device = args.device
        self.device = device
        self.exact_threshold = int(args.exact_threshold)
        self.seed = args.seed

        print("Loading rl_exact models ...", flush=True)
        self.policy_nonil = _load_policy(args.checkpoint_nonil, device)
        self.policy_nil = _load_policy(args.checkpoint_nil, device)
        self.bid_model = _load_bid_model(args.bid_checkpoint, device)

        # The solver/encoder are safe to share read-only across seats.
        self.exact_solver = ExactDoubleDummyCppFastestSolver()
        self.encoder = RLFeatureEncoder()
        self.rules = SpadesRules()

        # One player instance per seat (keeps position / trajectory isolated).
        self.players: list[RLExactPlayer] = [
            RLExactPlayer(
                policy_nets=[self.policy_nonil],
                exact_solver=self.exact_solver,
                encoder=self.encoder,
                exact_threshold=self.exact_threshold,
                is_training=False,           # argmax / greedy, no exploration
                bid_model=self.bid_model,
                bid_device=device,
            )
            for _ in range(4)
        ]
        self.ai_name = "rl_exact"

    # ── core dispatch ────────────────────────────────────────────────
    def choose_action(self, payload: dict[str, Any]) -> AiChoice:
        state, seat = build_local_state(payload)
        player = self.players[seat]
        player.position = seat
        player.hand = list(state.hands[seat])

        view = state.get_player_view(seat)
        view["state"] = state  # the contract RLExactPlayer.play_card expects

        if state.phase == Phase.BIDDING:
            return self._choose_bid(player, view)
        if state.phase == Phase.PLAYING:
            return self._choose_play(player, state, seat, view)
        raise ValueError(f"AI invoked in invalid phase: {state.phase}")

    def _choose_bid(self, player: RLExactPlayer, view: dict[str, Any]) -> AiChoice:
        # Present a normal single-round bid menu (no blind_nil prompt — the GUI
        # has a flat one-shot bidding flow).  RLExactPlayer.place_bid routes
        # through the MLP bid model via the bridge.
        legal_bids = ["nil"] + [numeric_bid_to_str(i) for i in range(1, 14)]
        raw = player.place_bid(legal_bids, view)
        if raw == "nil":
            return AiChoice(kind="bid", value=0, bid_type="nil", detail="mlp_bid")
        if isinstance(raw, str) and raw.startswith("bid_"):
            return AiChoice(kind="bid", value=int(raw.split("_")[1]),
                            bid_type="normal", detail="mlp_bid")
        # Defensive fallback
        return AiChoice(kind="bid", value=1, bid_type="normal", detail="fallback")

    def _choose_play(self, player: RLExactPlayer, state: GameState, seat: int,
                     view: dict[str, Any]) -> AiChoice:
        # Stateless nil-checkpoint switch (mirrors BothPlayer.set_teams).
        if _has_nil_bid(state.max_bid):
            player.policy_nets = [self.policy_nil]
        else:
            player.policy_nets = [self.policy_nonil]
        player.n_policies = 1

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
def build_response_handler(provider: RlExactProvider):
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
            if self.path not in {"/api/choose-action", "/choose-action"}:
                self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                choice = provider.choose_action(payload)
                self._send_json(200, {"ok": True, **choice_to_payload(choice, provider.ai_name)})
            except Exception as exc:  # pragma: no cover - surfaced to browser
                import traceback
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rl_exact AI backend for the Spades GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ai", default="rl_exact",
                        help="Kept for compatibility; only rl_exact is served.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="remaining cards <= this → exact solver (first "
                             "52-threshold cards use the policy net)")
    parser.add_argument("--checkpoint-nonil", type=str,
                        default=str(REPO_ROOT / "55_2.pt"),
                        help="policy for games where no one bids nil")
    parser.add_argument("--checkpoint-nil", type=str,
                        default=str(REPO_ROOT / "55_2nil.pt"),
                        help="policy for games where someone bids nil")
    parser.add_argument("--bid-checkpoint", type=str,
                        default=str(REPO_ROOT / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt"))
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
    provider = RlExactProvider(args)
    server = ThreadingHTTPServer((args.host, args.port), build_response_handler(provider))
    print(f"rl_exact backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"  exact_threshold={provider.exact_threshold} "
          f"(first {52 - provider.exact_threshold} cards use policy net)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
