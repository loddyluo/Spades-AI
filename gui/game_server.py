"""WebSocket game server for Spades: 2 humans (partners) vs 2 AI.

Architecture:
- Each room has a room code + seed. Two human clients connect with matching
  room code and seed, choosing partner seats (0&2 or 1&3).
- Server owns the authoritative GameState and orchestrates all turns.
- AI seats use RuleExactFirst4NilPlayer (imported directly, no HTTP).
- Human clients receive private game views (only their own hand visible).
- Blocking AI calls (_exact_play) run via run_in_executor to not stall the
  async event loop.
- Bidding uses the GO-MCTS MLP bid model (bid_nsfp.pt) via the bridge,
  matching backend.py's behaviour. Blind nil is auto-passed (not offered),
  also matching backend.py.

Usage:
    python gui/game_server.py --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

# ── Import paths ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
for _p in (str(REPO_ROOT), str(GO_MCTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trick_taking.card import Card, Suit, Rank, _STANDARD_CARDS, cards_to_bitset  # noqa: E402
from trick_taking.game_state import GameState, Phase, Bid, TrickRecord  # noqa: E402
from trick_taking.forced_outcome import (  # noqa: E402
    ShowdownResolution,
    apply_showdown_continuation,
    check_for_showdown,
)
from trick_taking.games.spades import SpadesRules  # noqa: E402
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (  # noqa: E402
    ExactDoubleDummyCppFastestSolver,
)
from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer  # noqa: E402
from strategy.hyperparam_config import HyperparamConfig  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────
TRICK_HOLD_SECONDS = 1.5  # pause after trick completes for animation
DEFAULT_CONFIG = REPO_ROOT / "configs" / "8.yaml"
DEFAULT_BID_CKPT = REPO_ROOT / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt"


# ── Model loading ────────────────────────────────────────────────────────

def _load_bid_model(path: str, device: str):
    """Load the GO-MCTS MLP bid model; None → heuristic fallback."""
    cp = Path(path)
    if not cp.exists():
        print(f"  [WARN] bid checkpoint not found: {cp} — bidding falls back to heuristic",
              flush=True)
        return None
    try:
        from models import load_bid_mlp_model  # noqa: E402
        model = load_bid_mlp_model(str(cp.resolve()), device)
        print(f"  [OK] loaded bid model: {cp}", flush=True)
        return model
    except Exception as exc:  # pragma: no cover
        print(f"  [WARN] failed to load bid model {cp}: {exc} — heuristic fallback",
              flush=True)
        return None


# ── Helpers ──────────────────────────────────────────────────────────────

# Frontend-compatible deck order: clubs→diamonds→hearts→spades, 2→A within each
_FRONTEND_SUIT_ORDER = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
_FRONTEND_DECK: list[Card] = [
    Card(s, r)
    for s in _FRONTEND_SUIT_ORDER
    for r in Rank
    if r >= Rank.TWO and r <= Rank.ACE
]


def _deal_hands_frontend_compat(seed: int) -> list[list[Card]]:
    """Deal four 13-card hands using the SAME PRNG and deck order as the
    frontend's dealHands(seed).  This guarantees that the same seed produces
    identical hands in both remote (server) and local (frontend) modes.

    Frontend PRNG: 32-bit xorshift (unsigned ops).
    Frontend deck: [C2..CA, D2..DA, H2..HA, S2..SA].
    Shuffle: Fisher-Yates with Math.floor(rng() * (i+1)).
    """
    cards = list(_FRONTEND_DECK)  # 52 cards in frontend order
    # Replicate the frontend's createRng / shuffle exactly
    value = (seed & 0xFFFFFFFF) or 1
    for i in range(51, 0, -1):  # Fisher-Yates: 51 .. 1
        # xorshift step (all ops on 32-bit unsigned)
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= (value & 0xFFFFFFFF) >> 17  # logical (unsigned) right shift
        value ^= (value << 5) & 0xFFFFFFFF
        value &= 0xFFFFFFFF
        j = int(((value % 1_000_000) / 1_000_000) * (i + 1))
        cards[i], cards[j] = cards[j], cards[i]
    return [cards[s * 13:(s + 1) * 13] for s in range(4)]

def _card_to_str(card: Card) -> str:
    """Serialize a Card to the frontend code string (e.g. 'AS', 'TH').

    Format is rank+suit (frontend convention — rank chars then suit char).
    """
    return f"{card.rank.short}{card.suit.short}"


def _card_from_code(code: str) -> Card:
    """Parse a frontend card code (rank+suit, e.g. 'AS', 'TH') into a Card.

    NOTE: Card.from_str expects suit+rank ('SA'), which is the OPPOSITE of
    what the frontend sends back.  This helper bridges the two conventions.
    """
    if len(code) < 2:
        raise ValueError(f"Invalid card code: {code!r}")
    rank_code = code[:-1]   # all chars except last → rank
    suit_code = code[-1]    # last char → suit
    return Card(suit=Suit.from_short(suit_code), rank=Rank.from_short(rank_code))


def _bid_to_frontend(bid_value: Any) -> dict[str, Any] | None:
    """Convert a Python bid value to the frontend {value, type} format."""
    if bid_value is None:
        return None
    if bid_value == "nil":
        return {"value": 0, "type": "nil"}
    if bid_value == "blind_nil":
        return {"value": 0, "type": "blind_nil"}
    if bid_value == "pass":
        return {"value": 0, "type": "pass"}
    if isinstance(bid_value, str) and bid_value.startswith("bid_"):
        return {"value": int(bid_value.split("_")[1]), "type": "normal"}
    return None


def _bid_from_frontend(frontend_bid: dict[str, Any]) -> str:
    """Convert a frontend {value, type} bid to the Python bid string."""
    bid_type = str(frontend_bid.get("type", "normal")).lower()
    if bid_type == "nil":
        return "nil"
    if bid_type in ("blind_nil", "bnil", "blind-nil"):
        return "blind_nil"
    if bid_type == "pass":
        return "pass"
    value = int(frontend_bid.get("value", 1))
    return f"bid_{value}"


# Display sort order: spades → hearts → clubs → diamonds, high-to-low within suit
_DISPLAY_SUIT_ORDER = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


def _sort_hand_for_display(cards: list[Card]) -> list[Card]:
    """Sort cards for frontend display: spades→hearts→clubs→diamonds, high→low."""
    return sorted(
        cards,
        key=lambda c: (_DISPLAY_SUIT_ORDER.get(c.suit, 99), -c.rank.value),
    )


def _build_client_state(
    gs: GameState, for_seat: int,
    trick_winner: int = -1,
    last_played_seat: int = -1,
    last_bid_seat: int = -1,
    log: list[dict[str, Any]] | None = None,
    showdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe game state dict for a specific human seat.

    The client only sees their own hand; opponents are just hand sizes.
    """
    # Current trick
    current_trick = [
        {"seat": seat, "card": _card_to_str(card)}
        for seat, card in gs.table_cards
    ]

    # Completed tricks
    completed_tricks = []
    for tr in gs.trick_history:
        completed_tricks.append({
            "trickNumber": len(completed_tricks) + 1,
            "winner": tr.winner,
            "cards": [
                {"seat": seat, "card": _card_to_str(card)}
                for seat, card in tr.cards
            ],
        })

    # Bids array (4 elements, null for not-yet-bid seats)
    bids_arr: list[dict[str, Any] | None] = [None, None, None, None]
    for b in gs.bids:
        if not b.is_pass:
            bids_arr[b.player_id] = _bid_to_frontend(b.value)

    is_bidding = gs.phase == Phase.BIDDING
    payload = {
        "type": "game_state",
        "seat": for_seat,
        "phase": "bidding" if is_bidding else
                 "playing" if gs.phase == Phase.PLAYING else "finished",
        "currentPlayer": gs.current_bidder if is_bidding else gs.turn,
        "leader": gs.trick_leader,
        "trickNumber": gs.tricks_played + 1,
        "spadesBroken": gs.spades_broken,
        "hand": [_card_to_str(c) for c in _sort_hand_for_display(gs.hands[for_seat])],
        "handSizes": [len(h) for h in gs.hands],
        "bids": bids_arr,
        "tricksWon": list(gs.tricks_won),
        "currentTrick": current_trick,
        "completedTricks": completed_tricks,
        "trickComplete": len(gs.table_cards) >= gs.num_players,
        "trickWinner": trick_winner,
        "lastPlayedSeat": last_played_seat,
        "lastBidSeat": last_bid_seat,
        "log": list(log) if log else [],
    }
    if showdown is not None:
        payload["showdown"] = showdown
    return payload


# ── Game Room ────────────────────────────────────────────────────────────

class GameRoom:
    """One game room: 2 human connections + 2 AI players."""

    def __init__(self, room_code: str, seed: int,
                 bid_model=None, bid_device: str = "cpu",
                 hyperparam_config=None, exact_solver=None,
                 exact_threshold: int = 36, num_workers: int = 0,
                 showdown_checker=None) -> None:
        self.room_code = room_code
        self.seed = seed
        self.connections: dict[int, Any] = {}   # {seat: websocket}
        self.ai_players: dict[int, RuleExactFirst4NilPlayer] = {}
        self.state: GameState | None = None
        self.rules = SpadesRules()
        # Shared AI components (loaded once at startup)
        self._bid_model = bid_model
        self._bid_device = bid_device
        self._hyperparam_config = hyperparam_config
        self._exact_solver = exact_solver
        self._exact_threshold = exact_threshold
        self._num_workers = num_workers
        # Human action handling
        self._action: dict[str, Any] | None = None
        self._action_event: asyncio.Event = asyncio.Event()
        self._expected_action_seat: int | None = None
        self._expected_action_type: str | None = None
        self._game_started = False
        # A separate confirmation barrier prevents a bid/play message from
        # ever being interpreted as consent to settle a revealed hand.
        self._showdown_checker = showdown_checker or check_for_showdown
        self.showdown_id = 0
        self.showdown_resolution: ShowdownResolution | None = None
        self.showdown_confirmations: set[int] = set()
        self.showdown_pending = False
        self._showdown_human_seats: frozenset[int] = frozenset()
        self._showdown_event: asyncio.Event = asyncio.Event()
        # Animation tracking (for frontend UI hints)
        self._last_trick_winner: int = -1
        self._last_played_seat: int = -1
        self._last_bid_seat: int = -1
        # Game event log (for frontend mini-log display)
        self._log: list[dict[str, Any]] = []
        self._seat_names = ["North", "East", "South", "West"]

    @property
    def ai_seats(self) -> list[int]:
        return [s for s in range(4) if s not in self.connections]

    def partner_of(self, seat: int) -> int:
        return (seat + 2) % 4

    def is_ready(self) -> bool:
        return len(self.connections) == 2

    # ── Connection management ────────────────────────────────────────

    def add_connection(self, seat: int, ws) -> str | None:
        """Try to add a human connection. Returns error message or None."""
        if seat in self.connections:
            return f"座位 {seat} 已被占用"
        if self._game_started:
            return "游戏已开始"
        self.connections[seat] = ws
        return None

    def remove_connection(self, seat: int) -> None:
        self.connections.pop(seat, None)
        if self.showdown_pending:
            self._showdown_event.set()
        if self._expected_action_seat == seat:
            self._action_event.set()

    def all_connected(self) -> bool:
        return len(self.connections) == 2 and not self._game_started

    # ── Game lifecycle ───────────────────────────────────────────────

    async def start_game(self) -> None:
        """Deal cards, create AI players, run bidding then playing phases."""
        self._game_started = True

        # ── Deal (frontend-compatible PRNG → same hands as local mode) ─
        hands = _deal_hands_frontend_compat(self.seed)

        self.state = GameState()
        self.state.init_for_deal(4, hands, [], _FRONTEND_DECK)
        self.state.trump_suit = Suit.SPADES
        self.state.teams = [0, 1, 0, 1]

        # First bidder / leader: fixed to seat 0 (North).
        dealer = 3       # West
        opener = 0       # North = dealer's left
        self.state.dealer_seat = dealer
        self.state.turn = opener
        self.state.current_bidder = opener
        self.state.trick_leader = opener

        # ── Create AI players ───────────────────────────────────────
        for seat in self.ai_seats:
            player = RuleExactFirst4NilPlayer(
                exact_solver=self._exact_solver,
                exact_threshold=self._exact_threshold,
                bid_model=self._bid_model,
                bid_device=self._bid_device,
                hyperparam_config=self._hyperparam_config,
                num_workers=self._num_workers,
            )
            player.start_game(seat, list(hands[seat]), 4)
            self.ai_players[seat] = player

        # ── Send initial connection info + game state to both humans ─
        self._log.append({"kind": "system", "text": "新牌局已开始"})
        for seat, ws in self.connections.items():
            opponent = self.partner_of(seat)
            await self._safe_send(ws, {
                "type": "opponent_joined",
                "opponentSeat": opponent,
                "yourSeat": seat,
            })
        # CRITICAL: broadcast initial state so humans see correct hands
        # BEFORE they are asked to bid. The frontend placeholder state
        # uses a different PRNG so its hands are wrong.
        self.state.phase = Phase.BIDDING  # set phase before broadcast
        await self._broadcast_state()

        # ── Run phases ──────────────────────────────────────────────
        try:
            await self._bidding_phase()
            await self._playing_phase()
            await self._broadcast_game_over()
        except Exception as exc:
            traceback.print_exc()
            for ws in self.connections.values():
                await self._safe_send(ws, {
                    "type": "error",
                    "message": f"服务器错误: {exc}",
                })

    # ── Bidding phase ───────────────────────────────────────────────

    async def _bidding_phase(self) -> None:
        self.state.phase = Phase.BIDDING

        while not self.rules.end_bidding(self.state):
            bidder = self.state.current_bidder
            legal = self.rules.legal_bids(self.state, bidder)

            # Auto-pass blind_nil for all players — matches backend.py
            # behaviour which never presents blind_nil as an option.
            if legal == ["blind_nil", "pass"]:
                bid_value = "pass"
            elif not legal:
                # Should not happen in Spades, but guard
                bid_value = "pass"
            elif bidder in self.connections:
                # ── Human turn ───────────────────────────────────
                bid_value = await self._wait_for_human_bid(bidder, legal)
            else:
                # ── AI turn ─────────────────────────────────────
                view = self.state.get_player_view(bidder)
                view["state"] = self.state  # required by place_bid MLP path
                ai = self.ai_players[bidder]
                bid_value = await asyncio.get_event_loop().run_in_executor(
                    None, ai.place_bid, legal, view
                )

            # Record bid
            is_pass = (bid_value == "pass")
            bid_record = Bid(player_id=bidder, value=bid_value, is_pass=is_pass)
            self.state.bids.append(bid_record)
            if not is_pass:
                self.state.max_bid[bidder] = bid_value
                # Log non-pass bids (pass is just blind_nil skip, invisible to user)
                fb = _bid_to_frontend(bid_value)
                label = "Nil" if (fb and fb.get("type") == "nil") else str(fb.get("value", "")) if fb else "—"
                self._log.append({
                    "kind": "bid", "seat": bidder,
                    "text": f"{self._seat_names[bidder]} 叫牌 {label}",
                })
            self._last_bid_seat = bidder

            # Notify AI players
            for seat, ai in self.ai_players.items():
                ai.bid_placed(bidder, bid_value)

            # Next bidder
            self.state.current_bidder = self.rules.next_bid_turn(self.state)

            # Broadcast updated state
            await self._broadcast_state()

        # Set teams for AI players (must happen before playing phase)
        bid_values = [b.value for b in self.state.bids if not b.is_pass]
        all_bids = [
            next((b.value for b in self.state.bids
                  if b.player_id == pid and not b.is_pass), None)
            for pid in range(4)
        ]
        for seat, ai in self.ai_players.items():
            ai.set_teams(self.state.teams, all_bids)

        self._log.append({"kind": "system", "text": "叫牌完成，进入出牌阶段"})

    async def _wait_for_human_bid(self, seat: int, legal: list[Any]) -> str:
        """Send your_turn to the human, send waiting to partner, wait for action."""
        ws = self.connections[seat]
        partner = self.partner_of(seat)

        legal_frontend = []
        for b in legal:
            fb = _bid_to_frontend(b)
            if fb:
                legal_frontend.append(fb)

        self._prepare_action_wait(seat, "bid")
        await self._safe_send(ws, {
            "type": "your_turn",
            "phase": "bidding",
            "legalBids": legal_frontend,
        })
        if partner in self.connections:
            await self._safe_send(self.connections[partner], {
                "type": "waiting",
                "message": "等待搭档叫牌...",
            })

        action = await self._wait_for_action()
        frontend_bid = action.get("bid", {})
        bid_value = _bid_from_frontend(frontend_bid)

        # Validate legality
        if bid_value not in legal:
            bid_value = legal[0] if legal else "bid_1"

        return bid_value

    # ── Playing phase ───────────────────────────────────────────────

    async def _playing_phase(self) -> None:
        self.state.phase = Phase.PLAYING
        # CRITICAL: broadcast phase change BEFORE first turn so the
        # frontend knows weʼre in playing mode.  Otherwise a human who
        # acts first would receive your_turn while game.phase is still
        # "bidding" and the UI shows no playable cards.
        await self._broadcast_state()

        while not self.rules.end_trickgame(self.state):
            self.state.table_cards = []
            self._last_trick_winner = -1

            for _ in range(4):
                current = self.state.turn
                hand = self.state.hands[current]
                legal_cards = self.rules.playable(self.state, hand, current)

                if not legal_cards:
                    raise RuntimeError(
                        f"Player {current} has no legal cards"
                    )

                if current in self.connections:
                    # ── Human turn ───────────────────────────────
                    card = await self._wait_for_human_play(current, legal_cards)
                else:
                    # ── AI turn ─────────────────────────────────
                    card = await self._run_ai_play(current, legal_cards)

                # Apply play
                self.state.play_card_to_table(current, card)
                if card.suit == Suit.SPADES:
                    self.state.trump_broken = True
                    self.state.spades_broken = True

                # Log played card
                self._log.append({
                    "kind": "play", "seat": current,
                    "text": f"{self._seat_names[current]} 出牌 {_card_to_str(card)}",
                })

                # Notify AI players
                for seat, ai in self.ai_players.items():
                    ai.card_played(current, card)

                # Track last played seat for frontend animation
                self._last_played_seat = current

                # If trick just completed, pre-compute winner so the
                # client sees trickWinner alongside trickComplete=true
                if len(self.state.table_cards) >= self.state.num_players:
                    self._last_trick_winner = self.rules.winner_trick(self.state)

                # Next player
                self.state.turn = (current + 1) % 4

                # Broadcast state after each card
                await self._broadcast_state()

            # Trick complete — finalize
            winner = self._last_trick_winner
            self.state.complete_trick(winner)
            self.state.turn = winner
            self.state.trick_leader = winner

            # Log trick winner
            trick_no = self.state.tricks_played
            self._log.append({
                "kind": "system",
                "text": f"第 {trick_no} 墩由 {self._seat_names[winner]} 赢下",
            })

            # Broadcast cleared table, then brief pause for animation
            await self._broadcast_state()
            if not self.rules.end_trickgame(self.state):
                if await self._maybe_offer_showdown():
                    self._log.append({"kind": "system", "text": "牌局结束"})
                    return
                await asyncio.sleep(TRICK_HOLD_SECONDS)

        self.state.phase = Phase.SCORING
        self._log.append({"kind": "system", "text": "牌局结束"})

    async def _wait_for_human_play(
        self, seat: int, legal_cards: list[Card]
    ) -> Card:
        """Send your_turn to the human, wait for their card choice."""
        ws = self.connections[seat]
        partner = self.partner_of(seat)

        legal_codes = [_card_to_str(c) for c in legal_cards]
        self._prepare_action_wait(seat, "play")
        await self._safe_send(ws, {
            "type": "your_turn",
            "phase": "playing",
            "legalCards": legal_codes,
        })
        if partner in self.connections:
            await self._safe_send(self.connections[partner], {
                "type": "waiting",
                "message": "等待搭档出牌...",
            })

        action = await self._wait_for_action()
        card_code = str(action.get("card", ""))
        try:
            card = _card_from_code(card_code)  # frontend sends rank+suit format
        except Exception:
            card = legal_cards[0]

        if card not in legal_cards:
            card = legal_cards[0]

        return card

    async def _run_ai_play(
        self, seat: int, legal_cards: list[Card]
    ) -> Card:
        """Run AI play_card in executor (may block for exact solver)."""
        ai = self.ai_players[seat]
        view = self.state.get_player_view(seat)
        view["state"] = self.state  # required by RuleExactFirst4NilPlayer

        loop = asyncio.get_event_loop()
        card = await loop.run_in_executor(
            None, ai.play_card, legal_cards, view
        )
        if card not in legal_cards:
            card = legal_cards[0]
        return card

    # ── State broadcast ─────────────────────────────────────────────

    async def _broadcast_state(self) -> None:
        """Send current game state to both human clients."""
        # Snapshot once so both clients receive identical confirmation data
        # even if the second confirmation arrives between socket writes.
        showdown = self._showdown_payload()
        for seat, ws in list(self.connections.items()):
            state_msg = _build_client_state(
                self.state, seat,
                trick_winner=self._last_trick_winner,
                last_played_seat=self._last_played_seat,
                last_bid_seat=self._last_bid_seat,
                log=self._log,
                showdown=showdown,
            )
            await self._safe_send(ws, state_msg)

    def _showdown_payload(self) -> dict[str, Any] | None:
        if not self.showdown_pending or self.showdown_resolution is None:
            return None
        return {
            "id": self.showdown_id,
            "revealedHands": [
                [_card_to_str(card) for card in _sort_hand_for_display(hand)]
                for hand in self.state.hands
            ],
            "teamTricks": list(self.showdown_resolution.team_tricks),
            "nilOutcomes": list(self.showdown_resolution.nil_outcomes),
            "confirmedSeats": sorted(self.showdown_confirmations),
        }

    async def _maybe_offer_showdown(self) -> bool:
        """Run the exact check and pause the room if it proves a fixed result."""
        if self.showdown_pending or self.state is None:
            return False
        if self.state.phase != Phase.PLAYING or self.state.table_cards:
            return False
        sizes = [len(hand) for hand in self.state.hands]
        if len(sizes) != 4 or len(set(sizes)) != 1 or not 1 <= sizes[0] <= 5:
            return False

        call = functools.partial(
            self._showdown_checker,
            copy.deepcopy(self.state),
            self._exact_solver,
            time_budget_seconds=1.0,
        )
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, call)
        except Exception:
            # An operational failure must never reveal cards or stop play.
            traceback.print_exc()
            return False
        if result.status != "fixed" or result.resolution is None:
            return False
        await self._offer_showdown(result.resolution)
        return True

    async def _offer_showdown(self, resolution: ShowdownResolution) -> None:
        """Reveal, collect two distinct confirmations, and then settle."""
        human_seats = frozenset(self.connections)
        if len(human_seats) != 2:
            raise RuntimeError("a human disconnected before showdown confirmation")

        self.showdown_id += 1
        self.showdown_resolution = resolution
        self.showdown_confirmations.clear()
        self._showdown_human_seats = human_seats
        self.showdown_pending = True
        self._showdown_event.clear()
        self._expected_action_seat = None
        self._expected_action_type = None
        self._action = None
        self._action_event.clear()
        await self._broadcast_state()

        try:
            broadcasted_confirmations: frozenset[int] = frozenset()
            while True:
                if frozenset(self.connections) != human_seats:
                    raise RuntimeError("a human disconnected during showdown confirmation")
                current_confirmations = frozenset(self.showdown_confirmations)
                if current_confirmations != broadcasted_confirmations:
                    await self._broadcast_state()
                    broadcasted_confirmations = current_confirmations
                    continue
                if human_seats.issubset(current_confirmations):
                    break
                self._showdown_event.clear()
                # No await occurred since clear; recheck before sleeping so a
                # confirmation already recorded in the set cannot be lost.
                if frozenset(self.showdown_confirmations) != broadcasted_confirmations:
                    continue
                await self._showdown_event.wait()

            self.state = apply_showdown_continuation(
                self.state,
                resolution.continuation,
            )
            self.state.phase = Phase.SCORING
            self._log.append({
                "kind": "system",
                "text": "双方已确认自动摊牌，按固定结果结算",
            })
            self.showdown_pending = False
            await self._broadcast_state()
        except Exception:
            self.showdown_pending = False
            raise

    async def _broadcast_game_over(self) -> None:
        """Send final scores to both humans."""
        rules = self.rules
        payoffs = rules.score(self.state)

        # Compute team scores (NS = team 0, EW = team 1)
        team_scores = [0.0, 0.0]
        for pid in range(4):
            team_scores[self.state.teams[pid]] += payoffs[pid]

        score_msg = {
            "type": "hand_over",
            "score": {
                "northSouth": team_scores[0],
                "eastWest": team_scores[1],
            },
            "tricksWon": list(self.state.tricks_won),
            "bids": [_bid_to_frontend(self.state.max_bid[p]) for p in range(4)],
        }
        for ws in self.connections.values():
            await self._safe_send(ws, score_msg)

    # ── Human action wait ───────────────────────────────────────────

    def _prepare_action_wait(self, seat: int, action_type: str) -> None:
        self._action_event.clear()
        self._action = None
        self._expected_action_seat = seat
        self._expected_action_type = action_type

    async def _wait_for_action(self) -> dict[str, Any]:
        """Wait for the current human to send a bid/play action."""
        try:
            await self._action_event.wait()
            if self._action is None:
                raise RuntimeError("the expected human disconnected")
            return self._action
        finally:
            self._expected_action_seat = None
            self._expected_action_type = None

    def receive_action(self, sender_seat: int, action: dict[str, Any]) -> bool:
        """Called from the WebSocket handler when a human sends an action."""
        action_type = str(action.get("type", ""))
        if self.showdown_pending:
            if action_type != "showdown_confirm":
                return False
            if sender_seat not in self._showdown_human_seats:
                return False
            if sender_seat not in self.connections:
                return False
            if action.get("showdownId") != self.showdown_id:
                return False
            if sender_seat in self.showdown_confirmations:
                return False
            self.showdown_confirmations.add(sender_seat)
            self._showdown_event.set()
            return True

        if action_type == "showdown_confirm":
            return False
        if sender_seat != self._expected_action_seat:
            return False
        if action_type != self._expected_action_type:
            return False
        if self._action is not None:
            return False
        self._action = action
        self._action_event.set()
        return True

    # ── Safe send ───────────────────────────────────────────────────

    @staticmethod
    async def _safe_send(ws, msg: dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass  # Client disconnected


# ── Global state ─────────────────────────────────────────────────────────

rooms: dict[str, GameRoom] = {}

# Shared AI components (loaded once at startup, shared read-only across rooms)
_shared_bid_model = None
_shared_bid_device = "cpu"
_shared_hyperparam_config = None
_shared_exact_solver = None
_shared_exact_threshold = 36
_shared_num_workers = 0


def _validate_partners(seat1: int, seat2: int) -> bool:
    """Check that two seats are partners (same team, different seats)."""
    return (seat1 % 2) == (seat2 % 2) and seat1 != seat2


# ── WebSocket handler ────────────────────────────────────────────────────

async def handler(ws) -> None:
    """Handle one WebSocket client connection."""
    my_room: GameRoom | None = None
    my_seat: int = -1

    try:
        async for raw_message in ws:
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                await GameRoom._safe_send(ws, {
                    "type": "error", "message": "无效的 JSON"
                })
                continue

            msg_type = str(msg.get("type", ""))

            if msg_type == "join":
                # ── Join room ────────────────────────────────────
                room_code = str(msg.get("room", "")).strip().upper()
                seed_str = str(msg.get("seed", ""))
                seat = int(msg.get("seat", -1))

                if not room_code:
                    await GameRoom._safe_send(ws, {
                        "type": "error", "message": "请输入房间号"
                    })
                    continue
                if seat not in (0, 1, 2, 3):
                    await GameRoom._safe_send(ws, {
                        "type": "error", "message": "座位必须是 0-3"
                    })
                    continue
                try:
                    seed = int(seed_str)
                    if seed < 0:
                        raise ValueError
                except (ValueError, TypeError):
                    await GameRoom._safe_send(ws, {
                        "type": "error", "message": "种子必须是非负整数"
                    })
                    continue

                # Get or create room
                if room_code not in rooms:
                    rooms[room_code] = GameRoom(
                        room_code, seed,
                        bid_model=_shared_bid_model,
                        bid_device=_shared_bid_device,
                        hyperparam_config=_shared_hyperparam_config,
                        exact_solver=_shared_exact_solver,
                        exact_threshold=_shared_exact_threshold,
                        num_workers=_shared_num_workers,
                    )
                room = rooms[room_code]

                # Validate seed match
                if room.seed != seed:
                    await GameRoom._safe_send(ws, {
                        "type": "error",
                        "message": f"种子不匹配：房间种子为 {room.seed}，你输入的是 {seed}"
                    })
                    continue

                # Validate partnership
                existing_seats = list(room.connections.keys())
                if existing_seats:
                    if not _validate_partners(existing_seats[0], seat):
                        partner_seat = (existing_seats[0] + 2) % 4
                        await GameRoom._safe_send(ws, {
                            "type": "error",
                            "message": (
                                f"座位 {seat} 与已有玩家 (座位 {existing_seats[0]}) "
                                f"不是对家。请选择座位 {partner_seat}"
                            )
                        })
                        continue

                # Add connection
                err = room.add_connection(seat, ws)
                if err:
                    await GameRoom._safe_send(ws, {
                        "type": "error", "message": err
                    })
                    continue

                my_room = room
                my_seat = seat

                await GameRoom._safe_send(ws, {
                    "type": "joined",
                    "room": room_code,
                    "seat": seat,
                    "seed": seed,
                })

                # If both connected, start game
                if room.is_ready():
                    asyncio.create_task(room.start_game())

            elif msg_type in ("bid", "play", "showdown_confirm"):
                # ── Human action ─────────────────────────────────
                if my_room is None:
                    await GameRoom._safe_send(ws, {
                        "type": "error", "message": "请先加入房间"
                    })
                    continue
                my_room.receive_action(my_seat, msg)

            else:
                await GameRoom._safe_send(ws, {
                    "type": "error", "message": f"未知消息类型: {msg_type}"
                })

    except Exception:
        pass  # Client disconnected
    finally:
        # Cleanup
        if my_room is not None and my_seat >= 0:
            my_room.remove_connection(my_seat)
            # If room is empty, remove it
            if not my_room.connections:
                rooms.pop(my_room.room_code, None)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spades WebSocket game server (2 humans vs 2 AI)"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                        help="WebSocket port (default: 8765)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for MLP bid model (default: cpu)")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="Remaining cards <= this → exact solver (default: 36)")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_CONFIG),
                        help="Path to hyperparam YAML config")
    parser.add_argument("--bid-checkpoint", type=str,
                        default=str(DEFAULT_BID_CKPT),
                        help="Path to bid MLP checkpoint (bid_nsfp.pt)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of parallel solver workers (0=auto, 1=sequential)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible dealing (default: random)")
    return parser.parse_args()


async def main() -> None:
    global _shared_bid_model, _shared_bid_device, _shared_hyperparam_config
    global _shared_exact_solver, _shared_exact_threshold, _shared_num_workers

    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(args.seed)
        except Exception:
            pass

    print("Spades game server starting")
    print(f"  Mode: 2 humans (partners) vs 2 AI (RuleExactFirst4NilPlayer)")
    print(f"  Humans: connect with matching room code + seed + partner seats")
    print()

    # ── Load shared AI components ──────────────────────────────────────
    print("Loading AI models ...", flush=True)

    _shared_bid_device = args.device
    _shared_exact_threshold = args.exact_threshold
    _shared_num_workers = args.num_workers

    # Hyperparam config
    config_path = Path(args.config)
    if config_path.exists():
        _shared_hyperparam_config = HyperparamConfig.from_yaml(str(config_path))
        print(f"  [OK] loaded config: {config_path}", flush=True)
    else:
        print(f"  [WARN] config not found: {config_path} — using defaults",
              flush=True)
        _shared_hyperparam_config = HyperparamConfig()

    # Bid model
    _shared_bid_model = _load_bid_model(args.bid_checkpoint, args.device)

    # Exact solver (shared read-only across rooms)
    _shared_exact_solver = ExactDoubleDummyCppFastestSolver()
    print(f"  [OK] exact solver ready", flush=True)

    print(f"  exact_threshold={_shared_exact_threshold} "
          f"(first {52 - _shared_exact_threshold} cards use rule-based / nil policy)",
          flush=True)
    print()

    import websockets
    async with websockets.serve(handler, args.host, args.port):
        print(f"Listening on ws://{args.host}:{args.port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
