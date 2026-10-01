"""Local Python AI backend for the Spades GUI — MLP first four + exact play.

The frontend (gui/src/game.js) sends ONLY public history plus the current
player's remaining hand.  This backend reconstructs a partial
`trick_taking.game_state.GameState` and drives the production play pipeline:

- First 4 tricks (remaining > exact_threshold): the deployed non-Nil MLP or
  one of four role-specific Nil MLPs, all blind to opponents' hands.
- Last 36 cards (remaining <= exact_threshold): the exact double-dummy
  solver with importance-sampling determinization.  It RECONSTRUCTS the
  opponents' hidden hands from public history — it never peeks at the
  human's real cards.
- Multi-Nil deals use the same deterministic role mapping as single-Nil play.

Bidding uses the selected residual-Q 100k acting bidder.  The original
bid_nsfp.pt remains frozen inside the late-play importance-sampling belief
model and as the acting bidder's fail-safe fallback.

Hyperparameter config defaults to configs/8.yaml.

The HTTP layer is stateless: every request rebuilds the GameState from the
posted payload and replays the full public trick history into the exact-stage
tracker, so there is no cross-request memory to keep in sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# ── Import paths ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
for _p in (str(REPO_ROOT), str(GO_MCTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gui.regret_jobs as regret_jobs  # noqa: E402

from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS, cards_to_bitset  # noqa: E402
from trick_taking.game_state import Bid, GameState, Phase, TrickRecord  # noqa: E402
from trick_taking.forced_outcome import (  # noqa: E402
    ShowdownStateError,
    check_for_showdown,
    validate_showdown_state,
)
from trick_taking.games.spades import SpadesRules  # noqa: E402
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (  # noqa: E402
    ExactDoubleDummyCppFastestSolver,
)

from rl.nil_solver_leaf_deployment import (  # noqa: E402
    DEFAULT_NIL_ACTOR_BUNDLE_PATH,
    DEFAULT_NIL_ACTOR_BUNDLE_SHA256,
    load_deployed_nil_actor_bundle,
)
from rl.solver_leaf_deployment import (  # noqa: E402
    DEFAULT_SOLVER_LEAF_ACTOR_PATH,
    DEFAULT_SOLVER_LEAF_ACTOR_SHA256,
    DEFAULT_SOLVER_LEAF_SIDECAR_SHA256,
    load_deployed_solver_leaf_actor,
)
from strategy.solver_leaf_mlp_exact_player import (  # noqa: E402
    SolverLeafMLPExactPlayer,
)
from strategy.rule_exact_first4_player import warm_up_exact_stage  # noqa: E402
from strategy.hyperparam_config import HyperparamConfig  # noqa: E402
from residual_bidder.actions import to_local_bid  # noqa: E402
from residual_bidder.deployment import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH as DEFAULT_ACTING_BID_CHECKPOINT,
    DEFAULT_CONFIG_PATH as DEFAULT_RESIDUAL_BIDDER_CONFIG,
    load_deployed_acting_bidder,
)


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

    # bids → max_bid[4] for the frozen NSFP encoder plus ordered public
    # history for the deployed bidder's deterministic sampling key.
    raw_bids = payload.get("bids", []) or []
    max_bid: list[Any] = [None, None, None, None]
    bid_history: list[Bid] = []
    for i in range(4):
        entry = raw_bids[i] if i < len(raw_bids) else None
        max_bid[i] = frontend_bid_to_local(entry)
    opener = int(payload.get("firstSeat", 0)) % 4
    for offset in range(4):
        bidder = (opener + offset) % 4
        if max_bid[bidder] is not None:
            bid_history.append(
                Bid(player_id=bidder, value=max_bid[bidder], is_pass=False)
            )

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
    state.bids = bid_history
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


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file, or None when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def describe_play_config(provider: "RuleExactProvider") -> dict[str, Any]:
    """Report exactly which hyperparameters produced a play decision.

    Provenance matters here: ``multiplier_clip``/``multiplier_clip_factor``
    change how the exact stage ranks actions, so a regret number is only
    meaningful next to the config that generated it.
    """
    return {
        "path": str(provider.config_path),
        "sha256": provider.config_sha256,
        "effective": asdict(provider.hyperparam_config),
    }


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
    """Holds shared models and one MLP/exact player per seat.

    Stateless across HTTP requests: each request rebuilds the GameState and
    replays the full public trick history before choosing an action.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        device = args.device
        self.device = device
        self.exact_threshold = int(args.exact_threshold)
        self.seed = args.seed

        print("Loading solver-leaf MLP + exact models ...", flush=True)

        # Load hyperparam config
        self.config_path = Path(args.config)
        self.hyperparam_config = HyperparamConfig.from_yaml(args.config)
        self.config_sha256 = sha256_file(self.config_path)
        print(
            f"  [OK] loaded config: {args.config} "
            f"(sha256={self.config_sha256})",
            flush=True,
        )
        print(
            "       multiplier_clip="
            f"{self.hyperparam_config.multiplier_clip} x "
            f"{self.hyperparam_config.multiplier_clip_factor}"
            + (
                " (no-op)"
                if self.hyperparam_config.multiplier_clip_factor == 1.0
                else " (ACTIVE: tail losses are rescaled)"
            ),
            flush=True,
        )

        self.acting_bidder = load_deployed_acting_bidder(
            checkpoint_path=Path(args.acting_bid_checkpoint),
            config_path=Path(args.residual_bidder_config),
            repo_root=REPO_ROOT,
            device=device,
            policy_seed=args.bid_policy_seed,
        )
        print(
            "  [OK] loaded acting bidder: "
            f"model_id={self.acting_bidder.model_id}",
            flush=True,
        )

        self.nonnil_play_actor = load_deployed_solver_leaf_actor(
            Path(args.nonnil_actor_checkpoint),
            expected_sha256=args.nonnil_actor_sha256,
            expected_sidecar_sha256=args.nonnil_actor_sidecar_sha256,
            device=device,
        )
        print(
            "  [OK] loaded non-Nil play actor: "
            f"model_id={self.nonnil_play_actor.model_id}",
            flush=True,
        )
        self.nil_play_actors = load_deployed_nil_actor_bundle(
            Path(args.nil_actor_bundle),
            expected_sha256=args.nil_actor_bundle_sha256,
            device=device,
        )
        print(
            "  [OK] loaded four-role Nil play actors: "
            f"model_id={self.nil_play_actors.model_id}",
            flush=True,
        )

        # The wrapper serializes entry to the native process-global caches, so
        # this instance can be shared safely across seats.
        self.exact_solver = ExactDoubleDummyCppFastestSolver()
        self.rules = SpadesRules()
        # RuleExact players keep mutable replay history.  The HTTP server is
        # threaded, so reset/replay/action selection must be one transaction.
        self._decision_lock = threading.Lock()

        # One stateful player per seat. All first-four choices and posterior
        # replay choices use MLPs; remaining <= threshold uses exact search.
        self.players: list[SolverLeafMLPExactPlayer] = [
            SolverLeafMLPExactPlayer(
                nonnil_actor=self.nonnil_play_actor.actor,
                nonnil_model_id=self.nonnil_play_actor.model_id,
                nonnil_actor_sha256=self.nonnil_play_actor.sha256,
                nil_actors=self.nil_play_actors.actors,
                nil_model_id=self.nil_play_actors.model_id,
                nil_bundle_sha256=self.nil_play_actors.sha256,
                exact_solver=self.exact_solver,
                exact_threshold=self.exact_threshold,
                # Acting bids are handled by self.acting_bidder.  The card
                # player still lazy-loads bid_nsfp.pt for IS belief weighting.
                bid_model=None,
                bid_device=device,
                hyperparam_config=self.hyperparam_config,
                num_workers=args.num_workers,
            )
            for _ in range(4)
        ]
        self.ai_name = "solver_leaf_mlp_exact_residual_q_100k"

        # Both of these are lazy in the normal path, and both are expensive the
        # first time: the belief MLP has to be loaded and the solver pool has to
        # spawn ten interpreters.  Without this, that bill lands inside the
        # player's turn on the first card of trick five (~5s); here it lands at
        # boot, where nothing is waiting.  All four seats are primed because
        # otherwise each one pays it on its own first exact-stage turn.
        # Failures are non-fatal.
        try:
            for player in self.players:
                player._ensure_bid_model_loaded()
        except Exception as error:  # pragma: no cover - diagnostics only
            print(f"  [WARN] belief model preload failed: {error}", flush=True)
        print("  warming solver pool ...", flush=True)
        warm_up_exact_stage(self.players[0]._num_workers)

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

    def analyze_play_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Replay one decision and return the pipeline's per-action Q table.

        This is the analysis twin of the play branch of
        :meth:`_choose_action_serialized`: the same ``build_local_state``
        reconstruction (own hand real, opponents' counts right, identities
        never read), the same public-history replay, the same per-seat player
        and therefore the same observer perspective (`state.turn` / `position`
        decide which team's Q the exact stage optimizes).

        The only difference is ``collect_action_q``: the exact stage then also
        reports the *uncropped* expected Q it aggregated for each legal action,
        which the live path throws away in favour of the clipped multiplier
        sum.  Nothing about the chosen card changes.
        """
        with self._decision_lock:
            state, seat = build_local_state(payload)
            if state.phase != Phase.PLAYING:
                raise ValueError(
                    f"regret analysis only covers card play, got phase {state.phase}"
                )
            player = self.players[seat]

            # Reconstruct the AI's original hand so inherited exact-stage public
            # replay has the same initial hand identity as the live request.
            ai_played: list[Card] = []
            for trick in state.trick_history:
                for pid, card in trick.cards:
                    if pid == seat:
                        ai_played.append(card)
            for pid, card in state.table_cards:
                if pid == seat:
                    ai_played.append(card)
            original_hand = list(state.hands[seat]) + ai_played

            player.start_game(seat, original_hand, 4)
            player.set_teams(state.teams, state.max_bid)
            for trick in state.trick_history:
                for pid, card in trick.cards:
                    player.card_played(pid, card)
            for pid, card in state.table_cards:
                player.card_played(pid, card)

            legal_cards = self.rules.playable(state, state.hands[seat], seat)
            if not legal_cards:
                raise ValueError(f"seat {seat} has no legal cards to play")

            view = state.get_player_view(seat)
            view["state"] = state

            was_collecting = player.collect_action_q
            player.collect_action_q = True
            try:
                card = player.play_card(legal_cards, view)
            finally:
                player.collect_action_q = was_collecting

            info = (
                dict(player.last_play_info)
                if isinstance(player.last_play_info, dict)
                else {}
            )
            return {
                "seat": seat,
                "chosenCard": card_to_code(card),
                "legalCards": [card_to_code(c) for c in legal_cards],
                "info": info,
            }

    def _choose_action_serialized(self, payload: dict[str, Any]) -> AiChoice:
        state, seat = build_local_state(payload)
        player = self.players[seat]

        # Reconstruct the AI's original hand so inherited exact-stage public
        # replay has the same initial hand identity as the authoritative game.
        ai_played: list[Card] = []
        for trick in state.trick_history:
            for pid, card in trick.cards:
                if pid == seat:
                    ai_played.append(card)
        for pid, card in state.table_cards:
            if pid == seat:
                ai_played.append(card)
        original_hand = list(state.hands[seat]) + ai_played

        # Reset the stateful exact-stage tracker and replay public history.
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
            return self._choose_bid(state, seat, payload)
        if state.phase == Phase.PLAYING:
            return self._choose_play(player, state, seat, view)
        raise ValueError(f"AI invoked in invalid phase: {state.phase}")

    def _choose_bid(
        self,
        state: GameState,
        seat: int,
        payload: dict[str, Any],
    ) -> AiChoice:
        # The local GUI has a flat one-shot bidding flow, so blind nil is not
        # offered.  Only this acting path uses residual Q; card play is unchanged.
        legal_bids = ["nil"] + [numeric_bid_to_str(i) for i in range(1, 14)]
        raw_seed = payload.get("seed")
        deal_id = f"local:{raw_seed}" if isinstance(raw_seed, int) else "local:unseeded"
        decision = self.acting_bidder.choose(
            state,
            legal_bids,
            logical_seat=seat,
            deal_id=deal_id,
            room_id="http-local",
        )
        raw = to_local_bid(decision.action)
        self.players[seat].last_bid_info = {
            "chosen_bid": raw,
            "policy_id": decision.effective_policy_id,
            "fallback_reason": decision.fallback_reason,
            "legal_bids": list(legal_bids),
        }
        if decision.fallback_reason is not None:
            raise RuntimeError(
                f"AI bidding triggered fallback: {decision.fallback_reason}"
            )
        if raw == "nil":
            return AiChoice(
                kind="bid", value=0, bid_type="nil", detail="residual_bid"
            )
        if isinstance(raw, str) and raw.startswith("bid_"):
            return AiChoice(kind="bid", value=int(raw.split("_")[1]),
                            bid_type="normal", detail="residual_bid")
        raise ValueError(f"deployed acting bidder returned invalid bid {raw!r}")

    def _choose_play(self, player: SolverLeafMLPExactPlayer, state: GameState, seat: int,
                     view: dict[str, Any]) -> AiChoice:
        # SolverLeafMLPExactPlayer internally routes every first-four decision
        # to a deployed MLP and all later decisions to the exact solver.

        legal_cards = self.rules.playable(state, state.hands[seat], seat)
        if not legal_cards:
            raise ValueError(f"seat {seat} has no legal cards to play")

        card = player.play_card(legal_cards, view)
        mode = ""
        fallback_reason = None
        if isinstance(player.last_play_info, dict):
            mode = str(player.last_play_info.get("mode", ""))
            fallback_reason = player.last_play_info.get("fallback_reason")
        if fallback_reason is not None or "fallback" in mode.lower():
            reason = fallback_reason or mode
            raise RuntimeError(f"AI card play triggered fallback: {reason}")
        if card not in legal_cards:
            raise RuntimeError(
                "AI card play returned an illegal card; fallback is disabled"
            )
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
# 「完整复盘」分析作业
# ────────────────────────────────────────────────────────────────────────
# 分析子进程必须用与父进程完全相同的 CLI 参数重建同一条 pipeline，
# 因此 main() 启动时把命令行原文记下来转发过去。
_BACKEND_ARGV: list[str] = []


def start_regret_job(record: Any) -> str:
    """持久化复盘记录并拉起一个独立进程做遗憾分析，返回 job id。

    为什么用子进程而不是线程：
    - 一次完整分析要做 30 多轮 3456 份 IS 提案 + 最多 256 次精确求解，耗时
      以十分钟计；放进 HTTP 进程会长时间占住 GIL，把正在进行的对局卡死。
    - 求解器的 native 全局缓存与 torch 权重都在子进程里重新初始化，天然与
      线上对局互不干扰；父进程只读磁盘上的作业文件，随时可以重启。
    """
    job_id = regret_jobs.create_job()
    regret_jobs.write_job_record(job_id, record)

    log_path = regret_jobs.worker_log_path(job_id)
    command = [
        sys.executable,
        "-m",
        "gui.regret_analysis",
        "--job-id",
        job_id,
        *_BACKEND_ARGV,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(REPO_ROOT)]
    )
    env["SPADES_REGRET_JOB_ROOT"] = str(regret_jobs.job_root())

    with open(log_path, "ab") as log_handle:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    return job_id


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
            parsed = urlparse(self.path)
            if parsed.path in {"/api/analyze-replay", "/analyze-replay"}:
                query = parse_qs(parsed.query)
                job_id = (query.get("jobId") or [""])[0]
                try:
                    snapshot = regret_jobs.job_snapshot(job_id)
                except regret_jobs.JobNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **snapshot})
                return
            if self.path in {"/", "/health", "/api/health"}:
                self._send_json(200, {
                    "ok": True,
                    "ai": provider.ai_name,
                    "seed": provider.seed,
                    "play_hyperparams": describe_play_config(provider),
                    "acting_bidder": provider.acting_bidder.describe(),
                    "nonnil_play_model": {
                        "model_id": provider.nonnil_play_actor.model_id,
                        "sha256": provider.nonnil_play_actor.sha256,
                    },
                    "nil_play_model": {
                        "model_id": provider.nil_play_actors.model_id,
                        "bundle_sha256": provider.nil_play_actors.sha256,
                    },
                })
                return
            self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            action_paths = {"/api/choose-action", "/choose-action"}
            showdown_paths = {"/api/check-showdown", "/check-showdown"}
            regret_paths = {"/api/analyze-replay", "/analyze-replay"}
            if self.path not in action_paths | showdown_paths | regret_paths:
                self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if self.path in regret_paths:
                try:
                    job_id = start_regret_job(payload)
                except Exception as exc:  # pragma: no cover - surfaced to browser
                    import traceback
                    traceback.print_exc()
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "jobId": job_id})
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
    parser = argparse.ArgumentParser(
        description="solver-leaf MLP + exact AI backend for the Spades GUI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ai", default="rule_exact",
                        help="Kept for compatibility; only rule_exact is served.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="remaining cards <= this → exact solver (first "
                             "52-threshold cards use deployed MLPs)")
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "configs" / "8.yaml"),
                        help="path to hyperparam YAML config for exact-stage sampling")
    parser.add_argument("--checkpoint-nil", type=str,
                        default=str(REPO_ROOT / "55_2nil.pt"),
                        help="[deprecated] production Nil play uses the four-role "
                             "actor bundle; this arg is ignored")
    parser.add_argument(
        "--nonnil-actor-checkpoint",
        type=str,
        default=str(DEFAULT_SOLVER_LEAF_ACTOR_PATH),
        help="hash-pinned non-Nil solver-leaf actor checkpoint",
    )
    parser.add_argument(
        "--nonnil-actor-sha256",
        type=str,
        default=DEFAULT_SOLVER_LEAF_ACTOR_SHA256,
        help="required SHA-256 for the non-Nil actor",
    )
    parser.add_argument(
        "--nonnil-actor-sidecar-sha256",
        type=str,
        default=DEFAULT_SOLVER_LEAF_SIDECAR_SHA256,
        help="required SHA-256 for the non-Nil actor metadata sidecar",
    )
    parser.add_argument(
        "--nil-actor-bundle",
        type=str,
        default=str(DEFAULT_NIL_ACTOR_BUNDLE_PATH),
        help="hash-pinned four-role Nil actor bundle manifest",
    )
    parser.add_argument(
        "--nil-actor-bundle-sha256",
        type=str,
        default=DEFAULT_NIL_ACTOR_BUNDLE_SHA256,
        help="required SHA-256 for the four-role Nil bundle manifest",
    )
    parser.add_argument("--bid-checkpoint", type=str,
                        default=str(REPO_ROOT / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt"),
                        help="[deprecated] bid_nsfp remains the frozen belief bidder")
    parser.add_argument("--acting-bid-checkpoint", type=str,
                        default=str(DEFAULT_ACTING_BID_CHECKPOINT),
                        help="path to the selected residual-Q acting checkpoint")
    parser.add_argument("--residual-bidder-config", type=str,
                        default=str(DEFAULT_RESIDUAL_BIDDER_CONFIG),
                        help="frozen residual bidder provenance config")
    parser.add_argument("--bid-policy-seed", type=int, default=None,
                        help="override the frozen acting-policy seed")
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
    global _BACKEND_ARGV
    args = parse_args()
    _BACKEND_ARGV = list(sys.argv[1:])
    if args.seed is not None:
        set_random_seed(args.seed)
    provider = RuleExactProvider(args)
    server = ThreadingHTTPServer((args.host, args.port), build_response_handler(provider))
    print(f"solver-leaf MLP backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"  exact_threshold={provider.exact_threshold} "
          f"(first {52 - provider.exact_threshold} cards use deployed MLPs)", flush=True)
    print(f"  solver_workers={provider.players[0]._num_workers}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
