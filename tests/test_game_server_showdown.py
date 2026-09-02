from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gui.game_server import AiFallbackError, GameRoom, _deal_hands_frontend_compat
from residual_bidder.actions import BidAction
from residual_bidder.hybrid import _initial_state
from rl.nil_solver_leaf_env import NIL_ROLES
from rl.policy_network import PolicyMLP
from trick_taking.card import Suit
from trick_taking.forced_outcome import (
    ShowdownCheck,
    check_for_showdown,
    deterministic_continuation,
    outcome_signature,
)
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules


class _FakeSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


def test_room_routes_all_ai_bids_through_deployed_acting_bidder() -> None:
    class FixedActingBidder:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def choose(self, state, legal_bids, **kwargs):
            self.calls.append({"state": state, "legal": list(legal_bids), **kwargs})
            return SimpleNamespace(
                action=BidAction.BID_3,
                effective_policy_id="residual-test",
                fallback_reason=None,
            )

    class BiddingPlayer:
        def __init__(self) -> None:
            self.last_bid_info = None
            self.notified: list[tuple[int, str]] = []
            self.team_bids = None

        def place_bid(self, legal_bids, view):
            raise AssertionError("legacy acting bidder must not be called")

        def bid_placed(self, bidder, bid):
            self.notified.append((bidder, bid))

        def set_teams(self, teams, bids):
            self.team_bids = (list(teams), list(bids))

    async def scenario() -> None:
        acting_bidder = FixedActingBidder()
        room = GameRoom("RESIDUAL", 202607220003, acting_bidder=acting_bidder)
        room.state = _initial_state(room.seed, room.rules)
        room.ai_players = {seat: BiddingPlayer() for seat in range(4)}

        await room._bidding_phase()

        assert room.state.max_bid == ["bid_3"] * 4
        assert len(acting_bidder.calls) == 4
        assert {call["logical_seat"] for call in acting_bidder.calls} == {0, 1, 2, 3}
        assert all(call["room_id"] == "RESIDUAL" for call in acting_bidder.calls)
        assert all(player.last_bid_info["policy_id"] == "residual-test"
                   for player in room.ai_players.values())

    asyncio.run(scenario())


def test_room_rejects_acting_bidder_fallback() -> None:
    class FallbackActingBidder:
        def choose(self, state, legal_bids, **kwargs):
            return SimpleNamespace(
                action=BidAction.BID_3,
                effective_policy_id="legacy-nsfp-fallback",
                fallback_reason="residual-policy-error: worker OOM",
            )

    class BiddingPlayer:
        last_bid_info = None

        def bid_placed(self, bidder, bid):
            return None

    async def scenario() -> None:
        room = GameRoom(
            "FALLBACK",
            202607220004,
            acting_bidder=FallbackActingBidder(),
        )
        room.state = _initial_state(room.seed, room.rules)
        room.ai_players = {seat: BiddingPlayer() for seat in range(4)}

        with pytest.raises(AiFallbackError, match=r"叫牌.*worker OOM"):
            await room._bidding_phase()

        assert all(bid.is_pass for bid in room.state.bids)
        assert room.state.max_bid == [None, None, None, None]

    asyncio.run(scenario())


def test_room_rejects_card_play_fallback_and_illegal_card_substitution() -> None:
    class PlayingPlayer:
        def __init__(self, mode: str, return_legal: bool) -> None:
            self.last_play_info = {}
            self.mode = mode
            self.return_legal = return_legal

        def play_card(self, legal_cards, view):
            self.last_play_info = {"mode": self.mode}
            if self.return_legal:
                return legal_cards[0]
            return next(card for card in view["state"].all_cards if card not in legal_cards)

    async def scenario() -> None:
        room = GameRoom("PLAY-FALLBACK", 202607220005)
        room.state = _late_state()
        seat = room.state.turn
        legal = room.rules.playable(room.state, room.state.hands[seat], seat)

        room.ai_players = {
            seat: PlayingPlayer("exact_no_match_fallback", return_legal=True)
        }
        with pytest.raises(
            AiFallbackError,
            match=r"出牌.*exact_no_match_fallback",
        ):
            await room._run_ai_play(seat, legal)

        room.ai_players = {
            seat: PlayingPlayer("exact_is_determinized", return_legal=False)
        }
        with pytest.raises(AiFallbackError, match=r"返回非法牌"):
            await room._run_ai_play(seat, legal)

    asyncio.run(scenario())


def test_room_records_ai_diagnostics_in_server_side_replay_store() -> None:
    class DetailedPlayer:
        def __init__(self) -> None:
            self.last_play_info = {}

        def play_card(self, legal_cards, view):
            card = legal_cards[0]
            self.last_play_info = {
                "mode": "exact_is_determinized",
                "action_scores": [{"action": card, "value": 0.0}],
            }
            return card

    async def scenario() -> None:
        room = GameRoom("DETAIL", 2026082102)
        room.state = _late_state()
        seat = room.state.turn
        legal = room.rules.playable(room.state, room.state.hands[seat], seat)
        room.ai_players = {seat: DetailedPlayer()}

        chosen = await room._run_ai_play(seat, legal)
        play_index = room.state.tricks_played * 4 + len(room.state.table_cards)

        assert room._play_analyses[play_index]["chosen_card"] == (
            f"{chosen.rank.short}{chosen.suit.short}"
        )
        assert room._play_analyses[play_index]["action_scores"][0]["action"] == (
            f"{chosen.rank.short}{chosen.suit.short}"
        )

    asyncio.run(scenario())


def test_remote_clients_receive_a_fatal_fallback_message() -> None:
    class FailingRoom(GameRoom):
        async def _bidding_phase(self) -> None:
            raise AiFallbackError("座位 1 出牌：exact_no_match_fallback")

    async def scenario() -> None:
        first = _FakeSocket()
        second = _FakeSocket()
        actor = PolicyMLP(input_dim=536, hidden_dims=[4], output_dim=52)
        room = FailingRoom(
            "GUI-ERROR",
            202607220006,
            exact_solver=object(),
            nonnil_play_actor=SimpleNamespace(
                actor=actor,
                model_id="nonnil-test",
                sha256="a" * 64,
            ),
            nil_play_actors=SimpleNamespace(
                actors={role: actor for role in NIL_ROLES},
                model_id="nil-test",
                sha256="b" * 64,
            ),
        )
        room.connections = {0: first, 2: second}

        await room.start_game()

        for socket in (first, second):
            message = socket.messages[-1]
            assert message["type"] == "error"
            assert message["code"] == "ai_fallback"
            assert message["fatal"] is True
            assert "exact_no_match_fallback" in message["message"]

    asyncio.run(scenario())


def _late_state() -> GameState:
    rules = SpadesRules()
    hands = _deal_hands_frontend_compat(20260721)
    state = GameState()
    state.init_for_deal(4, hands, [], [card for hand in hands for card in hand])
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.max_bid = ["nil", "bid_3", "bid_3", "bid_3"]
    state.bids = [
        Bid(player_id=seat, value=bid, is_pass=False)
        for seat, bid in enumerate(state.max_bid)
    ]
    state.trump_suit = Suit.SPADES
    state.turn = state.trick_leader = 0

    for _ in range(11):
        for _ in range(4):
            seat = state.turn
            legal = rules.playable(state, state.hands[seat], seat)
            card = min(legal, key=lambda candidate: candidate.card_id)
            state.play_card_to_table(seat, card)
            if card.suit == Suit.SPADES:
                state.trump_broken = state.spades_broken = True
            state.turn = (seat + 1) % 4
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = state.trick_leader = winner
    return state


def _resolution(state: GameState):
    _, terminal = deterministic_continuation(state)
    signature = outcome_signature(terminal)

    class FixedSolver:
        def analyze_forced_outcome(self, checked, time_budget_seconds=1.0):
            return {
                "status": "fixed",
                "team0_final_tricks": signature[0],
                "nil_broken_mask": signature[1],
            }

    result = check_for_showdown(state, FixedSolver())
    assert result.resolution is not None
    return result.resolution


def _room(checker=None) -> tuple[GameRoom, _FakeSocket, _FakeSocket]:
    first = _FakeSocket()
    second = _FakeSocket()
    room = GameRoom("ROOM", 1, exact_solver=object(), showdown_checker=checker)
    room.connections = {0: first, 2: second}
    room.state = _late_state()
    return room, first, second


def test_fixed_offer_reveals_all_hands_and_requires_two_distinct_humans() -> None:
    async def scenario() -> None:
        room, first, second = _room()
        resolution = _resolution(room.state)
        task = asyncio.create_task(room._offer_showdown(resolution))
        await asyncio.sleep(0)

        offer = first.messages[-1]["showdown"]
        assert offer["revealedHands"] == second.messages[-1]["showdown"]["revealedHands"]
        assert [len(hand) for hand in offer["revealedHands"]] == [2, 2, 2, 2]
        assert offer["confirmedSeats"] == []
        showdown_id = offer["id"]

        assert room.receive_action(0, {
            "type": "showdown_confirm",
            "showdownId": showdown_id,
        })
        await asyncio.sleep(0)
        assert not task.done()
        assert room.state.phase == Phase.PLAYING
        assert first.messages[-1]["showdown"]["confirmedSeats"] == [0]

        assert not room.receive_action(0, {
            "type": "showdown_confirm",
            "showdownId": showdown_id,
        })
        assert not room.receive_action(2, {
            "type": "showdown_confirm",
            "showdownId": showdown_id + 1,
        })
        assert not task.done()

        assert room.receive_action(2, {
            "type": "showdown_confirm",
            "showdownId": showdown_id,
        })
        await task

        assert room.state.phase == Phase.SCORING
        assert room.state.tricks_played == 13
        assert room.state.tricks_won == list(resolution.final_tricks_won)
        assert room.showdown_pending is False
        assert first.messages[-1].get("showdown") is None

    asyncio.run(scenario())


def test_timeout_checker_continues_without_revealing_hands() -> None:
    def timeout_checker(state, solver, *, time_budget_seconds):
        return ShowdownCheck(status="timeout")

    async def scenario() -> None:
        room, first, second = _room(timeout_checker)

        offered = await room._maybe_offer_showdown()

        assert offered is False
        assert room.showdown_pending is False
        assert all(message.get("showdown") is None for message in first.messages + second.messages)

    asyncio.run(scenario())


def test_showdown_checker_failure_is_reported_instead_of_silently_ignored() -> None:
    def failing_checker(state, solver, *, time_budget_seconds):
        raise MemoryError("showdown worker OOM")

    async def scenario() -> None:
        room, _, _ = _room(failing_checker)

        with pytest.raises(
            AiFallbackError,
            match=r"自动摊牌检查失败.*worker OOM",
        ):
            await room._maybe_offer_showdown()

        assert room.showdown_pending is False

    asyncio.run(scenario())


def test_disconnect_while_pending_is_not_consent() -> None:
    async def scenario() -> None:
        room, _, _ = _room()
        task = asyncio.create_task(room._offer_showdown(_resolution(room.state)))
        await asyncio.sleep(0)

        room.remove_connection(2)

        with pytest.raises(RuntimeError, match="disconnected"):
            await task
        assert room.state.phase == Phase.PLAYING
        assert room.state.tricks_played == 11

    asyncio.run(scenario())


def test_ordinary_actions_are_accepted_only_from_expected_sender() -> None:
    room, _, _ = _room()
    room._expected_action_seat = 0
    room._expected_action_type = "play"
    room._action_event.clear()

    assert not room.receive_action(2, {"type": "play", "card": "AS"})
    assert not room._action_event.is_set()
    assert not room.receive_action(0, {"type": "bid", "bid": {"value": 1}})
    assert not room._action_event.is_set()
    assert room.receive_action(0, {"type": "play", "card": "AS"})
    assert room._action_event.is_set()
