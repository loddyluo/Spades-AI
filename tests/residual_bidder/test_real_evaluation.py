from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from residual_bidder.actions import BidAction
from residual_bidder.hybrid import _initial_state
from residual_bidder.real_evaluation import ActingBidOverride, evaluate_real_deal
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.driver import GeneralCardGame
from trick_taking.game_state import Bid
from trick_taking.games.spades import SpadesRules


class _InnerPlayer:
    def __init__(self) -> None:
        self.started: tuple[int, int] | None = None
        self.team_bids: list[Any] | None = None

    def start_game(self, position: int, hand: list[Any], num_players: int) -> None:
        self.started = (position, num_players)

    def play_card(self, legal_cards: list[Any], state_view: dict[str, Any]) -> Any:
        del state_view
        return legal_cards[0]

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        del teams
        self.team_bids = list(bid_values)


class _Policy:
    def __init__(self, action: BidAction) -> None:
        self.action = action
        self.keys: list[Any] = []

    def sample(self, state, legal_bids, key, *, strict):
        del state, legal_bids
        assert strict is True
        self.keys.append(key)
        return SimpleNamespace(action=self.action, effective_policy_id="test-policy")


def test_acting_override_replaces_only_place_bid_and_uses_fixed_key() -> None:
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = _initial_state(202607300001, rules)
    inner = _InnerPlayer()
    policy = _Policy(BidAction.BID_3)
    player = ActingBidOverride(
        inner,
        policy,
        policy_seed=77,
        deal_id="deal-x",
        room_id="room-y",
    )
    player.start_game(state.current_bidder, list(state.hands[state.current_bidder]), 4)

    bid = player.place_bid(rules.legal_bids(state, state.current_bidder), {"state": state})

    assert bid == "bid_3"
    assert inner.started == (state.current_bidder, 4)
    assert player.play_card(["sentinel"], {}) == "sentinel"
    assert len(policy.keys) == 1
    key = policy.keys[0]
    assert (key.policy_seed, key.deal_id, key.room_id) == (77, "deal-x", "room-y")
    assert (key.logical_seat, key.bid_index) == (state.current_bidder, 0)


def test_real_duplicate_swaps_acting_policies_but_not_inner_player() -> None:
    candidate = _Policy(BidAction.BID_3)
    opponent = _Policy(BidAction.BID_4)
    built: list[_InnerPlayer] = []

    def player_factory() -> _InnerPlayer:
        player = _InnerPlayer()
        built.append(player)
        return player

    class _Runner:
        def __init__(self, *, players, seed, verbose, rules) -> None:
            del seed, verbose, rules
            self.players = players

        def play_game(self):
            room_id = self.players[0].room_id
            if room_id == "candidate-team-0":
                assert [p.acting_policy is candidate for p in self.players] == [
                    True,
                    False,
                    True,
                    False,
                ]
                return SimpleNamespace(scores=[30.0, -30.0, 30.0, -30.0])
            assert [p.acting_policy is candidate for p in self.players] == [
                False,
                True,
                False,
                True,
            ]
            return SimpleNamespace(scores=[10.0, -10.0, 10.0, -10.0])

    result = evaluate_real_deal(
        202607300002,
        candidate,
        opponent,
        player_factory,
        policy_seed=88,
        runner_factory=_Runner,
    )

    assert len(built) == 8
    assert result.room_team0_margin == 30.0
    assert result.room_team1_margin == -10.0
    assert result.duplicate_margin == 20.0


def test_match_runner_reports_bids_in_physical_seat_order() -> None:
    players = [_InnerPlayer() for _ in range(4)]
    runner = SpadesMatchRunner(
        players=players,
        seed=202607300003,
        verbose=False,
        rules=SpadesRules(enable_nil=True, enable_blind_nil=False),
    )
    chronological = [
        Bid(player_id=2, value="nil"),
        Bid(player_id=3, value="bid_3"),
        Bid(player_id=0, value="bid_4"),
        Bid(player_id=1, value="bid_2"),
    ]
    runner.state.bids = chronological
    runner.state.max_bid = ["bid_4", "bid_2", "nil", "bid_3"]

    runner._set_teams()

    assert all(
        player.team_bids == ["bid_4", "bid_2", "nil", "bid_3"]
        for player in players
    )


def test_general_driver_reports_bids_in_physical_seat_order() -> None:
    players = [_InnerPlayer() for _ in range(4)]
    game = GeneralCardGame(
        SpadesRules(enable_nil=True, enable_blind_nil=False),
        players,  # type: ignore[arg-type]
        seed=202607300004,
    )
    game._deal()
    game.state.bids = [
        Bid(player_id=3, value="bid_3"),
        Bid(player_id=0, value="nil"),
        Bid(player_id=1, value="bid_4"),
        Bid(player_id=2, value="bid_2"),
    ]
    game.state.max_bid = ["nil", "bid_4", "bid_2", "bid_3"]

    game._set_teams()

    assert all(
        player.team_bids == ["nil", "bid_4", "bid_2", "bid_3"]
        for player in players
    )
