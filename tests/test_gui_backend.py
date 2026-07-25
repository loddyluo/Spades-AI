import copy
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gui.backend import (
    RuleExactProvider,
    build_full_showdown_state,
    build_local_state,
)
from residual_bidder.actions import BidAction
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from trick_taking.card import Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.forced_outcome import ShowdownStateError
from trick_taking.games.spades import SpadesRules


def _bidding_payload() -> dict[str, object]:
    return {
        "seed": 202607220002,
        "firstSeat": 0,
        "phase": "bidding",
        "currentPlayer": 2,
        "leader": 0,
        "remainingHand": [f"{rank}H" for rank in "23456789TJQKA"],
        "bids": [
            {"type": "normal", "value": 3},
            {"type": "nil", "value": 0},
            None,
            None,
        ],
        "spadesBroken": False,
        "completedTricks": [],
        "currentTrick": [],
    }


def test_bidding_state_preserves_public_bid_history_for_residual_runtime() -> None:
    state, seat = build_local_state(_bidding_payload())

    assert seat == state.current_bidder == 2
    assert [(bid.player_id, bid.value) for bid in state.bids] == [
        (0, "bid_3"),
        (1, "nil"),
    ]
    assert state.max_bid == ["bid_3", "nil", None, None]


def test_provider_routes_acting_bid_through_residual_bidder_only() -> None:
    state, seat = build_local_state(_bidding_payload())

    class FixedActingBidder:
        def __init__(self) -> None:
            self.call = None

        def choose(self, chosen_state, legal_bids, **kwargs):
            self.call = (chosen_state, list(legal_bids), kwargs)
            return SimpleNamespace(
                action=BidAction.BID_4,
                effective_policy_id="residual-test",
                fallback_reason=None,
            )

    acting_bidder = FixedActingBidder()
    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.acting_bidder = acting_bidder
    provider.players = [SimpleNamespace(last_bid_info=None) for _ in range(4)]

    choice = provider._choose_bid(state, seat, _bidding_payload())

    assert (choice.value, choice.bid_type, choice.detail) == (4, "normal", "residual_bid")
    assert acting_bidder.call[0] is state
    assert acting_bidder.call[2] == {
        "logical_seat": 2,
        "deal_id": "local:202607220002",
        "room_id": "http-local",
    }
    assert provider.players[seat].last_bid_info["policy_id"] == "residual-test"


def test_provider_rejects_acting_bidder_fallback() -> None:
    state, seat = build_local_state(_bidding_payload())

    class FallbackActingBidder:
        def choose(self, chosen_state, legal_bids, **kwargs):
            return SimpleNamespace(
                action=BidAction.BID_4,
                effective_policy_id="legacy-nsfp-fallback",
                fallback_reason="residual-policy-error: worker OOM",
            )

    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.acting_bidder = FallbackActingBidder()
    provider.players = [SimpleNamespace(last_bid_info=None) for _ in range(4)]

    with pytest.raises(RuntimeError, match=r"bidding triggered fallback.*worker OOM"):
        provider._choose_bid(state, seat, _bidding_payload())

    assert provider.players[seat].last_bid_info["fallback_reason"].endswith(
        "worker OOM"
    )


def test_provider_rejects_card_play_fallback() -> None:
    payload = {
        "phase": "playing",
        "currentPlayer": 0,
        "leader": 0,
        "remainingHand": ["2H"],
        "bids": [
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
        ],
        "spadesBroken": False,
        "completedTricks": [],
        "currentTrick": [],
    }
    state, seat = build_local_state(payload)

    class FallbackPlayer:
        last_play_info: dict[str, str] = {}

        def play_card(self, legal_cards, view):
            self.last_play_info = {"mode": "exact_no_match_fallback"}
            return legal_cards[0]

    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.rules = SpadesRules()

    with pytest.raises(
        RuntimeError,
        match=r"card play triggered fallback.*exact_no_match_fallback",
    ):
        provider._choose_play(FallbackPlayer(), state, seat, {"state": state})


@pytest.mark.parametrize(
    ("public_history", "payload_flag", "expected"),
    [
        ({"currentTrick": [{"seat": 0, "card": "AS"}]}, False, True),
        (
            {
                "completedTricks": [
                    {
                        "cards": [
                            {"seat": 0, "card": "2H"},
                            {"seat": 1, "card": "AS"},
                            {"seat": 2, "card": "3H"},
                            {"seat": 3, "card": "4H"},
                        ]
                    }
                ]
            },
            False,
            True,
        ),
        ({"currentTrick": [{"seat": 0, "card": "AH"}]}, False, False),
        ({"currentTrick": [{"seat": 0, "card": "AH"}]}, True, True),
    ],
)
def test_build_local_state_derives_spades_broken_from_public_cards(
    public_history: dict[str, object], payload_flag: bool, expected: bool
) -> None:
    payload = {
        "phase": "playing",
        "currentPlayer": 1,
        "leader": 0,
        "spadesBroken": payload_flag,
        **public_history,
    }

    state, _ = build_local_state(payload)

    assert state.spades_broken is expected
    assert state.trump_broken is expected


def test_http_and_authoritative_states_share_seed_and_proposal_sequence() -> None:
    ranks = "23456789TJQKA"
    payload = {
        "phase": "playing",
        "currentPlayer": 0,
        "leader": 0,
        "remainingHand": [f"{rank}H" for rank in ranks],
        "bids": [
            {"type": "normal", "value": 3},
            {"type": "normal", "value": 4},
            {"type": "normal", "value": 3},
            {"type": "normal", "value": 3},
        ],
        "spadesBroken": False,
        "completedTricks": [],
        "currentTrick": [],
    }
    http_state, seat = build_local_state(payload)
    authoritative_state = copy.deepcopy(http_state)
    authoritative_state.hands = [
        list(reversed(http_state.hands[0])),
        [card for card in _STANDARD_CARDS if card.suit == Suit.CLUBS],
        [card for card in _STANDARD_CARDS if card.suit == Suit.DIAMONDS],
        [card for card in _STANDARD_CARDS if card.suit == Suit.SPADES],
    ]
    authoritative_state.hand_bitsets = [
        cards_to_bitset(hand) for hand in authoritative_state.hands
    ]
    authoritative_state.all_cards = list(reversed(authoritative_state.all_cards))
    rules = SpadesRules()
    http_legal = rules.playable(http_state, http_state.hands[seat], seat)
    authoritative_legal = rules.playable(
        authoritative_state,
        authoritative_state.hands[seat],
        seat,
    )
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    http_seed = player._decision_seed(http_state, seat, http_legal)
    authoritative_seed = player._decision_seed(
        authoritative_state,
        seat,
        authoritative_legal,
    )
    played = {0: [], 1: [], 2: [], 3: []}
    http_proposal = player._generate_proposal(
        http_state.all_cards,
        seat,
        http_state.hands[seat],
        played,
        random.Random(http_seed),
    )
    authoritative_proposal = player._generate_proposal(
        authoritative_state.all_cards,
        seat,
        authoritative_state.hands[seat],
        played,
        random.Random(authoritative_seed),
    )

    assert http_seed == authoritative_seed
    assert [[card.card_id for card in hand] for hand in http_proposal] == [
        [card.card_id for card in hand] for hand in authoritative_proposal
    ]


class _OverlapDetectingPlayer:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.last_play_info: dict[str, str] = {}

    def start_game(self, position, hand, num_players) -> None:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)

    def set_teams(self, teams, bids) -> None:
        return None

    def card_played(self, player_id, card) -> None:
        return None

    def play_card(self, legal_cards, state_view):
        try:
            return legal_cards[0]
        finally:
            with self._guard:
                self.active -= 1


def test_provider_serializes_replay_on_reused_mutable_players() -> None:
    player = _OverlapDetectingPlayer()
    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.players = [player, player, player, player]
    provider.rules = SpadesRules()
    provider._decision_lock = threading.Lock()
    payload = {
        "phase": "playing",
        "currentPlayer": 0,
        "leader": 0,
        "remainingHand": ["2H"],
        "bids": [
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
        ],
        "spadesBroken": False,
        "completedTricks": [],
        "currentTrick": [],
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(provider.choose_action, [payload, payload]))

    assert [result.card for result in results] == ["2H", "2H"]
    assert player.max_active == 1


_SHOWDOWN_HISTORY = [
    [(0, "3D"), (1, "AD"), (2, "JD"), (3, "QD")],
    [(1, "6C"), (2, "KC"), (3, "9C"), (0, "JC")],
    [(2, "8D"), (3, "KD"), (0, "9D"), (1, "2C")],
    [(3, "TC"), (0, "KH"), (1, "4C"), (2, "AC")],
    [(2, "2D"), (3, "4D"), (0, "6D"), (1, "5S")],
    [(1, "8H"), (2, "JH"), (3, "5H"), (0, "9H")],
    [(2, "8C"), (3, "5C"), (0, "3H"), (1, "7C")],
    [(2, "7D"), (3, "TS"), (0, "TD"), (1, "AH")],
    [(3, "2S"), (0, "8S"), (1, "AS"), (2, "JS")],
    [(1, "6H"), (2, "2H"), (3, "6S"), (0, "QH")],
    [(3, "QC"), (0, "7S"), (1, "QS"), (2, "3C")],
]


def _showdown_payload() -> dict[str, object]:
    return {
        "phase": "playing",
        "currentPlayer": 1,
        "leader": 1,
        "remainingHands": [
            ["4H", "9S"],
            ["TH", "KS"],
            ["7H", "5D"],
            ["3S", "4S"],
        ],
        "bids": [
            {"type": "nil", "value": 0},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
            {"type": "normal", "value": 1},
        ],
        "spadesBroken": True,
        "tricksWon": [0, 4, 4, 3],
        "completedTricks": [
            {
                "cards": [
                    {"seat": seat, "card": card}
                    for seat, card in trick
                ]
            }
            for trick in _SHOWDOWN_HISTORY
        ],
        "currentTrick": [],
    }


def test_full_showdown_builder_preserves_all_authoritative_hands() -> None:
    payload = _showdown_payload()

    state = build_full_showdown_state(payload)

    assert [
        [f"{card.rank.short}{card.suit.short}" for card in hand]
        for hand in state.hands
    ] == payload["remainingHands"]
    assert state.tricks_won == [0, 4, 4, 3]
    assert state.turn == state.trick_leader == 1
    assert state.tricks_played == 11
    assert len(state.trick_history) == 11


def test_full_showdown_builder_rejects_claimed_trick_count_mismatch() -> None:
    payload = _showdown_payload()
    payload["tricksWon"] = [1, 3, 4, 3]

    with pytest.raises(ShowdownStateError, match="tricksWon"):
        build_full_showdown_state(payload)


def test_provider_serializes_fixed_showdown_without_touching_acting_players() -> None:
    class FixedSolver:
        def analyze_forced_outcome(self, state, time_budget_seconds=1.0):
            return {
                "status": "fixed",
                "team0_final_tricks": 4,
                "nil_broken_mask": 0,
            }

    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.exact_solver = FixedSolver()

    response = provider.check_showdown(_showdown_payload())

    assert response["status"] == "fixed"
    assert response["resolution"]["teamTricks"] == [4, 9]
    assert response["resolution"]["nilOutcomes"] == [True, None, None, None]
    assert len(response["resolution"]["continuation"]) == 8


def test_showdown_builder_rejects_in_progress_trick_and_duplicate_cards() -> None:
    mid_trick = _showdown_payload()
    mid_trick["currentTrick"] = [{"seat": 1, "card": "TH"}]
    with pytest.raises(ShowdownStateError):
        build_full_showdown_state(mid_trick)

    duplicate = _showdown_payload()
    duplicate["remainingHands"][1][0] = "4H"
    with pytest.raises(ShowdownStateError):
        build_full_showdown_state(duplicate)


def test_acting_state_builder_does_not_consume_full_showdown_hands() -> None:
    payload = _showdown_payload()
    payload["remainingHand"] = list(payload["remainingHands"][1])

    state, seat = build_local_state(payload)

    assert seat == 1
    assert state.hands[seat] != []
    assert [card.card_id for card in state.hands[0]] != [
        card.card_id for card in build_full_showdown_state(payload).hands[0]
    ]
