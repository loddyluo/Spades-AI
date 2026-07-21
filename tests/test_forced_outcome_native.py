from __future__ import annotations

import copy
import ctypes
import os
import random

import pytest

from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers._native_compile import FASTEST_BUILD_RECIPE
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from trick_taking.solvers.native_lib_loader import (
    NATIVE_LIBRARY_ABI_VERSION,
    compute_native_build_id,
)


RULES = SpadesRules()
FASTEST_REQUIRED_SYMBOLS = (
    "solve_native",
    "solve_native_with_q",
    "analyze_forced_outcome_native",
)


def test_fastest_native_library_reports_current_build_id() -> None:
    solver = ExactDoubleDummyCppFastestSolver()
    assert solver.native_available

    source = os.path.join(
        os.path.dirname(__file__),
        "..",
        "trick_taking",
        "solvers",
        "exact_double_dummy_cpp_fastest_core.cpp",
    )
    expected_build_id = compute_native_build_id(
        source,
        required_symbols=FASTEST_REQUIRED_SYMBOLS,
        abi_version=NATIVE_LIBRARY_ABI_VERSION,
        build_recipe=FASTEST_BUILD_RECIPE,
    )

    build_id_function = solver._lib.spades_native_build_id
    build_id_function.argtypes = []
    build_id_function.restype = ctypes.c_char_p
    abi_function = solver._lib.spades_native_abi_version
    abi_function.argtypes = []
    abi_function.restype = ctypes.c_uint32

    assert build_id_function().decode("ascii") == expected_build_id
    assert abi_function() == NATIVE_LIBRARY_ABI_VERSION


def _card(code: str) -> Card:
    return Card(
        suit=Suit.from_short(code[-1]),
        rank=Rank.from_short(code[:-1]),
    )


def _state(
    hand_codes: list[list[str]],
    tricks_won: list[int],
    bids: list[str],
) -> GameState:
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [[_card(code) for code in hand] for hand in hand_codes]
    state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = list(bids)
    state.teams = [0, 1, 0, 1]
    state.turn = 0
    state.trick_leader = 0
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.trump_broken = True
    state.spades_broken = True
    state.tricks_won = list(tricks_won)
    state.cards_won = [[] for _ in range(4)]
    state.trick_history = []
    state.tricks_played = 11
    return state


def _apply_reference_card(state: GameState, player: int, card: Card) -> GameState:
    child = copy.deepcopy(state)
    child.play_card_to_table(player, card)
    if card.suit == Suit.SPADES:
        child.trump_broken = True
        child.spades_broken = True
    child.turn = (player + 1) % child.num_players
    if child.trick_complete:
        winner = RULES.winner_trick(child)
        child.complete_trick(winner)
        child.turn = winner
        child.trick_leader = winner
    return child


def terminal_signatures(state: GameState) -> set[tuple[int, int]]:
    if state.tricks_played >= 13:
        team0 = state.tricks_won[0] + state.tricks_won[2]
        nil_mask = sum(
            1 << seat
            for seat, bid in enumerate(state.max_bid)
            if bid in ("nil", "blind_nil") and state.tricks_won[seat] > 0
        )
        return {(team0, nil_mask)}

    player = state.turn
    signatures: set[tuple[int, int]] = set()
    for card in RULES.playable(state, state.hands[player], player):
        signatures.update(
            terminal_signatures(_apply_reference_card(state, player, card))
        )
    return signatures


FIXTURES = [
    pytest.param(
        [["TD", "QH"], ["6H", "KS"], ["8S", "5H"], ["2H", "3S"]],
        [3, 3, 3, 2],
        ["bid_1"] * 4,
        {(6, 0), (7, 0)},
        id="team-tricks-variable",
    ),
    pytest.param(
        [["4C", "5H"], ["6H", "AS"], ["QS", "8H"], ["7H", "3C"]],
        [0, 4, 4, 3],
        ["nil", "bid_1", "bid_1", "bid_1"],
        {(5, 0), (5, 1)},
        id="only-nil-outcome-variable",
    ),
    pytest.param(
        [["4H", "9S"], ["TH", "KS"], ["7H", "5D"], ["3S", "4S"]],
        [0, 4, 4, 3],
        ["nil", "bid_1", "bid_1", "bid_1"],
        {(4, 0)},
        id="fixed-with-nil",
    ),
]


@pytest.mark.parametrize(
    ("hand_codes", "tricks_won", "bids", "expected"),
    FIXTURES,
)
def test_reference_oracle_fixtures(
    hand_codes: list[list[str]],
    tricks_won: list[int],
    bids: list[str],
    expected: set[tuple[int, int]],
) -> None:
    assert terminal_signatures(_state(hand_codes, tricks_won, bids)) == expected


@pytest.mark.parametrize(
    ("hand_codes", "tricks_won", "bids", "expected"),
    FIXTURES,
)
def test_native_forced_outcome_matches_reference_fixtures(
    hand_codes: list[list[str]],
    tricks_won: list[int],
    bids: list[str],
    expected: set[tuple[int, int]],
) -> None:
    solver = ExactDoubleDummyCppFastestSolver()

    result = solver.analyze_forced_outcome(
        _state(hand_codes, tricks_won, bids),
        time_budget_seconds=1.0,
    )

    if len(expected) == 1:
        assert result["status"] == "fixed"
        assert (
            result["team0_final_tricks"],
            result["nil_broken_mask"],
        ) == next(iter(expected))
    else:
        assert result["status"] == "variable"


def test_zero_budget_is_inconclusive_without_poisoning_next_search() -> None:
    hand_codes, tricks_won, bids, expected = FIXTURES[-1].values
    state = _state(hand_codes, tricks_won, bids)
    solver = ExactDoubleDummyCppFastestSolver()

    timed_out = solver.analyze_forced_outcome(state, time_budget_seconds=0.0)
    completed = solver.analyze_forced_outcome(state, time_budget_seconds=1.0)

    assert timed_out["status"] == "timeout"
    assert completed["status"] == "fixed"
    assert (
        completed["team0_final_tricks"],
        completed["nil_broken_mask"],
    ) == next(iter(expected))


def _random_two_trick_state(rng: random.Random) -> GameState:
    cards = rng.sample(list(_STANDARD_CARDS), 8)
    hands = [cards[seat * 2 : (seat + 1) * 2] for seat in range(4)]
    winners = [rng.randrange(4) for _ in range(11)]
    tricks_won = [winners.count(seat) for seat in range(4)]
    bids = ["nil" if rng.random() < 0.25 else "bid_1" for _ in range(4)]

    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = hands
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = bids
    state.teams = [0, 1, 0, 1]
    state.turn = rng.randrange(4)
    state.trick_leader = state.turn
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.trump_broken = rng.choice([False, True])
    state.spades_broken = state.trump_broken
    state.tricks_won = tricks_won
    state.cards_won = [[] for _ in range(4)]
    state.trick_history = []
    state.tricks_played = 11
    return state


def test_randomized_oracle_parity() -> None:
    rng = random.Random(20260721)
    solver = ExactDoubleDummyCppFastestSolver()

    for _ in range(200):
        state = _random_two_trick_state(rng)
        expected = terminal_signatures(state)
        actual = solver.analyze_forced_outcome(state, time_budget_seconds=1.0)

        assert actual["status"] != "timeout"
        if len(expected) == 1:
            assert actual["status"] == "fixed"
            assert (
                actual["team0_final_tricks"],
                actual["nil_broken_mask"],
            ) == next(iter(expected))
        else:
            assert actual["status"] == "variable"


def test_five_trick_fixed_position_completes_inside_wall_clock_budget() -> None:
    ranks = list(Rank)[:5]
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [
        [Card(suit, rank) for rank in ranks]
        for suit in (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES)
    ]
    state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = ["bid_1"] * 4
    state.teams = [0, 1, 0, 1]
    state.turn = state.trick_leader = 0
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.trump_broken = state.spades_broken = False
    state.tricks_won = [2, 2, 2, 2]
    state.cards_won = [[] for _ in range(4)]
    state.trick_history = []
    state.tricks_played = 8

    result = ExactDoubleDummyCppFastestSolver().analyze_forced_outcome(
        state,
        time_budget_seconds=1.0,
    )

    assert result["status"] == "fixed"
    assert result["team0_final_tricks"] == 4
    assert result["elapsed_ms"] < 1100
