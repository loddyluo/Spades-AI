from __future__ import annotations

import copy
import multiprocessing
import random

import pytest

from strategy.hyperparam_config import BudgetConfig, HyperparamConfig
from strategy.rule_exact_first4_player import (
    _SOLVER_MP_START_METHOD,
    RuleExactFirst4Player,
    _parallel_solve_worker,
)
from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import GameState, Phase, TrickRecord


def _card(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


def _state(
    hands: list[list[Card]],
    *,
    turn: int = 0,
    tricks_played: int = 0,
    tricks_won: list[int] | None = None,
) -> GameState:
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [list(hand) for hand in hands]
    state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = ["bid_2", "bid_2", "bid_2", "bid_2"]
    state.teams = [0, 1, 0, 1]
    state.turn = turn
    state.trick_leader = turn
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.trump_broken = False
    state.spades_broken = False
    state.tricks_won = list(tricks_won or [0, 0, 0, 0])
    state.cards_won = [[], [], [], []]
    state.trick_history = []
    state.played_bitset = 0
    state.tricks_played = tricks_played
    return state


class _RecordingSolver:
    def __init__(self, q_values: dict[int, float] | None = None) -> None:
        self.q_values = q_values
        self.snapshots: list[dict[str, object]] = []

    def solve_with_q_fast(self, state: GameState) -> dict[int, float]:
        self.snapshots.append(
            {
                "turn": state.turn,
                "tricks_played": state.tricks_played,
                "tricks_won": list(state.tricks_won),
                "table": [(pid, card.card_id) for pid, card in state.table_cards],
            }
        )
        if self.q_values is not None:
            return dict(self.q_values)
        return {card.card_id: 0.0 for card in state.hands[state.turn]}


class _FailIfDecisionLogicRuns(RuleExactFirst4Player):
    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        raise AssertionError("exact search must not run for a forced action")

    def _rule_play(self, legal_cards: list[Card], state_view: dict) -> Card:
        raise AssertionError("rule logic must not run for a forced action")


@pytest.mark.parametrize(
    ("cards_per_hand", "expected_mode"),
    [
        (1, "last_card_direct"),
        (2, "single_action_direct"),
        (10, "single_action_direct"),
    ],
)
def test_single_legal_card_skips_all_decision_logic(
    cards_per_hand: int,
    expected_mode: str,
) -> None:
    hands = [
        list(_STANDARD_CARDS[seat * cards_per_hand : (seat + 1) * cards_per_hand])
        for seat in range(4)
    ]
    state = _state(hands, turn=0)
    legal_cards = [hands[0][0]]
    player = _FailIfDecisionLogicRuns(exact_solver=object(), exact_threshold=36)
    player.start_game(0, hands[0], 4)

    chosen = player.play_card(legal_cards, {"state": state})

    assert chosen == legal_cards[0]
    assert player.last_play_info == {"mode": expected_mode}


@pytest.mark.parametrize(
    ("actor", "played_q", "other_q", "expected"),
    [
        (0, 10.0, -10.0, 1.0),
        (0, -10.0, 10.0, 0.25),
        (1, -10.0, 10.0, 1.0),
        (1, 10.0, -10.0, 0.25),
    ],
)
def test_importance_weight_uses_acting_team_q_direction(
    actor: int,
    played_q: float,
    other_q: float,
    expected: float,
) -> None:
    played = _card(Suit.HEARTS, Rank.TWO)
    other = _card(Suit.HEARTS, Rank.THREE)
    filler = _card(Suit.CLUBS, Rank.TWO)
    hands = [[filler] for _ in range(4)]
    hands[actor] = [played, other]
    solver = _RecordingSolver({played.card_id: played_q, other.card_id: other_q})
    config = HyperparamConfig(trick_num_threshold=0, bad_action_weight="0.25")
    player = RuleExactFirst4Player(
        exact_solver=solver,
        hyperparam_config=config,
        num_workers=1,
    )

    weight = player._compute_importance_weight(
        hands,
        [(actor, played)],
        bid_prod=1.0,
        original_state=_state(hands, turn=actor),
    )

    assert weight == pytest.approx(expected)


def test_replay_prefix_rebuilds_trick_counters_before_solver_calls() -> None:
    ha = _card(Suit.HEARTS, Rank.ACE)
    hk = _card(Suit.HEARTS, Rank.KING)
    s2 = _card(Suit.SPADES, Rank.TWO)
    hq = _card(Suit.HEARTS, Rank.QUEEN)
    c2 = _card(Suit.CLUBS, Rank.TWO)
    hands = [[ha], [hk], [s2, c2], [hq]]
    # The low spade must trump the ace of hearts, proving replay uses Spades
    # winner semantics rather than merely selecting the highest led-suit card.
    sequence = [(0, ha), (1, hk), (2, s2), (3, hq), (2, c2)]
    solver = _RecordingSolver()
    config = HyperparamConfig(trick_num_threshold=0)
    player = RuleExactFirst4Player(
        exact_solver=solver,
        hyperparam_config=config,
        num_workers=1,
    )
    current_state = _state(
        hands,
        turn=2,
        tricks_played=1,
        tricks_won=[0, 0, 1, 0],
    )

    weight = player._compute_importance_weight(
        hands,
        sequence,
        bid_prod=1.0,
        original_state=current_state,
    )

    assert weight == pytest.approx(1.0)
    assert [snapshot["tricks_played"] for snapshot in solver.snapshots] == [0, 0, 0, 0, 1]
    assert [snapshot["tricks_won"] for snapshot in solver.snapshots[:4]] == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert solver.snapshots[4]["tricks_won"] == [0, 0, 1, 0]


def _visible_state_pair() -> tuple[GameState, GameState, list[Card], list[Card]]:
    own = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.SPADES, Rank.THREE),
    ]
    first = _state(
        [
            own,
            [_card(Suit.DIAMONDS, Rank.TWO), _card(Suit.DIAMONDS, Rank.THREE)],
            [_card(Suit.CLUBS, Rank.TWO), _card(Suit.CLUBS, Rank.THREE)],
            [_card(Suit.SPADES, Rank.TWO), _card(Suit.SPADES, Rank.THREE)],
        ],
        turn=0,
        tricks_played=1,
        tricks_won=[0, 0, 0, 1],
    )
    history_cards = [
        (0, _card(Suit.HEARTS, Rank.FOUR)),
        (1, _card(Suit.HEARTS, Rank.FIVE)),
        (2, _card(Suit.HEARTS, Rank.SIX)),
        (3, _card(Suit.HEARTS, Rank.SEVEN)),
    ]
    first.trick_history = [TrickRecord(cards=history_cards, winner=3, leader=0)]
    first.table_cards = [(3, _card(Suit.HEARTS, Rank.EIGHT))]
    first.trick_leader = 3
    first.spades_broken = True
    first.trump_broken = True

    second = copy.deepcopy(first)
    second.hands[0] = list(reversed(second.hands[0]))
    second.hands[1] = [
        _card(Suit.SPADES, Rank.ACE),
        _card(Suit.CLUBS, Rank.ACE),
    ]
    second.hands[2] = [
        _card(Suit.SPADES, Rank.KING),
        _card(Suit.CLUBS, Rank.KING),
    ]
    second.hands[3] = [
        _card(Suit.SPADES, Rank.QUEEN),
        _card(Suit.CLUBS, Rank.QUEEN),
    ]
    second.hand_bitsets = [cards_to_bitset(hand) for hand in second.hands]
    second.all_cards = list(reversed(second.all_cards))
    # Only the heart is legal while following the current heart lead.  The
    # legal set therefore stays fixed when the observer's off-suit card changes.
    return first, second, [own[0]], [own[0]]


def test_decision_seed_uses_visible_state_not_hidden_identities_or_order() -> None:
    first, second, first_legal, second_legal = _visible_state_pair()
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    first_seed = player._decision_seed(first, 0, first_legal)
    second_seed = player._decision_seed(second, 0, second_legal)

    assert first_seed == second_seed
    assert random.Random(first_seed).getstate() == random.Random(second_seed).getstate()


def test_decision_seed_changes_when_observer_hand_changes() -> None:
    first, _, first_legal, _ = _visible_state_pair()
    changed = copy.deepcopy(first)
    replacement = _card(Suit.SPADES, Rank.EIGHT)
    changed.hands[0][1] = replacement
    changed.hand_bitsets[0] = cards_to_bitset(changed.hands[0])
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    assert player._decision_seed(first, 0, first_legal) != player._decision_seed(
        changed,
        0,
        first_legal,
    )


def test_decision_seed_changes_when_public_table_changes() -> None:
    first, _, first_legal, _ = _visible_state_pair()
    changed = copy.deepcopy(first)
    changed.table_cards = [(3, _card(Suit.HEARTS, Rank.NINE))]
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    assert player._decision_seed(first, 0, first_legal) != player._decision_seed(
        changed,
        0,
        first_legal,
    )


def test_generate_proposal_is_canonical_across_deck_and_hand_order() -> None:
    observer_hand = list(_STANDARD_CARDS[:13])
    played = {0: [], 1: [], 2: [], 3: []}
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    first = player._generate_proposal(
        list(_STANDARD_CARDS),
        0,
        observer_hand,
        played,
        random.Random(20260717),
    )
    second = player._generate_proposal(
        list(reversed(_STANDARD_CARDS)),
        0,
        list(reversed(observer_hand)),
        played,
        random.Random(20260717),
    )

    assert [[card.card_id for card in hand] for hand in first] == [
        [card.card_id for card in hand] for hand in second
    ]


class _CapturePoolPlayer(RuleExactFirst4Player):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng_prefixes: list[tuple[float, ...]] = []

    def _build_is_pool(self, state, observer_id, rng, **kwargs):
        self.rng_prefixes.append(tuple(rng.random() for _ in range(4)))
        return [], []


class _ProposalSensitiveSolver:
    """Make the selected action depend on the sampled hidden-card allocation."""

    def __init__(self, first: Card, second: Card) -> None:
        self.first = first
        self.second = second
        self.marker_owners: list[int] = []

    def solve_with_q_fast(self, state: GameState) -> dict[int, float]:
        hidden = [
            (card.card_id, pid)
            for pid, hand in enumerate(state.hands)
            if pid != state.turn
            for card in hand
        ]
        marker_owner = min(hidden)[1]
        self.marker_owners.append(marker_owner)
        if marker_owner % 2:
            return {self.first.card_id: 2.0, self.second.card_id: 1.0}
        return {self.first.card_id: 1.0, self.second.card_id: 2.0}


def test_exact_play_restarts_the_same_random_stream_for_the_same_visible_state() -> None:
    legal = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.HEARTS, Rank.THREE),
    ]
    hands = [
        legal,
        [_card(Suit.DIAMONDS, Rank.TWO), _card(Suit.DIAMONDS, Rank.THREE)],
        [_card(Suit.CLUBS, Rank.TWO), _card(Suit.CLUBS, Rank.THREE)],
        [_card(Suit.SPADES, Rank.TWO), _card(Suit.SPADES, Rank.THREE)],
    ]
    state = _state(hands, turn=0, tricks_played=11)
    equivalent_state = copy.deepcopy(state)
    equivalent_state.hands[0] = list(reversed(equivalent_state.hands[0]))
    equivalent_state.hands[1] = [
        _card(Suit.SPADES, Rank.ACE),
        _card(Suit.CLUBS, Rank.ACE),
    ]
    equivalent_state.hands[2] = [
        _card(Suit.SPADES, Rank.KING),
        _card(Suit.CLUBS, Rank.KING),
    ]
    equivalent_state.hands[3] = [
        _card(Suit.SPADES, Rank.QUEEN),
        _card(Suit.CLUBS, Rank.QUEEN),
    ]
    equivalent_state.hand_bitsets = [
        cards_to_bitset(hand) for hand in equivalent_state.hands
    ]
    equivalent_state.all_cards = list(reversed(equivalent_state.all_cards))
    solver = _ProposalSensitiveSolver(legal[0], legal[1])
    budget = BudgetConfig(thresholds=[], default_top_k=1, default_max_samples=1)
    player = _CapturePoolPlayer(
        exact_solver=solver,
        hyperparam_config=HyperparamConfig(budget=budget),
        num_workers=1,
    )
    player.position = 0

    first_action = player._exact_play(copy.deepcopy(state), list(legal))
    second_action = player._exact_play(equivalent_state, list(reversed(legal)))

    assert player.rng_prefixes[0] == player.rng_prefixes[1]
    assert solver.marker_owners[0] == solver.marker_owners[1]
    expected = legal[0] if solver.marker_owners[0] % 2 else legal[1]
    assert first_action == second_action == expected


def test_exact_fallback_is_canonical_when_solver_is_unavailable() -> None:
    low = _card(Suit.HEARTS, Rank.TWO)
    high = _card(Suit.HEARTS, Rank.ACE)
    state = _state([[low, high], [], [], []], turn=0)
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)
    player.exact_solver = None

    assert player._exact_play(state, [high, low]) == low
    assert player._exact_play(state, [low, high]) == low


def test_exact_no_match_fallback_is_canonical() -> None:
    low = _card(Suit.HEARTS, Rank.TWO)
    high = _card(Suit.HEARTS, Rank.ACE)
    state = _state(
        [
            [low, high],
            [_card(Suit.CLUBS, Rank.TWO)],
            [_card(Suit.DIAMONDS, Rank.TWO)],
            [_card(Suit.SPADES, Rank.TWO)],
        ],
        turn=0,
        tricks_played=12,
    )
    budget = BudgetConfig(thresholds=[], default_top_k=1, default_max_samples=1)
    player = _CapturePoolPlayer(
        exact_solver=_RecordingSolver({}),
        hyperparam_config=HyperparamConfig(budget=budget),
        num_workers=1,
    )
    player.position = 0

    assert player._exact_play(copy.deepcopy(state), [high, low]) == low
    assert player._exact_play(copy.deepcopy(state), [low, high]) == low
    assert player.last_play_info == {"mode": "exact_no_match_fallback"}


def test_parallel_solver_worker_runs_in_clean_spawned_process() -> None:
    cards = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.HEARTS, Rank.THREE),
        _card(Suit.HEARTS, Rank.FOUR),
        _card(Suit.HEARTS, Rank.FIVE),
    ]
    state = _state(
        [[card] for card in cards],
        turn=0,
        tricks_played=12,
        tricks_won=[3, 3, 3, 3],
    )
    work_item = (state, 0, copy.deepcopy(state.hands))
    context = multiprocessing.get_context(_SOLVER_MP_START_METHOD)

    with context.Pool(1) as pool:
        results = pool.map(_parallel_solve_worker, [work_item])

    assert set(results[0]) == {cards[0].card_id}
