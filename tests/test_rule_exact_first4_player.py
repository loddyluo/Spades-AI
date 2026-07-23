from __future__ import annotations

import copy
import itertools
import multiprocessing
import random
from collections import Counter
from contextlib import nullcontext

import pytest
import torch

import strategy.rule_exact_first4_player as rule_exact_module
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


def _four_trick_replay_fixture(
) -> tuple[list[list[Card]], list[tuple[int, Card]], Card]:
    clubs_low = [
        _card(Suit.CLUBS, rank)
        for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)
    ]
    clubs_high = [
        _card(Suit.CLUBS, rank)
        for rank in (Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE)
    ]
    diamonds = [
        _card(Suit.DIAMONDS, rank)
        for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)
    ]
    hearts = [
        _card(Suit.HEARTS, rank)
        for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)
    ]
    appended = _card(Suit.HEARTS, Rank.SIX)
    hands = [
        [clubs_low[p], clubs_high[p], diamonds[p], hearts[p]]
        for p in range(4)
    ]
    hands[0].append(appended)
    sequence = [
        *[(p, clubs_low[p]) for p in range(4)],
        *[(p, diamonds[p]) for p in range(4)],
        *[(p, hearts[p]) for p in range(4)],
        *[(p, clubs_high[p]) for p in range(4)],
    ]
    return hands, sequence, appended


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


def test_importance_replay_snapshot_evaluates_only_appended_actions() -> None:
    hands, prefix, appended = _four_trick_replay_fixture()
    solver = _RecordingSolver()
    player = RuleExactFirst4Player(
        exact_solver=solver,
        hyperparam_config=HyperparamConfig(trick_num_threshold=0),
        num_workers=1,
    )
    state = _state(hands, turn=0)

    prefix_weight, snapshot = (
        player._compute_importance_weight_with_snapshot(
            hands,
            prefix,
            bid_prod=1.0,
            original_state=state,
            observer_id=0,
        )
    )

    assert prefix_weight > 0.0
    assert snapshot is not None
    assert len(solver.snapshots) == len(prefix)

    extended_weight, extended_snapshot = (
        player._compute_importance_weight_with_snapshot(
            hands,
            prefix + [(0, appended)],
            bid_prod=1.0,
            original_state=state,
            observer_id=0,
            replay_snapshot=snapshot,
        )
    )

    assert extended_weight > 0.0
    assert extended_snapshot is not None
    assert len(solver.snapshots) == len(prefix) + 1

    full_replay_player = RuleExactFirst4Player(
        exact_solver=_RecordingSolver(),
        hyperparam_config=HyperparamConfig(trick_num_threshold=0),
        num_workers=1,
    )
    full_replay_weight = full_replay_player._compute_importance_weight(
        hands,
        prefix + [(0, appended)],
        bid_prod=1.0,
        original_state=state,
        observer_id=0,
    )
    assert extended_weight == pytest.approx(full_replay_weight)


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


def test_conditional_proposal_sampler_is_uniform_over_valid_deals() -> None:
    pool = [
        _card(Suit.CLUBS, Rank.TWO),
        _card(Suit.CLUBS, Rank.THREE),
        _card(Suit.DIAMONDS, Rank.TWO),
        _card(Suit.HEARTS, Rank.TWO),
    ]
    used = [
        card for card in _STANDARD_CARDS
        if card not in pool
    ]
    observer_hand = used[:13]
    played = {
        0: [],
        1: used[13:25],
        2: used[25:37],
        3: used[37:48],
    }
    void_suits = {
        1: {Suit.CLUBS},
        2: {Suit.DIAMONDS},
        3: {Suit.HEARTS},
    }
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)
    context = player._prepare_proposal_sampler(
        list(_STANDARD_CARDS),
        0,
        observer_hand,
        played,
        void_suits,
    )

    valid_assignments = []
    for owners in itertools.product((1, 2, 3), repeat=len(pool)):
        if Counter(owners) != Counter({1: 1, 2: 1, 3: 2}):
            continue
        if any(
            card.suit in void_suits[owner]
            for card, owner in zip(pool, owners)
        ):
            continue
        valid_assignments.append(owners)
    assert context.total_completions == len(valid_assignments)

    rng = random.Random(20260723)
    counts: Counter[tuple[int, ...]] = Counter()
    for _ in range(6000):
        proposal = player._generate_proposal(
            list(_STANDARD_CARDS),
            0,
            observer_hand,
            played,
            rng,
            void_suits=void_suits,
            sampler_context=context,
        )
        owner_by_card = {
            card.card_id: owner
            for owner in (1, 2, 3)
            for card in proposal[owner]
            if card in pool
        }
        assignment = tuple(
            owner_by_card[card.card_id] for card in pool
        )
        counts[assignment] += 1

    assert set(counts) == set(valid_assignments)
    expected = 6000 / len(valid_assignments)
    assert all(
        abs(count - expected) < expected * 0.12
        for count in counts.values()
    )


@pytest.mark.parametrize(
    "bids",
    [
        ["bid_2", "bid_3", "bid_4", "bid_5"],
        ["bid_2", "bid_3", "nil", "bid_5"],
    ],
)
def test_bitset_batch_replay_matches_scalar_replay(
    bids: list[str],
) -> None:
    initial_hands, sequence, _ = _four_trick_replay_fixture()
    proposals = [
        copy.deepcopy(initial_hands),
        copy.deepcopy(initial_hands),
    ]
    bid_prods = [0.25, 0.75]
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)

    batch = player._compute_batch_replay_weights(
        proposals,
        sequence,
        bid_prods,
        bids,
        observer_id=0,
    )
    scalar = [
        player._compute_importance_weight_with_snapshot(
            proposal,
            sequence,
            max_bid=bids,
            bid_prod=bid_prod,
            observer_id=0,
        )
        for proposal, bid_prod in zip(proposals, bid_prods)
    ]

    for (batch_weight, batch_snapshot), (
        scalar_weight,
        scalar_snapshot,
    ) in zip(batch, scalar):
        assert batch_weight == pytest.approx(scalar_weight)
        assert batch_snapshot == scalar_snapshot


def test_replay_skips_state_copy_before_solver_weighting_window(
    monkeypatch,
) -> None:
    initial_hands, sequence, _ = _four_trick_replay_fixture()
    player = RuleExactFirst4Player(
        exact_solver=object(),
        hyperparam_config=HyperparamConfig(trick_num_threshold=8),
        num_workers=1,
    )

    def fail_copy(_value):
        raise AssertionError("deepcopy should not run before trick 9")

    monkeypatch.setattr(rule_exact_module.copy, "deepcopy", fail_copy)
    weight, snapshot = player._compute_importance_weight_with_snapshot(
        initial_hands,
        sequence,
        bid_prod=1.0,
        original_state=_state(initial_hands),
        observer_id=0,
    )

    assert weight > 0.0
    assert snapshot is not None


def test_solver_pool_uses_single_item_chunks(monkeypatch) -> None:
    calls = []

    class FakePool:
        def map(self, function, items, chunksize):
            calls.append((function, list(items), chunksize))
            return [{} for _ in items]

    entry = rule_exact_module._PersistentSolverPool(
        pool=FakePool(),
        map_lock=nullcontext(),
        owner_pid=0,
    )
    monkeypatch.setattr(
        rule_exact_module,
        "_get_persistent_solver_pool",
        lambda _workers: entry,
    )

    items = [(object(), 0, [])]
    assert rule_exact_module._map_persistent_solver_pool(3, items) == [{}]
    assert calls == [
        (
            rule_exact_module._exact_solver_worker.parallel_solve_worker,
            items,
            1,
        )
    ]


class _CapturePoolPlayer(RuleExactFirst4Player):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng_prefixes: list[tuple[float, ...]] = []

    def _build_is_pool(self, state, observer_id, rng, **kwargs):
        self.rng_prefixes.append(tuple(rng.random() for _ in range(4)))
        return [], []


class _StaticPoolPlayer(RuleExactFirst4Player):
    def __init__(
        self,
        pool_hands: list[list[list[Card]]],
        pool_weights: list[float],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pool_hands = pool_hands
        self.pool_weights = pool_weights

    def _build_is_pool(self, state, observer_id, rng, **kwargs):
        self._last_pool_cache_hit = False
        return self.pool_hands, self.pool_weights


class _FixedProposalPlayer(RuleExactFirst4Player):
    def __init__(self, proposal: list[list[Card]]) -> None:
        super().__init__(exact_solver=object(), num_workers=1)
        self.proposal = proposal

    def _generate_proposal(self, *args, **kwargs) -> list[list[Card]]:
        return copy.deepcopy(self.proposal)

    def _compute_batch_bid_prods(
        self,
        proposals: list[list[list[Card]]],
        max_bid: list[str],
    ) -> list[float]:
        return [1.0] * len(proposals)


class _CountingFixedProposalPlayer(_FixedProposalPlayer):
    def __init__(self, proposal: list[list[Card]]) -> None:
        self.generate_calls = 0
        self.bid_batch_calls = 0
        super().__init__(proposal)

    def _generate_proposal(self, *args, **kwargs) -> list[list[Card]]:
        self.generate_calls += 1
        return super()._generate_proposal(*args, **kwargs)

    def _compute_batch_bid_prods(
        self,
        proposals: list[list[list[Card]]],
        max_bid: list[str],
    ) -> list[float]:
        self.bid_batch_calls += 1
        return [1.0] * len(proposals)


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


class _CountingBidEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, hand, previous_bids, position):
        self.calls += 1
        return torch.tensor(
            [float(len(hand)), float(len(previous_bids)), float(position)],
            dtype=torch.float32,
        )


class _CountingBidModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, features):
        self.calls += 1
        return torch.zeros((features.shape[0], 16), dtype=torch.float32)


def test_batch_bid_likelihood_deduplicates_and_caches_hand_features() -> None:
    proposal = [
        [_card(Suit.CLUBS, Rank.TWO)],
        [_card(Suit.DIAMONDS, Rank.THREE)],
        [_card(Suit.HEARTS, Rank.FOUR)],
        [_card(Suit.SPADES, Rank.FIVE)],
    ]
    encoder = _CountingBidEncoder()
    model = _CountingBidModel()
    player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)
    player._bid_encoder_is = encoder
    player._bid_model_is = model
    player._bid_device_is = "cpu"
    bids = ["bid_2", "bid_3", "bid_4", "bid_5"]

    first = player._compute_batch_bid_prods(
        [copy.deepcopy(proposal), copy.deepcopy(proposal)],
        bids,
    )
    second = player._compute_batch_bid_prods([copy.deepcopy(proposal)], bids)

    assert first == pytest.approx([second[0], second[0]])
    assert encoder.calls == 4
    assert model.calls == 1


def test_current_table_rank_blocks_equal_magnitude_enforcement() -> None:
    high = _card(Suit.HEARTS, Rank.QUEEN)
    low = _card(Suit.HEARTS, Rank.TEN)
    table = _card(Suit.HEARTS, Rank.JACK)
    hands = [
        [high, low],
        [_card(Suit.HEARTS, Rank.TWO), _card(Suit.CLUBS, Rank.TWO)],
        [_card(Suit.HEARTS, Rank.THREE), _card(Suit.CLUBS, Rank.THREE)],
        [_card(Suit.CLUBS, Rank.FOUR)],
    ]
    state = _state(hands, turn=0, tricks_played=11)
    state.table_cards = [(3, table)]
    state.trick_leader = 3
    state.played_bitset = table.bit

    budget = BudgetConfig(thresholds=[], default_top_k=1, default_max_samples=1)
    player = _CapturePoolPlayer(
        exact_solver=_RecordingSolver(
            {high.card_id: -8.0, low.card_id: 28.0}
        ),
        hyperparam_config=HyperparamConfig(budget=budget),
        num_workers=1,
    )
    player.position = 0

    assert player._exact_play(state, [high, low]) == low


def test_current_trick_rank_does_not_trigger_bad_equal_magnitude_weight() -> None:
    def replay_weight(lead_rank: Rank) -> float:
        c2, c3, c4, c5 = [
            _card(Suit.CLUBS, rank)
            for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)
        ]
        c6, c7, c8, c9 = [
            _card(Suit.CLUBS, rank)
            for rank in (Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE)
        ]
        d2, d3, d4, d5 = [
            _card(Suit.DIAMONDS, rank)
            for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)
        ]
        d6, d7, d8, d9 = [
            _card(Suit.DIAMONDS, rank)
            for rank in (Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE)
        ]
        lead = _card(Suit.HEARTS, lead_rank)
        h2 = _card(Suit.HEARTS, Rank.TWO)
        h3 = _card(Suit.HEARTS, Rank.THREE)
        ht = _card(Suit.HEARTS, Rank.TEN)
        hq = _card(Suit.HEARTS, Rank.QUEEN)
        s2 = _card(Suit.SPADES, Rank.TWO)
        s3 = _card(Suit.SPADES, Rank.THREE)
        s4 = _card(Suit.SPADES, Rank.FOUR)
        hands = [
            [c2, d3, c8, d9, lead, s2],
            [c3, d4, c9, d6, h2, s3],
            [c4, d5, c6, d7, hq, ht],
            [c5, d2, c7, d8, h3, s4],
        ]
        sequence = [
            (0, c2), (1, c3), (2, c4), (3, c5),
            (3, d2), (0, d3), (1, d4), (2, d5),
            (2, c6), (3, c7), (0, c8), (1, c9),
            (1, d6), (2, d7), (3, d8), (0, d9),
            (0, lead), (1, h2), (2, ht),
        ]
        player = RuleExactFirst4Player(exact_solver=object(), num_workers=1)
        return player._compute_importance_weight(
            hands,
            sequence,
            bid_prod=1.0,
            observer_id=0,
        )

    # J♥ lies between Q♥ and T♥ but is still on the current table.  It must
    # block the equal-magnitude group exactly like any other outstanding rank.
    assert replay_weight(Rank.JACK) == pytest.approx(replay_weight(Rank.NINE))


def test_build_is_pool_forwards_nil_bids_to_replay_weighting(
    tmp_path,
    monkeypatch,
) -> None:
    low = _card(Suit.HEARTS, Rank.TWO)
    high = _card(Suit.HEARTS, Rank.ACE)
    proposal = [
        [_card(Suit.CLUBS, Rank.TWO)],
        [_card(Suit.CLUBS, Rank.THREE)],
        [low, high],
        [_card(Suit.CLUBS, Rank.FOUR)],
    ]
    state = _state(
        [proposal[0], proposal[1], [high], proposal[3]],
        turn=3,
    )
    state.max_bid = ["bid_2", "bid_2", "nil", "bid_2"]
    state.table_cards = [(2, low)]
    state.trick_leader = 2
    state.played_bitset = low.bit
    player = _FixedProposalPlayer(proposal)

    # _build_is_pool writes its diagnostics in the current directory.
    monkeypatch.chdir(tmp_path)
    _, weights = player._build_is_pool(
        state,
        observer_id=0,
        rng=random.Random(1),
        num_proposals=1,
        num_proposals_limit=1,
        min_pool_size=1,
    )

    # A nil bidder with H2/HA correctly leads H2.  If max_bid is dropped, the
    # normal first-four rule expects HA instead and incorrectly applies ×0.81.
    assert weights == [pytest.approx(1.0)]


def test_build_is_pool_reuses_posterior_for_extended_history(
    tmp_path,
    monkeypatch,
) -> None:
    initial_hands, prefix, appended = _four_trick_replay_fixture()

    def state_for(sequence: list[tuple[int, Card]]) -> GameState:
        remaining_hands = [list(hand) for hand in initial_hands]
        for player_id, card in sequence:
            remaining_hands[player_id].remove(card)
        state = _state(
            remaining_hands,
            turn=0,
            tricks_played=len(sequence) // 4,
        )
        state.trick_history = [
            TrickRecord(
                cards=list(sequence[offset:offset + 4]),
                winner=3,
                leader=sequence[offset][0],
            )
            for offset in range(0, len(sequence) - len(sequence) % 4, 4)
        ]
        tail_start = len(state.trick_history) * 4
        state.table_cards = list(sequence[tail_start:])
        state.played_bitset = cards_to_bitset(
            [card for _, card in sequence]
        )
        return state

    player = _CountingFixedProposalPlayer(initial_hands)
    player.config.trick_num_threshold = 99
    player.start_game(0, initial_hands[0], 4)
    monkeypatch.chdir(tmp_path)

    first_hands, first_weights = player._build_is_pool(
        state_for(prefix),
        observer_id=0,
        rng=random.Random(1),
        num_proposals=1,
        num_proposals_limit=1,
        min_pool_size=1,
    )

    # backend.py calls start_game on every HTTP decision.  The identical deal
    # key must preserve the posterior while the history-prefix check protects
    # against a different or rewound game.
    player.start_game(0, initial_hands[0], 4)
    second_hands, second_weights = player._build_is_pool(
        state_for(prefix + [(0, appended)]),
        observer_id=0,
        rng=random.Random(2),
        num_proposals=1,
        num_proposals_limit=1,
        min_pool_size=1,
    )

    assert first_hands == second_hands == [initial_hands]
    assert first_weights and second_weights
    assert player.generate_calls == 1
    assert player.bid_batch_calls == 1
    assert player._last_pool_cache_hit is True

    player._build_is_pool(
        state_for(prefix),
        observer_id=0,
        rng=random.Random(3),
        num_proposals=1,
        num_proposals_limit=1,
        min_pool_size=1,
    )

    assert player.generate_calls == 2
    assert player.bid_batch_calls == 2
    assert player._last_pool_cache_hit is False


def test_exact_play_restarts_the_same_random_stream_for_the_same_visible_state() -> None:
    legal = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.HEARTS, Rank.FOUR),
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


@pytest.mark.parametrize(
    ("swap_is_fill", "top_k"),
    [
        (True, 3),
        (False, 7),
    ],
)
def test_max_samples_is_final_solver_sample_cap(
    swap_is_fill: bool,
    top_k: int,
) -> None:
    legal = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.HEARTS, Rank.FOUR),
    ]
    state = _state(
        [
            legal,
            [_card(Suit.CLUBS, Rank.TWO)],
            [_card(Suit.DIAMONDS, Rank.TWO)],
            [_card(Suit.SPADES, Rank.TWO)],
        ],
        turn=0,
        tricks_played=11,
    )
    proposals = [
        [
            list(legal),
            [_card(Suit.SPADES, Rank(rank_value))],
            [_card(Suit.CLUBS, Rank.TWO)],
            [_card(Suit.DIAMONDS, Rank.TWO)],
        ]
        for rank_value in range(Rank.TWO.value, Rank.JACK.value + 1)
    ]
    solver = _RecordingSolver(
        {legal[0].card_id: 0.0, legal[1].card_id: 1.0}
    )
    budget = BudgetConfig(
        thresholds=[],
        default_top_k=top_k,
        default_max_samples=5,
    )
    player = _StaticPoolPlayer(
        proposals,
        [float(len(proposals) - i) for i in range(len(proposals))],
        exact_solver=solver,
        hyperparam_config=HyperparamConfig(
            budget=budget,
            swap_is_fill=swap_is_fill,
        ),
        num_workers=1,
        debug=True,
    )
    player.position = 0

    player._exact_play(state, legal)

    assert len(solver.snapshots) == 5
    assert player.last_play_info["samples"] == 5


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


def test_worker_solver_is_initialized_once_per_process(monkeypatch) -> None:
    created = []

    class FakeSolver:
        pass

    def create_solver():
        solver = FakeSolver()
        created.append(solver)
        return solver

    monkeypatch.setattr(
        rule_exact_module,
        "ExactDoubleDummyCppFastestSolver",
        create_solver,
    )
    monkeypatch.setattr(rule_exact_module, "_WORKER_SOLVER", None)

    first = rule_exact_module._get_worker_solver()
    second = rule_exact_module._get_worker_solver()

    assert first is second
    assert created == [first]


def test_persistent_solver_pool_is_reused(monkeypatch) -> None:
    created_pools = []

    class FakePool:
        def __init__(self, workers, initializer):
            self.workers = workers
            self.initializer = initializer
            self.terminated = False
            self.joined = False

        def terminate(self):
            self.terminated = True

        def join(self):
            self.joined = True

    class FakeContext:
        def Pool(self, workers, initializer):
            pool = FakePool(workers, initializer)
            created_pools.append(pool)
            return pool

    monkeypatch.setattr(rule_exact_module, "_SOLVER_POOLS", {})
    monkeypatch.setattr(
        rule_exact_module.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )

    first = rule_exact_module._get_persistent_solver_pool(3)
    second = rule_exact_module._get_persistent_solver_pool(3)

    assert first is second
    assert len(created_pools) == 1
    assert created_pools[0].workers == 3
    assert (
        created_pools[0].initializer
        is rule_exact_module._exact_solver_worker.initialize_solver_worker
    )

    rule_exact_module._shutdown_persistent_solver_pools()
    assert created_pools[0].terminated is True
    assert created_pools[0].joined is True
