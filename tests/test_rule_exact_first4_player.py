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
    """Counts encoded *rows*; the production path encodes each row once, batched."""

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def encode_indices_batch(self, hand_indices, bid_slots, positions):
        self.calls += 1
        self.rows += len(positions)
        return torch.tensor(
            [
                [
                    float(len(hand)),
                    float(len([slot for slot in slots if slot >= 0])),
                    float(position),
                ]
                for hand, slots, position in zip(hand_indices, bid_slots, positions)
            ],
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
    # Four distinct (bids, seat, hand) rows are encoded once in total: the
    # first call encodes all four, the second one is served by the LRU cache,
    # and the model still runs exactly one batched forward.
    assert encoder.rows == 4
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


def test_uniform_determinization_fallback_records_per_proposal_q() -> None:
    """IS 池退化到均匀 determinization 时，逐提案 Q 也必须留下来。

    后 9 墩靠后的决策（剩余牌很少）会走到这条兜底分支；如果只记期望值，
    完整复盘的「反推终局分布」在这些决策上就没有原料可用。
    """
    hands = [
        [_card(Suit.SPADES, Rank.ACE), _card(Suit.HEARTS, Rank.TWO)],
        [_card(Suit.SPADES, Rank.KING), _card(Suit.HEARTS, Rank.THREE)],
        [_card(Suit.SPADES, Rank.QUEEN), _card(Suit.HEARTS, Rank.FOUR)],
        [_card(Suit.SPADES, Rank.JACK), _card(Suit.HEARTS, Rank.FIVE)],
    ]
    state = _state(hands, turn=0)
    budget = BudgetConfig(thresholds=[], default_top_k=1, default_max_samples=3)
    player = RuleExactFirst4Player(
        exact_solver=_RecordingSolver({card.card_id: 12.0 for card in hands[0]}),
        hyperparam_config=HyperparamConfig(budget=budget),
        num_workers=1,
    )
    player.position = 0
    player.collect_action_q = True
    # 强制走「IS 池为空 → 均匀 determinization」这条分支。
    player._build_is_pool = lambda *args, **kwargs: ([], [])

    player._exact_play(state, list(hands[0]))

    info = player.last_play_info
    assert info["mode"] == "exact_is_determinized"
    samples = info["proposal_samples"]
    assert len(samples) == 3, "fallback 下每份均匀采样都要留一条提案"
    assert all(abs(entry["weight"] - 1 / 3) < 1e-12 for entry in samples)
    assert all(
        set(entry["q"]) == {"AS", "2H"} for entry in samples
    ), "提案里必须按牌码给出每个合法动作的 Q"
    assert info["expected_q"] == {"AS": 12.0, "2H": 12.0}


def test_equal_magnitude_representative_maps_to_the_group_max() -> None:
    """求解器只保留等大牌张组内最大的一张，其余要能映射回代表牌。"""
    hearts = Suit.HEARTS

    def heart(rank: Rank) -> Card:
        return _card(hearts, rank)

    hand = [heart(Rank.ACE), heart(Rank.KING), heart(Rank.QUEEN)]
    rep = RuleExactFirst4Player._equal_magnitude_representative
    assert rep(heart(Rank.QUEEN), hand, {}) == heart(Rank.ACE)
    assert rep(heart(Rank.KING), hand, {}) == heart(Rank.ACE)
    # 已经是组内最大 → 无需替换
    assert rep(heart(Rank.ACE), hand, {}) is None
    # 同花色只有一张 → 不构成等大组
    assert rep(heart(Rank.ACE), [heart(Rank.ACE), _card(Suit.SPADES, Rank.TWO)], {}) is None
    # 中间的点数还在别人手里 → 不等大
    sparse = [heart(Rank.ACE), heart(Rank.QUEEN)]
    assert rep(heart(Rank.QUEEN), sparse, {}) is None
    # 中间的点数已经打出 → 等大
    assert rep(heart(Rank.QUEEN), sparse, {hearts: {Rank.KING.value}}) == heart(Rank.ACE)
    # 牌不在手里
    assert rep(_card(hearts, Rank.TWO), hand, {}) is None


def test_merged_card_maps_back_to_the_representative_that_used_to_zero_the_pool() -> None:
    """第 9 墩起：打出等大牌张里较小的一张，不该把整条提案的权重清零。

    求解器的根动作表只留组内最大的牌（等大牌张过滤），所以查 `action_q` 会
    落空。旧行为据此把提案作废，IS 池随之清零、静默退回均匀 determinization；
    现在先映射回代表牌，查得到就照常判"最优 / 非最优"。
    """
    hearts = Suit.HEARTS
    ace, queen = _card(hearts, Rank.ACE), _card(hearts, Rank.QUEEN)
    rep = RuleExactFirst4Player._equal_magnitude_representative

    # A 与 Q 之间只隔着 K：K 还在别人手里时两者不等大，查不到就不是等大牌。
    assert rep(queen, [ace, queen], {}) is None
    # K 已打出时两者等大，Q 必须映射到求解器真正保留的 A。
    assert rep(queen, [ace, queen], {hearts: {Rank.KING.value}}) == ace


class _StubBatchSolver:
    """只提供批处理路径需要的两个钩子，Q 值由测试注入。"""

    def __init__(self, q_values: dict[int, float]) -> None:
        self.q_values = dict(q_values)

    @staticmethod
    def _bid_to_native(value):
        if value is None or value == "nil":
            return 0
        if value == "blind_nil":
            return 14
        if isinstance(value, str) and value.startswith("bid_"):
            return int(value.split("_")[1])
        return int(value) if isinstance(value, int) else 0

    def pack_native_payload(self, *args, **kwargs) -> tuple:
        return ()

    def solve_native_with_q_payload(self, payload):  # noqa: D401 - hasattr 探针
        return dict(self.q_values)


class _BatchWeightPlayer(RuleExactFirst4Player):
    """把整批求解请求换成固定 Q 表，其余逻辑保持原样。"""

    def __init__(self, q_values: dict[int, float], penalty: float) -> None:
        budget = BudgetConfig(thresholds=[], default_top_k=1, default_max_samples=1)
        super().__init__(
            exact_solver=_StubBatchSolver(q_values),
            hyperparam_config=HyperparamConfig(
                budget=budget,
                bad_action_penalty_factor=penalty,
                bad_action_weight="1.0",      # 求解器那一步固定 ×1，便于隔离
                trick_num_threshold=4,        # 步 16 起进入求解器判定
            ),
            num_workers=1,
        )

    def _solve_solver_payloads(self, payloads):
        return [dict(self.exact_solver.q_values) for _ in payloads]


def _teammate_step_scenario(teammate_card: Card):
    """构造 17 步公开历史：第 16 步由搭档领出 `teammate_card`。

    前 16 步恰好 4 墩，其中 ♣7 已经被打出并进入已完成墩，于是搭档手里的
    ♣8 与 ♣6 构成「等大牌张」组，组内最大的是 ♣8。
    """
    def hearts(*ranks): return [_card(Suit.HEARTS, r) for r in ranks]
    def diamonds(*ranks): return [_card(Suit.DIAMONDS, r) for r in ranks]
    def clubs(*ranks): return [_card(Suit.CLUBS, r) for r in ranks]

    hands = [
        hearts(Rank.TWO, Rank.SIX) + diamonds(Rank.TWO, Rank.SIX) + clubs(Rank.TWO, Rank.SIX),
        hearts(Rank.THREE, Rank.SEVEN) + diamonds(Rank.THREE, Rank.SEVEN) + clubs(Rank.THREE),
        hearts(Rank.FOUR, Rank.EIGHT) + diamonds(Rank.FOUR, Rank.NINE) + clubs(Rank.SEVEN, Rank.EIGHT, Rank.SIX),
        hearts(Rank.FIVE, Rank.NINE) + diamonds(Rank.FIVE, Rank.TEN) + clubs(Rank.FIVE),
    ]
    sequence = [
        (0, _card(Suit.HEARTS, Rank.TWO)), (1, _card(Suit.HEARTS, Rank.THREE)),
        (2, _card(Suit.HEARTS, Rank.FOUR)), (3, _card(Suit.HEARTS, Rank.FIVE)),
        (1, _card(Suit.DIAMONDS, Rank.THREE)), (2, _card(Suit.DIAMONDS, Rank.FOUR)),
        (3, _card(Suit.DIAMONDS, Rank.FIVE)), (0, _card(Suit.DIAMONDS, Rank.TWO)),
        (2, _card(Suit.CLUBS, Rank.SEVEN)), (3, _card(Suit.CLUBS, Rank.FIVE)),
        (0, _card(Suit.CLUBS, Rank.TWO)), (1, _card(Suit.CLUBS, Rank.THREE)),
        (3, _card(Suit.HEARTS, Rank.NINE)), (0, _card(Suit.HEARTS, Rank.SIX)),
        (1, _card(Suit.HEARTS, Rank.SEVEN)), (2, _card(Suit.HEARTS, Rank.EIGHT)),
        (2, teammate_card),
    ]
    teams = [0, 1, 0, 1]
    state = _state(hands, turn=2, tricks_played=4, tricks_won=[1, 1, 1, 1])
    state.teams = teams
    return hands, sequence, state


def _teammate_step_weight(teammate_card: Card, penalty: float) -> float:
    hands, sequence, state = _teammate_step_scenario(teammate_card)
    ace_like_q = {teammate_card.card_id: 10.0}
    # 让求解器认为「组内最大那张」是该队最优动作：Q 规则于是给 ×1。
    best = _card(teammate_card.suit, Rank.EIGHT) if teammate_card.rank is Rank.SIX else teammate_card
    player = _BatchWeightPlayer({best.card_id: 10.0, teammate_card.card_id: 0.0}, penalty)
    results = player._compute_importance_weights_slow_batch(
        [hands],
        sequence,
        [1.0],
        ["bid_2"] * 4,
        0,          # observer_id = 0 → 搭档是座位 2
        state,
    )
    assert results, "批处理路径应当返回结果"
    return results[0][0]


def test_teammate_non_max_equal_card_is_penalised_at_solver_steps() -> None:
    """第 16 步起（同一条 >=16 的守卫覆盖第 9 墩之后）搭档打了等大牌张里
    较小的一张时，在求解器给的 ×x / ×1 之外还要再乘 bad_action_penalty_factor。
    """
    six = _card(Suit.CLUBS, Rank.SIX)      # 等大组 {♣8, ♣6} 里的较小者
    eight = _card(Suit.CLUBS, Rank.EIGHT)  # 组内最大者

    # 同一局面只换这一步出的牌，其余（包括前四墩复现带来的惩罚）完全相同，
    # 比值就精确隔离出第 16 步的等大牌张惩罚。
    penalised = _teammate_step_weight(six, 0.81) / _teammate_step_weight(eight, 0.81)
    assert penalised == pytest.approx(0.81)

    # 把系数设成 1.0（等于关掉这条惩罚）时，两者应当完全相等。
    neutral = _teammate_step_weight(six, 1.0) / _teammate_step_weight(eight, 1.0)
    assert neutral == pytest.approx(1.0)
