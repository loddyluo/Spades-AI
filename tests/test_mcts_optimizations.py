"""
tests/test_mcts_optimizations.py

Purpose:
- Smoke tests to verify `TruncatedMCTSStrategy` behavior after performance
  optimizations (replace generic deepcopy with solver's lightweight copy).

Test conventions / I/O:
- Each test constructs a small `TruncatedMCTSConfig` with tiny simulation
  counts to run quickly.
- Inputs: none (tests build demo states internally).
- Outputs: assertions that the strategy methods run without raising and
  return values of expected types:
  - `choose_action(state)` -> `Card|None`
  - `play_full_game(state)` -> `list[Card]`

These tests do NOT assert numerical equivalence to previous outputs; they
ensure the code path remains functional and doesn't crash after changes.
"""

from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy, _build_demo_state


def test_choose_action_runs_quickly():
    cfg = TruncatedMCTSConfig(
        exact_threshold=24,
        leaf_threshold=24,
        simulations_per_action=1,
        determinization_count=1,
        mcts_determinization_count=1,
    )
    strat = TruncatedMCTSStrategy(cfg)
    state = _build_demo_state(seed=0)
    action = strat.choose_action(state)
    # action may be None or a Card; just ensure no exception and type is acceptable
    assert action is None or hasattr(action, "card_id")


def test_choose_action_does_not_mutate_state():
    cfg = TruncatedMCTSConfig(
        exact_threshold=24,
        leaf_threshold=24,
        simulations_per_action=1,
        determinization_count=1,
        mcts_determinization_count=1,
        use_determinization=False,
    )
    strat = TruncatedMCTSStrategy(cfg)
    state = _build_demo_state(seed=2)
    snapshot = (
        state.turn,
        state.trick_leader,
        state.played_bitset,
        tuple(len(hand) for hand in state.hands),
        tuple(tuple(card.card_id for card in hand) for hand in state.hands),
    )
    first = strat.choose_action(state)
    second = strat.choose_action(state)
    assert first == second
    assert snapshot == (
        state.turn,
        state.trick_leader,
        state.played_bitset,
        tuple(len(hand) for hand in state.hands),
        tuple(tuple(card.card_id for card in hand) for hand in state.hands),
    )


def test_play_full_game_runs():
    cfg = TruncatedMCTSConfig(
        exact_threshold=10,
        leaf_threshold=6,
        simulations_per_action=1,
        determinization_count=1,
        mcts_determinization_count=1,
    )
    strat = TruncatedMCTSStrategy(cfg)
    state = _build_demo_state(seed=1)
    seq = strat.play_full_game(state)
    assert isinstance(seq, list)
    # each element should be Card-like
    for c in seq:
        assert hasattr(c, "card_id")
