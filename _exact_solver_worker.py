"""Top-level Torch-free worker for determinized exact solves."""

from __future__ import annotations

import copy
from typing import Any

from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


_WORKER_SOLVER: ExactDoubleDummyCppFastestSolver | None = None


def initialize_solver_worker() -> None:
    """Load and validate one native solver per spawned process."""
    global _WORKER_SOLVER
    _WORKER_SOLVER = ExactDoubleDummyCppFastestSolver()


def _get_worker_solver() -> ExactDoubleDummyCppFastestSolver:
    global _WORKER_SOLVER
    if _WORKER_SOLVER is None:
        initialize_solver_worker()
    assert _WORKER_SOLVER is not None
    return _WORKER_SOLVER


def solve_proposal(
    args: tuple,
    solver: Any,
) -> dict[int, float]:
    """Apply one initial-hand proposal and return native root Q values."""
    state, observer_id, hand_proposal = args
    sim_state = copy.deepcopy(state)

    played_by: dict[int, set[int]] = {player: set() for player in range(4)}
    for record in sim_state.trick_history:
        for player, card in record.cards:
            played_by[player].add(card.card_id)
    for player, card in sim_state.table_cards:
        played_by[player].add(card.card_id)

    for player in range(4):
        if player == observer_id:
            continue
        sim_state.hands[player] = [
            card
            for card in hand_proposal[player]
            if card.card_id not in played_by[player]
        ]

    if hasattr(sim_state, "hand_bitsets"):
        sim_state.hand_bitsets = [
            sum(1 << card.card_id for card in hand)
            for hand in sim_state.hands
        ]
    return solver.solve_with_q_fast(sim_state)


def solve_proposal_safely(
    args: tuple,
    solver: Any,
) -> dict[int, float]:
    try:
        return solve_proposal(args, solver)
    except Exception:
        return {}


def parallel_solve_worker(args: tuple) -> dict[int, float]:
    """Spawn-safe entry point; this module deliberately does not import Torch."""
    try:
        solver = _get_worker_solver()
    except Exception:
        return {}
    return solve_proposal_safely(args, solver)
