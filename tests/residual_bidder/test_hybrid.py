from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from residual_bidder.actions import BidAction, from_local_bid
from residual_bidder.cli.generate_hybrid import (
    _pack_worker_result,
    _unpack_worker_result,
)
from residual_bidder.hybrid import (
    generate_hybrid_deal,
    load_hybrid_npz,
    save_hybrid_npz,
    stack_hybrid_deals,
)
from residual_bidder.nsfp import NSFPObservation
from trick_taking.game_state import GameState, Phase


class _FakeNSFP:
    def __init__(self, centers: tuple[BidAction, ...] | None = None) -> None:
        self.centers = centers or (BidAction.BID_5,) * 4
        self.observed_prefixes: list[tuple[str | None, ...]] = []

    def observe(self, state: GameState) -> NSFPObservation:
        bid_index = len(state.bids)
        center = self.centers[bid_index]
        self.observed_prefixes.append(tuple(state.max_bid))
        encoded = torch.zeros(149, dtype=torch.float32)
        encoded[0] = float(state.current_bidder)
        encoded[1] = float(bid_index)
        scores = torch.arange(14, dtype=torch.float32) / 100.0
        scores[int(center)] = 10.0
        return NSFPObservation(
            encoded_149=encoded,
            raw_logits_16=torch.zeros(16, dtype=torch.float32),
            legal_scores_14=scores,
            center=center,
        )


class _BidDifferenceSolver:
    def __init__(self) -> None:
        self.calls = 0
        self.boundaries: list[tuple[int, tuple[int, ...]]] = []

    def solve(self, state: GameState) -> float:
        self.calls += 1
        assert state.phase is Phase.PLAYING
        assert state.tricks_played == 4
        assert state.table_cards == []
        assert tuple(len(hand) for hand in state.hands) == (9, 9, 9, 9)
        assert sum(len(hand) for hand in state.hands) == 36
        self.boundaries.append((state.tricks_played, tuple(len(hand) for hand in state.hands)))
        bid_indices = [int(from_local_bid(value)) for value in state.max_bid]
        team0 = bid_indices[0] + bid_indices[2]
        team1 = bid_indices[1] + bid_indices[3]
        return float((team0 - team1) * 100)


def test_one_deal_emits_four_rows_with_local_team_perspective_targets() -> None:
    nsfp = _FakeNSFP()
    solver = _BidDifferenceSolver()

    result = generate_hybrid_deal(202607210001, nsfp, solver)

    assert result.deal_id == "deal-202607210001"
    assert result.shuffle_seed == 202607210001
    assert result.solver_calls == 9
    assert solver.calls == 9
    assert len(result.rows) == 4
    assert {row.physical_seat for row in result.rows} == {0, 1, 2, 3}
    assert [row.bid_index for row in result.rows] == [0, 1, 2, 3]
    for row in result.rows:
        assert row.room_id == row.physical_seat % 2
        assert row.center is BidAction.BID_5
        assert row.features.shape == (167,)
        assert row.features.dtype == torch.float32
        assert row.mask.tolist() == [1.0, 1.0]
        assert row.targets.tolist() == pytest.approx([-1.0, 1.0])
        assert row.baseline_margin == 0.0


def test_boundary_centers_run_only_existing_local_alternatives() -> None:
    nsfp = _FakeNSFP(
        (BidAction.NIL, BidAction.BID_13, BidAction.NIL, BidAction.BID_13)
    )
    solver = _BidDifferenceSolver()

    result = generate_hybrid_deal(202607210002, nsfp, solver)

    assert result.solver_calls == 5
    assert solver.calls == 5
    for row in result.rows:
        if row.center is BidAction.NIL:
            assert row.mask.tolist() == [0.0, 1.0]
            assert row.targets[0].item() == 0.0
        else:
            assert row.center is BidAction.BID_13
            assert row.mask.tolist() == [1.0, 0.0]
            assert row.targets[1].item() == 0.0


def test_forced_branch_replays_frozen_continuation_from_changed_auction() -> None:
    nsfp = _FakeNSFP()
    solver = _BidDifferenceSolver()

    generate_hybrid_deal(202607210003, nsfp, solver)

    # The first alternative branch forces bid_4 at bid index zero.  Later NSFP
    # observations must see that changed prefix rather than the baseline bid_5.
    assert any(prefix.count("bid_4") == 1 for prefix in nsfp.observed_prefixes[4:])


def test_stacked_arrays_are_reproducible_and_deal_grouped() -> None:
    first = stack_hybrid_deals(
        [
            generate_hybrid_deal(seed, _FakeNSFP(), _BidDifferenceSolver())
            for seed in (202607210010, 202607210011)
        ]
    )
    second = stack_hybrid_deals(
        [
            generate_hybrid_deal(seed, _FakeNSFP(), _BidDifferenceSolver())
            for seed in (202607210010, 202607210011)
        ]
    )

    assert first.features.shape == (8, 167)
    assert first.targets.shape == (8, 2)
    assert first.masks.shape == (8, 2)
    for name in first.array_names():
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.deal_ids.tolist() == ["deal-202607210010"] * 4 + [
        "deal-202607210011"
    ] * 4


def test_npz_round_trip_uses_only_pickle_free_arrays(tmp_path: Path) -> None:
    deals = [generate_hybrid_deal(202607210020, _FakeNSFP(), _BidDifferenceSolver())]
    destination = tmp_path / "hybrid.npz"

    expected = save_hybrid_npz(destination, deals)
    loaded = load_hybrid_npz(destination)

    with np.load(destination, allow_pickle=False) as archive:
        assert set(archive.files) == set(expected.array_names())
        assert all(archive[name].dtype != object for name in archive.files)
    for name in expected.array_names():
        assert np.array_equal(getattr(expected, name), getattr(loaded, name))


def test_multiprocess_payload_contains_no_torch_storage() -> None:
    deal = generate_hybrid_deal(202607210030, _FakeNSFP(), _BidDifferenceSolver())

    payload = _pack_worker_result(deal)

    def assert_plain(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, (tuple, list)):
            for item in value:
                assert_plain(item)

    assert_plain(payload)
    restored = _unpack_worker_result(payload)
    expected = stack_hybrid_deals([deal])
    actual = stack_hybrid_deals([restored])
    assert restored.solver_calls == deal.solver_calls
    for name in expected.array_names():
        assert np.array_equal(getattr(expected, name), getattr(actual, name))
