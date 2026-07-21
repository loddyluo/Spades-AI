from __future__ import annotations

import dataclasses
import inspect
import math

import pytest

from residual_bidder import random_tape
from residual_bidder.random_tape import BidSamplingKey, policy_uniform


def _key(**changes: object) -> BidSamplingKey:
    values: dict[str, object] = {
        "policy_seed": 123456789,
        "deal_id": "deal-α",
        "room_id": "room-B",
        "logical_seat": 2,
        "bid_index": 3,
    }
    values.update(changes)
    return BidSamplingKey(**values)  # type: ignore[arg-type]


def test_policy_uniform_has_an_exact_canonical_blake2b_fixture() -> None:
    key = _key()

    assert policy_uniform(key) == 0.2859608005581722
    assert policy_uniform(dataclasses.replace(key)) == policy_uniform(key)
    assert 0.0 < policy_uniform(key) < 1.0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("policy_seed", 123456790),
        ("deal_id", "deal-β"),
        ("room_id", "room-C"),
        ("logical_seat", 3),
        ("bid_index", 2),
    ],
)
def test_policy_uniform_separates_every_canonical_field(
    field: str, replacement: object
) -> None:
    assert policy_uniform(_key(**{field: replacement})) != policy_uniform(_key())


def test_policy_tape_has_no_shuffle_checkpoint_or_policy_identity_input() -> None:
    assert [field.name for field in dataclasses.fields(BidSamplingKey)] == [
        "policy_seed",
        "deal_id",
        "room_id",
        "logical_seat",
        "bid_index",
    ]
    assert list(inspect.signature(policy_uniform).parameters) == ["key"]
    with pytest.raises(TypeError):
        BidSamplingKey(1, "deal", "room", 0, 0, shuffle_seed=9)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        policy_uniform(_key(), policy_id="changed")  # type: ignore[call-arg]


class _ExtremeDigest:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def digest(self) -> bytes:
        return self.value


@pytest.mark.parametrize(
    ("digest", "expected"),
    [
        (b"\x00" * 8, 0.5 / float(1 << 64)),
        (b"\xff" * 8, math.nextafter(1.0, 0.0)),
    ],
)
def test_policy_uniform_stays_open_at_extreme_digests(
    digest: bytes, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        random_tape.hashlib,
        "blake2b",
        lambda *args, **kwargs: _ExtremeDigest(digest),
    )

    result = policy_uniform(_key())

    assert result == expected
    assert 0.0 < result < 1.0


@pytest.mark.parametrize(
    "key",
    [
        object(),
        BidSamplingKey(True, "deal", "room", 0, 0),
        BidSamplingKey(1, "", "room", 0, 0),
        BidSamplingKey(1, "deal", "", 0, 0),
        BidSamplingKey(1, "deal", "room", 4, 0),
        BidSamplingKey(1, "deal", "room", 0, -1),
    ],
)
def test_policy_uniform_rejects_noncanonical_keys(key: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        policy_uniform(key)  # type: ignore[arg-type]
