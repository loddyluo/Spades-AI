"""Deterministic, immutable random tape for stochastic bid decisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


POLICY_TAPE_DOMAIN = b"residual-bidder-policy-tape-v1"


@dataclass(frozen=True)
class BidSamplingKey:
    """The complete canonical identity of one bid-policy variate."""

    policy_seed: int
    deal_id: str
    room_id: str
    logical_seat: int
    bid_index: int


def _validate_key(key: BidSamplingKey) -> None:
    if not isinstance(key, BidSamplingKey):
        raise TypeError("key must be a BidSamplingKey")
    if type(key.policy_seed) is not int:
        raise TypeError("policy_seed must be an integer")
    if not isinstance(key.deal_id, str) or not key.deal_id:
        raise ValueError("deal_id must be a nonempty string")
    if not isinstance(key.room_id, str) or not key.room_id:
        raise ValueError("room_id must be a nonempty string")
    if type(key.logical_seat) is not int or not 0 <= key.logical_seat < 4:
        raise ValueError("logical_seat must be an integer from 0 through 3")
    if type(key.bid_index) is not int or key.bid_index < 0:
        raise ValueError("bid_index must be a nonnegative integer")


def policy_uniform(key: BidSamplingKey) -> float:
    """Map a canonical bid key to one reproducible open-interval uniform."""

    _validate_key(key)
    payload = json.dumps(
        asdict(key),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.blake2b(
        POLICY_TAPE_DOMAIN + b"\0" + payload,
        digest_size=8,
    ).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    uniform = (integer + 0.5) / float(1 << 64)
    if uniform >= 1.0:
        return math.nextafter(1.0, 0.0)
    return uniform
