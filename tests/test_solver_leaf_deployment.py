from __future__ import annotations

import pytest

from rl.nil_solver_leaf_deployment import load_deployed_nil_actor_bundle
from rl.solver_leaf_deployment import load_deployed_solver_leaf_actor


def test_default_production_play_checkpoints_load_and_run() -> None:
    nonnil = load_deployed_solver_leaf_actor()
    nil = load_deployed_nil_actor_bundle()

    assert nonnil.metadata["schema"] == "solver-leaf-actor-v1"
    assert nonnil.metadata["input_dim"] == 536
    assert nonnil.metadata["output_dim"] == 52
    assert set(nil.actors) == {
        "nil_self",
        "nil_partner",
        "nil_upper",
        "nil_lower",
    }


def test_production_loaders_fail_closed_on_wrong_hash() -> None:
    with pytest.raises(ValueError, match="actor SHA-256 mismatch"):
        load_deployed_solver_leaf_actor(expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="bundle SHA-256 mismatch"):
        load_deployed_nil_actor_bundle(expected_sha256="0" * 64)
