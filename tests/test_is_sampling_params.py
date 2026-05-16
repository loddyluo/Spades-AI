import importlib

from types import SimpleNamespace


def test_evaluate_model_matchups_default_is_count():
    mod = importlib.import_module("evaluate.evaluate_model_matchups")
    args = mod.parse_args()
    rt = mod.build_runtime(args)
    assert rt.local_mcts_config.mcts_determinization_count == 10


def test_evaluate_our_mcts_vs_rule_v2_default_is_count():
    mod = importlib.import_module("evaluate.evaluate_our_mcts_vs_rule_v2")
    args = mod.parse_args()
    rt = mod.build_runtime(args)
    assert rt.local_mcts_config.mcts_determinization_count == 10
