"""Hyperparameter configuration for RuleExactFirst4Player.

Provides a dataclass-based config that can be loaded from YAML files,
enabling hyperparameter search via matchups between two config variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BudgetThreshold:
    """One row in the budget table: when remaining_in <= max_remaining, use these values."""
    max_remaining: int
    top_k: int
    max_samples: int


@dataclass
class BudgetConfig:
    """Budget table: maps remaining_in ranges to (top_k, max_samples)."""
    thresholds: list[BudgetThreshold] = field(default_factory=lambda: [
        BudgetThreshold(max_remaining=16, top_k=128, max_samples=256),
        BudgetThreshold(max_remaining=24, top_k=64, max_samples=128),
        BudgetThreshold(max_remaining=28, top_k=32, max_samples=64),
        BudgetThreshold(max_remaining=32, top_k=16, max_samples=32),
    ])
    default_top_k: int = 12
    default_max_samples: int = 24

    def lookup(self, remaining_in: int) -> tuple[int, int]:
        """Return (top_k, max_samples) for the given remaining card count.

        Checks thresholds from smallest remaining_in to largest, so that
        the most specific (narrowest) matching threshold is used.
        """
        for t in sorted(self.thresholds, key=lambda x: x.max_remaining):
            if remaining_in <= t.max_remaining:
                return t.top_k, t.max_samples
        return self.default_top_k, self.default_max_samples


@dataclass
class HyperparamConfig:
    """Complete hyperparameter configuration for RuleExactFirst4Player.

    Field descriptions / defaults:
      num_proposals: int = 1234
          Batch size for proposal generation in _build_is_pool.
      num_proposals_limit: int = 5678
          Maximum total proposals to attempt before giving up.
      min_pool_size: int = 100
          Minimum number of valid (w > 0) proposals required.
      budget: BudgetConfig
          Maps remaining_in ranges to (top_k, max_samples) values.
      bad_action_weight: str = "x"
          Multiplier for bad actions during importance weight computation.
          Values:
            - "x"        : weight *= bad_count / total  (原版行为)
            - "0.5"      : weight *= 0.5  (常数)
            - "0.0"      : weight *= 0.0  (有坏动作就淘汰)
            - "1.0"      : weight *= 1.0  (不惩罚)
            - "x**2"     : weight *= x²
      trick_num_threshold: int = 8
          Tricks with index >= this value use solver Q-values for weighting
          (0-indexed; 8 = 9th trick onward).
      swap_is_fill: bool = False
          If True, do diversity fill first (sorted by weight), then IS from
          remaining proposals.  If False (default), do IS first, then fill.
      multiplier_clip: float = 40.0
          Q-value multiplier clipping threshold in _exact_play (lines 353/358).
      multiplier_clip_factor: float = 1.0
          Multiplication factor applied when the clip threshold is exceeded.
    """

    # IS pool
    num_proposals: int = 1234
    num_proposals_limit: int = 5678
    min_pool_size: int = 100

    # Budget table
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Importance weight computation
    bad_action_weight: str = "x"  # "x"=bad_count/total; 数字字符串=常数
    trick_num_threshold: int = 8

    # Selection ordering
    swap_is_fill: bool = False

    # Q-value multiplier clipping
    multiplier_clip: float = 40.0
    multiplier_clip_factor: float = 1.0

    # Parallelism
    num_workers: int = 0  # 0 = auto (cpu_count - 1), >0 = fixed count

    @classmethod
    def default(cls) -> HyperparamConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> HyperparamConfig:
        """Load configuration from a YAML file.

        The YAML file is expected to be a flat/dict structure matching the
        field names above.  Missing keys get their default values.
        """
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        # Pop nested structures
        is_pool_raw = raw.pop("is_pool", {}) or {}
        budget_raw = raw.pop("budget", {}) or {}

        # Build BudgetConfig
        thresholds_raw = budget_raw.get("thresholds", []) or []
        thresholds = [
            BudgetThreshold(
                max_remaining=t["remaining_in"],
                top_k=t["top_k"],
                max_samples=t["max_samples"],
            )
            for t in thresholds_raw
        ]
        budget = BudgetConfig(
            thresholds=thresholds or BudgetConfig().thresholds,
            default_top_k=budget_raw.get("default", {}).get("top_k", BudgetConfig().default_top_k),
            default_max_samples=budget_raw.get("default", {}).get("max_samples", BudgetConfig().default_max_samples),
        )

        # Merge is_pool keys into the top-level raw dict
        raw.update(is_pool_raw)

        return cls(
            num_proposals=raw.get("num_proposals", 1234),
            num_proposals_limit=raw.get("num_proposals_limit", 5678),
            min_pool_size=raw.get("min_pool_size", 100),
            budget=budget,
            bad_action_weight=raw.get("bad_action_weight", "x"),
            trick_num_threshold=raw.get("trick_num_threshold", 8),
            swap_is_fill=bool(raw.get("swap_is_fill", False)),
            multiplier_clip=float(raw.get("multiplier_clip", 40.0)),
            multiplier_clip_factor=float(raw.get("multiplier_clip_factor", 1.0)),
            num_workers=int(raw.get("num_workers", 0)),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Write the current configuration to a YAML file."""
        import yaml

        # Structure for human-readable output
        obj: dict[str, Any] = {
            "is_pool": {
                "num_proposals": self.num_proposals,
                "num_proposals_limit": self.num_proposals_limit,
                "min_pool_size": self.min_pool_size,
            },
            "budget": {
                "thresholds": [
                    {
                        "remaining_in": t.max_remaining,
                        "top_k": t.top_k,
                        "max_samples": t.max_samples,
                    }
                    for t in self.budget.thresholds
                ],
                "default": {
                    "top_k": self.budget.default_top_k,
                    "max_samples": self.budget.default_max_samples,
                },
            },
            "bad_action_weight": self.bad_action_weight,
            "trick_num_threshold": self.trick_num_threshold,
            "swap_is_fill": self.swap_is_fill,
            "multiplier_clip": self.multiplier_clip,
            "multiplier_clip_factor": self.multiplier_clip_factor,
            "num_workers": self.num_workers,
        }

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
