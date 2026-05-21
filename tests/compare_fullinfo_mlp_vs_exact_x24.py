"""Evaluate full-info MLPs against exact-solver labels on x=24 samples.

This script loads the 20,000-sample dataset from `data/spades_dd_gpu_full_x24_n20000.pt`,
selects the bid-specific full-info model by the current player's bid, and compares the
predicted value_view against the exact solver label stored in the dataset.

Inputs / outputs:
- load_samples(dataset_path: Path, num_samples: int) -> list[dict]
  Input: path to a `.pt` dataset and the desired sample count.
  Output: list of sample dictionaries loaded from the dataset.
- parse_bid_value(bid_value: object) -> int
  Input: bid string or numeric bid value.
  Output: integer trick bid, where nil/blind_nil map to 0.
- evaluate_samples(samples: list[dict], bid0_model: FullInfoValueMLP, bidpos_model: FullInfoValueMLP, value_scale: float) -> dict[int, dict]
  Input: dataset samples, the two bid-specific models, and the value scaling factor.
  Output: dictionary keyed by bid value, each entry containing count, mean_abs_diff, and median_abs_diff.
- main() -> None
  Input: CLI arguments.
  Output: printed summary table by bid and overall.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import load_bucket_dataset
from new_mlp.model import FullInfoValueMLP
from trick_taking.utils.feature_encoder import FullInfoSpadesFeatureEncoder


def parse_bid_value(bid_value: Any) -> int:
    """Convert a bid representation into an integer trick count.

    Input:
    - bid_value: strings like "nil", "blind_nil", "bid_7", or numeric values.

    Output:
    - int: 0 for nil/blind_nil, otherwise the numeric trick count.
    """
    if bid_value is None:
        return 0
    if isinstance(bid_value, str):
        if bid_value in {"nil", "blind_nil"}:
            return 0
        if bid_value.startswith("bid_"):
            return int(bid_value.split("_", 1)[1])
    if isinstance(bid_value, (int, float)):
        return int(bid_value)
    return 0


def load_samples(dataset_path: Path, num_samples: int) -> list[dict[str, Any]]:
    """Load samples from a `.pt` dataset file.

    Input:
    - dataset_path: path to the dataset file.
    - num_samples: number of samples to keep from the front of the dataset.

    Output:
    - list of sample dictionaries.
    """
    loaded = load_bucket_dataset(dataset_path)
    samples = loaded["samples"]
    if len(samples) < num_samples:
        raise ValueError(f"dataset has only {len(samples)} samples, requested {num_samples}")
    return samples[:num_samples]


def _stack_features(features: list[Any]) -> torch.Tensor:
    """Stack a list of feature rows into a single float32 tensor.

    Input:
    - features: list of per-sample feature rows, each being a tensor or array-like of shape (1385,).

    Output:
    - torch.Tensor of shape (N, 1385), dtype float32.
    """
    arrays = [np.asarray(feature, dtype=np.float32) for feature in features]
    return torch.from_numpy(np.stack(arrays, axis=0))


def evaluate_samples(
    samples: list[dict[str, Any]],
    bid0_model: FullInfoValueMLP,
    bidpos_model: FullInfoValueMLP,
    value_scale: float,
) -> dict[int, dict[str, float]]:
    """Evaluate the two models on a dataset and aggregate by bid.

    Input:
    - samples: list of dataset sample dictionaries.
    - bid0_model: model used when current player's bid <= 0.
    - bidpos_model: model used when current player's bid > 0.
    - value_scale: multiplier to convert the network output back to point scale.

    Output:
    - dict mapping bid value -> summary dict with keys count, mean_abs_diff, median_abs_diff.
    """
    grouped_features: dict[int, list[Any]] = defaultdict(list)
    grouped_targets: dict[int, list[float]] = defaultdict(list)

    for sample in samples:
        current_player = int(sample.get("current_player", 0))
        bids = list(sample.get("state_summary", {}).get("bids", []))
        bid_raw = bids[current_player] if current_player < len(bids) else None
        bid_value = parse_bid_value(bid_raw)

        value_view = sample["value_view"]
        target = float(value_view.item() if hasattr(value_view, "item") else value_view)
        grouped_features[bid_value].append(sample["feature"])
        grouped_targets[bid_value].append(target)

    stats: dict[int, dict[str, float]] = {}
    for bid_value, features in sorted(grouped_features.items(), key=lambda item: item[0]):
        targets = grouped_targets[bid_value]
        model = bid0_model if bid_value <= 0 else bidpos_model
        batch = _stack_features(features)
        with torch.no_grad():
            predictions_scaled = model.predict(batch)
        predictions = np.asarray(predictions_scaled, dtype=np.float64) * float(value_scale)
        targets_arr = np.asarray(targets, dtype=np.float64)
        abs_diffs = np.abs(predictions - targets_arr)
        stats[bid_value] = {
            "count": float(len(abs_diffs)),
            "mean_abs_diff": float(mean(abs_diffs.tolist())),
            "median_abs_diff": float(median(abs_diffs.tolist())),
        }
    return stats


def _load_model(checkpoint_path: Path, input_dim: int, device: str) -> FullInfoValueMLP:
    """Load a full-info value model from disk.

    Input:
    - checkpoint_path: checkpoint file path.
    - input_dim: feature dimension expected by the model.
    - device: target device string.

    Output:
    - Fully loaded FullInfoValueMLP instance.
    """
    model = FullInfoValueMLP(input_dim=input_dim)
    model.load(str(checkpoint_path), device=device)
    return model


def main() -> None:
    """CLI entry point.

    Input:
    - --dataset: dataset path, defaults to data/spades_dd_gpu_full_x24_n20000.pt.
    - --num-samples: number of samples to evaluate.
    - --bid0-checkpoint: checkpoint for the bid==0 model.
    - --bidpos-checkpoint: checkpoint for the bid>0 model.
    - --device: torch device used for inference.
    - --value-scale: scale factor to convert network output back to point units.

    Output:
    - Prints per-bid sample counts and mean absolute error, plus an overall summary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/spades_dd_gpu_full_x24_n40000.pt")
    parser.add_argument("--num-samples", type=int, default=40000)
    parser.add_argument("--bid0-checkpoint", type=str, default="result/fullinfo_bid0_9.pth")
    parser.add_argument("--bidpos-checkpoint", type=str, default="result/fullinfo_bidpos_9.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--value-scale", type=float, default=25.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = load_samples(dataset_path, args.num_samples)
    encoder = FullInfoSpadesFeatureEncoder()
    sample_feature_dim = int(samples[0]["feature"].shape[0] if hasattr(samples[0]["feature"], "shape") else len(samples[0]["feature"]))
    if sample_feature_dim != encoder.total_dim:
        raise ValueError(f"feature dim mismatch: dataset={sample_feature_dim}, encoder={encoder.total_dim}")

    bid0_checkpoint = Path(args.bid0_checkpoint)
    bidpos_checkpoint = Path(args.bidpos_checkpoint)
    if not bid0_checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {bid0_checkpoint}")
    if not bidpos_checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {bidpos_checkpoint}")

    bid0_model = _load_model(bid0_checkpoint, encoder.total_dim, args.device)
    bidpos_model = _load_model(bidpos_checkpoint, encoder.total_dim, args.device)

    stats = evaluate_samples(samples, bid0_model, bidpos_model, args.value_scale)

    total_count = 0
    total_abs_diffs: list[float] = []
    print("Per-bid statistics")
    for bid_value in sorted(stats.keys()):
        entry = stats[bid_value]
        count = int(entry["count"])
        total_count += count
        total_abs_diffs.extend([])
        print(
            f"叫 {bid_value} 墩 | 样本 {count:5d} | "
            f"平均绝对差 {entry['mean_abs_diff']:.6f} | 中位数绝对差 {entry['median_abs_diff']:.6f}"
        )

    # Recompute the overall stats from the grouped entries.
    all_abs_diffs: list[float] = []
    for bid_value in sorted(stats.keys()):
        # The summary dict only stores aggregates, so we rebuild the overall average from
        # the per-bid mean weighted by count. This keeps the output deterministic and simple.
        pass

    weighted_sum = 0.0
    for bid_value in sorted(stats.keys()):
        entry = stats[bid_value]
        count = int(entry["count"])
        weighted_sum += entry["mean_abs_diff"] * count

    overall_mean = weighted_sum / max(total_count, 1)
    print(f"整体 | 样本 {total_count:5d} | 平均绝对差 {overall_mean:.6f}")


if __name__ == "__main__":
    main()
