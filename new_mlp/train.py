"""Train full-information value MLPs for bid==0 and bid>0 cases.

Usage examples:
    /mnt/c/Users/35559/Spades-AI/.venv/bin/python new_mlp/train.py \
        --bid0-save result/fullinfo_bid0.pth \
        --bidpos-save result/fullinfo_bidpos.pth \
        --train-dataset data/spades_dd_gpu_full_x24_n1000000.pt \
        --test-dataset data/spades_dd_gpu_full_x24_n50000.pt \
        --epochs 200 \
        --batch-size 1024

Module functions (inputs/outputs):
- load_training_samples(dataset_path: Path) -> list[dict]
    Input: exact dataset file path.
    Output: list of sample dicts loaded from that single .pt file.
- split_samples_by_bid(samples: list[dict]) -> tuple[list[dict], list[dict]]
    Input: sample list containing `state_summary` and `current_player`.
    Output: (bid0_samples, bidpos_samples).
- prepare_value_arrays(samples: list[dict], value_scale: float) -> dict[str, np.ndarray]
    Input: sample list with `feature` and `value_view`.
    Output: dict with `features` (N,D) and `value_targets` (N,1).
- train_value_model(..., eval_samples: list[dict], eval_every: int, patience_evals: int, min_delta: float, weight_decay: float) -> dict[str, float]
    Input: train arrays, model, device, training hyperparams, and evaluation samples.
    Output: summary of the best validation checkpoint and early-stopping state.
- main() -> None
    Input: CLI args.
    Output: trains and saves two checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import load_bucket_dataset
from new_mlp.model import FullInfoValueMLP


def load_training_samples(dataset_path: Path) -> list[dict[str, Any]]:
    """Load dataset samples from one exact dataset file.

    Input:
    - dataset_path: exact dataset file path.

    Output:
    - List of sample dicts loaded from the file.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    loaded = load_bucket_dataset(dataset_path)
    samples = loaded.get("samples", [])
    if not samples:
        raise FileNotFoundError(f"no samples loaded from {dataset_path}")
    return samples


def _parse_bid_value(bid_value: Any) -> int:
    """Convert a bid value to an integer trick count.

    Input:
    - bid_value: bid string ("bid_3", "nil", "blind_nil") or numeric.

    Output:
    - int trick count; nil/blind_nil treated as 0.
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


def _current_player_bid(sample: dict[str, Any]) -> int:
    """Return the current player's bid as an integer.

    Input:
    - sample: dataset sample dict containing `state_summary` and `current_player`.

    Output:
    - int bid value (0 for nil/blind_nil).
    """
    state_summary = sample.get("state_summary", {})
    bids = list(state_summary.get("bids", []))
    current_player = int(sample.get("current_player", 0))
    bid_value = bids[current_player] if current_player < len(bids) else None
    return _parse_bid_value(bid_value)


def split_samples_by_bid(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples into bid==0 and bid>0 groups.

    Input:
    - samples: list of dataset samples.

    Output:
    - (bid0_samples, bidpos_samples)
    """
    bid0_samples: list[dict[str, Any]] = []
    bidpos_samples: list[dict[str, Any]] = []
    for sample in samples:
        bid = _current_player_bid(sample)
        if bid <= 0:
            bid0_samples.append(sample)
        else:
            bidpos_samples.append(sample)
    return bid0_samples, bidpos_samples


def prepare_value_arrays(samples: list[dict[str, Any]], value_scale: float = 25.0) -> dict[str, np.ndarray]:
    """Convert samples into numpy arrays for value training.

    Input:
    - samples: list of dataset samples with `feature` and `value_view` fields.
    - value_scale: scaling divisor for value targets.

    Output:
    - dict with keys:
        features: np.ndarray (N, D)
        value_targets: np.ndarray (N, 1)
    """
    if not samples:
        raise ValueError("samples must not be empty")

    feature_rows = []
    value_targets = []
    for sample in samples:
        feature_rows.append(np.asarray(sample["feature"], dtype=np.float32))
        value_targets.append([float(sample["value_view"]) / value_scale])

    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "value_targets": np.asarray(value_targets, dtype=np.float32),
    }


def _stack_features(features: list[Any]) -> torch.Tensor:
    arrays = [np.asarray(feature, dtype=np.float32) for feature in features]
    return torch.from_numpy(np.stack(arrays, axis=0))


def evaluate_model_on_samples(
    model: FullInfoValueMLP,
    samples: list[dict[str, Any]],
    value_scale: float,
    batch_size: int,
) -> dict[str, float]:
    if not samples:
        raise ValueError("evaluation samples must not be empty")

    features = _stack_features([sample["feature"] for sample in samples])
    targets = np.asarray([float(sample["value_view"]) for sample in samples], dtype=np.float32)

    predictions: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_pred = model.predict(features[start:end])
            batch_np = np.asarray(batch_pred, dtype=np.float64)
            if batch_np.ndim == 0:
                batch_np = batch_np.reshape(1)
            predictions.extend((batch_np * float(value_scale)).tolist())

    abs_diffs = np.abs(np.asarray(predictions, dtype=np.float64) - targets.astype(np.float64))
    return {
        "count": float(len(abs_diffs)),
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "median_abs_diff": float(np.median(abs_diffs)),
    }


def train_value_model(
    features: np.ndarray,
    targets: np.ndarray,
    model: FullInfoValueMLP,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    eval_samples: list[dict[str, Any]],
    eval_every: int,
    patience_evals: int,
    min_delta: float,
    eval_batch_size: int,
    checkpoint_path: str,
    bid_kind: str,
    weight_decay: float,
    value_scale: float = 25.0,
) -> dict[str, float]:
    """Train a value-only model on the provided arrays.

    Input:
    - features: np.ndarray (N, D)
    - targets: np.ndarray (N, 1)
    - model: FullInfoValueMLP instance
    - device: "cpu" or "cuda"
    - epochs: training epochs
    - batch_size: minibatch size
    - lr: learning rate

    Output:
    - Summary of best validation metrics and the early-stopping outcome.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(features)
    y_t = torch.from_numpy(targets)

    losses: list[float] = []
    best_eval_mae = float("inf")
    best_epoch = 0
    best_state = None
    evals_without_improvement = 0

    for epoch in range(epochs):
        permutation = torch.randperm(len(X_t))
        X_t = X_t[permutation]
        y_t = y_t[permutation]
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        batches = 0
        for start in range(0, len(X_t), batch_size):
            end = min(start + batch_size, len(X_t))
            batch_x = X_t[start:end].to(device)
            batch_y = y_t[start:end].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)["value"]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # outputs and batch_y are in scaled units (value/view divided by value_scale)
            # convert MAE back to original point units by multiplying by value_scale
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(outputs - batch_y)).item() * float(value_scale)
            running_mae += batch_mae
            batches += 1
        avg_loss = running_loss / max(batches, 1)
        avg_mae = running_mae / max(batches, 1)
        losses.append(avg_loss)

        should_eval = ((epoch + 1) % max(eval_every, 1) == 0) or (epoch + 1 == epochs)
        if should_eval:
            eval_stats = evaluate_model_on_samples(
                model,
                eval_samples,
                value_scale=value_scale,
                batch_size=eval_batch_size,
            )
            eval_mae = float(eval_stats["mean_abs_diff"])
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | loss={avg_loss:.6f} | "
                f"train_MAE={avg_mae:.6f} pts | val_MAE={eval_mae:.6f} pts"
            )
            if eval_mae + min_delta < best_eval_mae:
                best_eval_mae = eval_mae
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience_evals:
                    print(
                        f"Early stopping at epoch {epoch + 1} for {bid_kind}: "
                        f"best val_MAE={best_eval_mae:.6f} pts at epoch {best_epoch}"
                    )
                    break
        else:
            print(f"Epoch {epoch + 1:03d}/{epochs} | loss={avg_loss:.6f} | train_MAE={avg_mae:.6f} pts")

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "best_epoch": float(best_epoch),
        "best_val_mae": float(best_eval_mae),
        "epochs_ran": float(len(losses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=str, default="data/spades_dd_gpu_full_x24_n1000000.pt", help="Exact training dataset file")
    parser.add_argument("--test-dataset", type=str, default="data/spades_dd_gpu_full_x24_n50000.pt", help="Exact validation/test dataset file")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--value-scale", type=float, default=25.0, help="Value scaling divisor")
    parser.add_argument("--eval-every", type=int, default=10, help="Run validation every N epochs")
    parser.add_argument("--patience-evals", type=int, default=6, help="Early-stop after this many failed validation checks")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation improvement to reset patience")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate inside the MLP")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64, 32], help="Hidden layer sizes for the value MLP")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Batch size used for validation/test evaluation")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--bid0-save", type=str, required=True, help="Checkpoint path for bid==0 model")
    parser.add_argument("--bidpos-save", type=str, required=True, help="Checkpoint path for bid>0 model")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = Path(args.train_dataset)
    test_dataset = Path(args.test_dataset)

    train_samples = load_training_samples(train_dataset)
    test_samples = load_training_samples(test_dataset)

    bid0_samples, bidpos_samples = split_samples_by_bid(train_samples)
    test_bid0_samples, test_bidpos_samples = split_samples_by_bid(test_samples)

    if not bid0_samples:
        raise ValueError("No bid==0 samples found; check dataset or prefix")
    if not bidpos_samples:
        raise ValueError("No bid>0 samples found; check dataset or prefix")
    if not test_bid0_samples:
        raise ValueError("No bid==0 samples found in test dataset")
    if not test_bidpos_samples:
        raise ValueError("No bid>0 samples found in test dataset")

    bid0_arrays = prepare_value_arrays(bid0_samples, value_scale=args.value_scale)
    bidpos_arrays = prepare_value_arrays(bidpos_samples, value_scale=args.value_scale)

    bid0_model = FullInfoValueMLP(input_dim=bid0_arrays["features"].shape[1], hidden_dims=args.hidden_dims, dropout=args.dropout)
    bidpos_model = FullInfoValueMLP(input_dim=bidpos_arrays["features"].shape[1], hidden_dims=args.hidden_dims, dropout=args.dropout)

    print(f"train dataset: {train_dataset} | samples={len(train_samples)}")
    print(f"test dataset:  {test_dataset} | samples={len(test_samples)}")

    print(f"bid==0 train samples: {len(bid0_samples)} | test samples: {len(test_bid0_samples)}")
    bid0_summary = train_value_model(
        bid0_arrays["features"],
        bid0_arrays["value_targets"],
        bid0_model,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=test_bid0_samples,
        eval_every=args.eval_every,
        patience_evals=args.patience_evals,
        min_delta=args.min_delta,
        eval_batch_size=args.eval_batch_size,
        checkpoint_path=args.bid0_save,
        bid_kind="bid0",
        weight_decay=args.weight_decay,
        value_scale=args.value_scale,
    )
    print(
        f"Saved bid==0 model to {args.bid0_save} | best_epoch={int(bid0_summary['best_epoch'])} "
        f"| best_val_MAE={bid0_summary['best_val_mae']:.6f} pts"
    )

    print(f"bid>0 train samples: {len(bidpos_samples)} | test samples: {len(test_bidpos_samples)}")
    bidpos_summary = train_value_model(
        bidpos_arrays["features"],
        bidpos_arrays["value_targets"],
        bidpos_model,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=test_bidpos_samples,
        eval_every=args.eval_every,
        patience_evals=args.patience_evals,
        min_delta=args.min_delta,
        eval_batch_size=args.eval_batch_size,
        checkpoint_path=args.bidpos_save,
        bid_kind="bidpos",
        weight_decay=args.weight_decay,
        value_scale=args.value_scale,
    )
    print(
        f"Saved bid>0 model to {args.bidpos_save} | best_epoch={int(bidpos_summary['best_epoch'])} "
        f"| best_val_MAE={bidpos_summary['best_val_mae']:.6f} pts"
    )


if __name__ == "__main__":
    main()
