"""Regression tests for full-information dataset generation.

Inputs:
- `generate_bucket_sample(target_remaining, seed, full_info=True)` expects an
	integer remaining-card target and a seed.
- `save_bucket_dataset(samples, output_path, full_info=True)` expects a list of
	samples and a filesystem path to write.

Outputs:
- Asserts that full-info samples use the full-information feature dimension.
- Asserts that the saved dataset metadata records `full_info=True`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))

from data.training_data import generate_bucket_sample, load_bucket_dataset, save_bucket_dataset
from trick_taking.utils.feature_encoder import FullInfoSpadesFeatureEncoder


def test_generate_bucket_sample_full_info_dimension() -> None:
		"""Verify full-info samples match the full-information encoder dimension.

		Input:
		- A deterministic seed and target_remaining bucket.

		Output:
		- Asserts feature length equals `FullInfoSpadesFeatureEncoder.total_dim`.
		"""
		encoder = FullInfoSpadesFeatureEncoder()
		sample = generate_bucket_sample(24, seed=2026, full_info=True)
		assert sample["feature"].shape[0] == encoder.total_dim


def test_save_bucket_dataset_marks_full_info(tmp_path: Path) -> None:
    """Verify dataset metadata records the full-info flag.

    Input:
    - One generated full-info sample and a temporary output directory.

    Output:
    - Asserts meta["full_info"] is True after saving/loading.
    """
    sample = generate_bucket_sample(24, seed=2027, full_info=True)
    out_path = tmp_path / "full_info_dataset.pt"
    save_bucket_dataset([sample], out_path, full_info=True)
    loaded = load_bucket_dataset(out_path)
    assert loaded["meta"]["full_info"] is True