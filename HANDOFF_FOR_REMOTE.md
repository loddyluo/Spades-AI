# Hand-off: Spades-AI MCTS + Full-Info MLP Work (snapshot 2026-05-20)

Purpose
-------
This file captures the full context needed for another developer (or another Copilot instance) to pick up work on the Spades-AI repository where we left off. It summarizes the goals, what we changed, test status, exact commands to reproduce data generation / training / tests, and the next tasks.

High-level goal
---------------
- Fix overfitting observed in the original x=24 MLP by training two separate full-information value networks:
  - one for games where the current player's bid == 0 (nil/ blind_nil),
  - one for games where bid > 0.
- Use full-information inputs for leaf evaluation in truncated MCTS when remaining cards <= 24 (we can use sampled opponents' hands).

Summary of what was implemented
--------------------------------
1. Feature encoder
   - Added `FullInfoSpadesFeatureEncoder` (in `trick_taking/utils/feature_encoder.py`).
   - Input dimension: 1385 = original 1229 + 3*52 (one-hot for remaining cards of the three opponents).
   - Purpose: produce feature vectors that include opponent remaining cards for full-info MLP.

2. Data generation pipeline
   - `data/training_data.py` updated to accept `full_info: bool` and to use `FullInfoSpadesFeatureEncoder` when requested.
   - `data/generate_dataset.py` and `data/generate_dataset_gpu.py` extended with `--full-info` flags.
   - Saved dataset metadata includes `full_info: true` for provenance.

3. New MLP package (separate from `mlp/`)
   - `new_mlp/model.py`: `FullInfoValueMLP` (value-only network). Input dim 1385, FC layers [2048,1024,512,256] -> 1 scalar.
   - `new_mlp/train.py`: training script that splits samples by bidder (bid==0 vs bid>0) and trains two separate models.
   - CLI example included in code header; training target is `value_view / 25.0` (scale used during training).

4. MCTS integration
   - `strategy/truncated_mcts_strategy.py` was modified to load two optional full-info checkpoints:
     - `TruncatedMCTSConfig.full_info_bid0_checkpoint`
     - `TruncatedMCTSConfig.full_info_bidpos_checkpoint`
   - A `FullInfoSpadesFeatureEncoder` instance is used when a full-info model is selected.
   - `_select_full_info_model(state)` chooses which model to use based on the current player's bid extracted from `state.max_bid`.
   - `_leaf_value()` uses full-info model if available; otherwise falls back to the old model.

Tests and CI
------------
All new/modified tests are in the `tests/` directory. The following test files were added and are passing in the current environment:

- `tests/test_full_info_feature_encoder.py` (2 tests) — PASSED
- `tests/test_full_info_dataset_generation.py` (2 tests) — PASSED
- `tests/test_new_mlp_model.py` (2 tests) — PASSED
- `tests/test_full_info_mcts_integration.py` (3 tests) — PASSED

Run them locally with:

```bash
# run all new tests
pytest tests/test_full_info_*.py -v
```

Repro commands (data / train / use)
----------------------------------
1) Generate full-info dataset (GPU multi-process recommended for large N):

```bash
python data/generate_dataset_gpu.py \
  --xs 24 \
  --num_samples 50000 \
  --full-info \
  --seed-start 2000000 \
  --prefix spades_dd_gpu_full \
  --num-workers 16
```

Output: `data/spades_dd_gpu_full_x24_n50000.pt` (or similar files matching the prefix).

2) Train the two bid-stratified full-info MLPs:

```bash
python new_mlp/train.py \
  --xs 24 \
  --dataset-prefix spades_dd_gpu_full \
  --bid0-save result/fullinfo_bid0.pth \
  --bidpos-save result/fullinfo_bidpos.pth \
  --epochs 200 \
  --batch-size 1024 \
  --device cpu
```

Outputs: `result/fullinfo_bid0.pth` and `result/fullinfo_bidpos.pth`.

3) Use them in MCTS (Python snippet):

```python
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy
config = TruncatedMCTSConfig(
    full_info_bid0_checkpoint="result/fullinfo_bid0.pth",
    full_info_bidpos_checkpoint="result/fullinfo_bidpos.pth",
    full_info_value_scale=25.0,
    leaf_threshold=24,
    exact_threshold=24,
)
strategy = TruncatedMCTSStrategy(config)
# then use strategy.choose_action(state)
```

Files changed (most relevant)
-----------------------------
- Added/Modified:
  - trick_taking/utils/feature_encoder.py  (added FullInfoSpadesFeatureEncoder)
  - data/training_data.py  (full_info propagation)
  - data/generate_dataset.py  (CLI: --full-info)
  - data/generate_dataset_gpu.py  (CLI: --full-info)
  - new_mlp/model.py  (FullInfoValueMLP)
  - new_mlp/train.py  (training/CLI)
  - strategy/truncated_mcts_strategy.py  (full-info model loading & selection)
  - tests/test_full_info_*.py  (4 test files)
  - copilot_wyy.md  (updated notes)
  - FULLINFO_IMPLEMENTATION_SUMMARY.md (detailed summary)
  - FULLINFO_QUICK_START.md (quick start guide)
  - HANDOFF_FOR_REMOTE.md (this file)

Current test status (local environment)
---------------------------------------
- All newly-added tests pass: 9/9.
- Existing test sets that were run after integration also passed in the environment used when changes were made.

Assumptions and caveats
-----------------------
- The full-info models assume that, at leaf time, we can use determinized opponents' hands (sampled by IS in MCTS) to produce meaningful features. The IS mechanism itself was not changed here, except for hooking in the prior oracle (the rule-based oracle) into weights earlier; IS correctness remains a research question.
- Model input dimension change requires regenerating datasets before training new models; old checkpoints using 1229-dim inputs are incompatible with new 1385-dim models.
- Training data quality (seed ranges / distribution) impacts generalization heavily; previously observed overfitting was caused by seed overlap and distribution mismatch.

What to do next (priority list for the next developer)
------------------------------------------------------
1. Generate a large full-info dataset (recommended 50k–100k x=24 samples) using `data/generate_dataset_gpu.py --full-info`.
2. Train `fullinfo_bid0` and `fullinfo_bidpos` using `new_mlp/train.py` (see command above).
3. Evaluate MAE on fresh seeds (not overlapping with dataset seed range) and compare against the old single-model baseline.
4. If MAE improves meaningfully, run full-game MCTS evaluations to measure winrate changes.
5. Profile MCTS to measure the extra cost of full-info encoding + model forward passes, optimize if necessary.

Quick troubleshooting
-------------------
- If model loading fails: verify `result/*.pth` exists and that it was trained with `input_dim=1385`.
- If tests fail: run `pytest tests/test_full_info_*.py -q` to see stack traces.
- If dataset generation is slow: reduce `--num-workers` to match CPU core saturation or run smaller batches.

Pointers to important code locations
-----------------------------------
- Feature encoder: `trick_taking/utils/feature_encoder.py`
- Dataset generation: `data/training_data.py`, `data/generate_dataset_gpu.py`
- New model & training: `new_mlp/model.py`, `new_mlp/train.py`
- MCTS integration: `strategy/truncated_mcts_strategy.py`
- Tests: `tests/test_full_info_*.py`
- Project notes: `copilot_wyy.md`, `FULLINFO_IMPLEMENTATION_SUMMARY.md`, `FULLINFO_QUICK_START.md`

Contact / context
-----------------
- Branch: `MCTS+MLP` (repo `Spades-AI`)
- Collab repo: `Spades_AI_GO-MCTS` (used for prior oracle references)
- Date of snapshot: 2026-05-20

If anything here is unclear or you want me to also create a small smoke-run script (generate a tiny dataset, run 1 epoch training, run the new MCTS leaf for a single state) on the remote machine, say so and I will add it and run it locally.
