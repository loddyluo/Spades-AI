# Spades AI

A Spades card game AI with reinforcement learning (RL), exact double-dummy solving, and a web-based GUI. The project trains an RL policy network (MLP) for the first 4 tricks of each hand and uses an exact solver for the remaining cards. Bidding is handled by a separate MLP model from the GO-MCTS submodule.

---

## Part 1: Quick Start

### Dependencies

- **Python** >= 3.10 (developed on 3.13)
- **PyTorch** >= 2.0 (`torch`, `torch.nn`, `torch.optim`)
- **NumPy**, **tqdm**
- **transformers** (for the collaborator GPT-2 policy/value model)
- **TensorBoard** (`torch.utils.tensorboard`) for training logging
- **Node.js** >= 20 (for the GUI frontend)

A typical install:

```bash
pip install torch numpy tqdm transformers tensorboard
```

### GUI — Play Against the AI

Terminal 1 — Python AI backend:

```bash
python gui/backend.py --port 8001 --exact-threshold 36
```

Terminal 2 — Web dev server:

```bash
cd gui && npm install && npm run dev
```

Then open **http://localhost:5173/** in a browser.

The backend loads two policy checkpoints (`55_2.pt` for non-nil games, `55_2nil.pt` when any player bids nil) and one bid model (`Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt`). If a checkpoint is missing it falls back to random weights or heuristic bidding.

### Evaluation — rl_exact vs DDS

```bash
python rl/eval_rl_multicpu.py --num-games 1500 --seed 61 --num-workers 30 \
    --load-checkpoint ./rl_checkpoints/best.pt
```

Other evaluation modes:

```bash
# DDS vs MCTS (cheating AIs, both see all cards)
python evaluate/evaluate_dds_vs_mcts.py --num-games 128 --num-workers 32

# RL-Exact vs DDS (the main evaluation)
python evaluate/evaluate_dds_vs_rl.py --num-games 128 --num-workers 32

# Rule-Based-First4 vs DDS
python evaluate/eval_rule_first4_multicpu.py --num-games 1500 --num-workers 30
```

### Training — Policy Gradient (Reinforcement Learning)

**Pre-training** (from scratch):

```bash
python rl/pretrain_rl_multicpu.py --num-games 20000 --seed 42 --lr 0.001 \
    --num-workers 32 --update-interval 120 --num-epochs 1 --entropy-coef 0.05
```

**Full RL training** (continues from a pretrained checkpoint):

```bash
python rl/train_rl_multicpu.py --num-games 100000 --seed 114514 --lr 0.000001 \
    --num-workers 30 --update-interval 300 --num-epochs 1 \
    --load-checkpoint ./rl_checkpoints/pretrain/pretrain_best.pt
```

Monitor training:

```bash
tensorboard --logdir runs/rl_train --port 6011
```

**Nil-specific training:**

```bash
python rl/nil_rl_multicpu.py --num-games 500000 --seed 42 --lr 0.0001 \
    --num-workers 32 --update-interval 300 --num-epochs 1 --entropy-coef 0.05
```

Both rl_exact players share the same policy network parameters during training. Training uses a **team match** format (see Part 2).

---

## Part 2: rl_exact Player — How It Works

### Overview

`RLExactPlayer` ([rl/rl_exact_player.py](rl/rl_exact_player.py)) uses two strategies depending on how many cards remain:

1. **Policy network** (first ~16 cards, i.e. remaining > `exact_threshold=36`): an MLP outputs logits over all 52 cards; the highest-scoring legal card is selected (argmax during evaluation, sampled during training).
2. **Exact solver** (last ~36 cards, i.e. remaining <= `exact_threshold`): the double-dummy solver from the DDS library sees all hands (via importance-sampling determinization of opponent hands from public info) and picks the card that maximizes the team's trick total.

### Card-Playing Flow

```
play_card(legal_cards, state_view)
  │
  ├─ remaining = sum(len(h) for h in state.hands)   # cards left in all hands
  │
  ├─ remaining <= exact_threshold (36)?
  │     └─ YES → _exact_play()
  │              Use ExactDoubleDummyCppFastestSolver.
  │              For each legal card, run DDS on each possible
  │              opponent-hand determinization, average the trick counts,
  │              pick the card with the highest expected tricks.
  │
  └─ NO → _policy_play()
           Encode the current state (264-dim feature vector via RLFeatureEncoder).
           Run through PolicyMLP → 55 logits (one per card).
           Mask illegal actions → softmax → pick argmax (eval) or sample (train).
```

### Bidding

Bidding uses the **MLPBidPlayer** from the GO-MCTS bridge ([evaluate/GO-MCTS/models.py](evaluate/GO-MCTS/models.py)), which wraps `BidMLP` (a small MLP trained on double-dummy data). The checkpoint is `Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt`. If the checkpoint is unavailable, bidding falls back to a heuristic based on hand strength.

### Checkpoint Switching (Nil)

The player automatically switches between two policy checkpoints based on the actual bids:
- **No nil bid** → use `55_2.pt` (non-nil policy)
- **Someone bids nil/blind_nil** → use `55_2nil.pt` (nil-specific policy)

This is because nil contracts fundamentally change the optimal play (the nil bidder must avoid taking tricks).

### Team-Match Training (for RL)

During training, both rl_exact players (seats 0+2 or 1+3) use the same shared policy network. Each training iteration plays two games of the same deal with swapped seats, and the reward is:

```
reward = (score_rl_exact_team - score_dds_team)_game1
       + (score_rl_exact_team - score_dds_team)_game2
```

This cancels out the advantage of a specific deal, giving a pure signal about the policy's relative strength vs DDS.

### PolicyMLP Architecture

```
Input:  264-dim feature vector (RLFeatureEncoder)
        ├─ Part 1 (164 dims): bids, played cards (chronological), hand bitmap,
        │                     per-suit counts, led-suit tracking
        └─ Part 2 (100 dims): 10 base features repeated 10× each
                (play position, partner status, legal follow options, nil bids)

Hidden: [1024, 512, 512] with ReLU activations
Output: 55 logits (52 card IDs + 3 special actions)

Training: policy gradient (REINFORCE) with:
  - entropy bonus for exploration
  - advantage normalization across a batch
  - Adam optimizer
```

---

## Part 3: Repository Structure

### Top-Level Directories

```
trick_taking/        Core card-game engine (framework)
strategy/            AI player strategies (MCTS, rule-based, exact)
rl/                  Reinforcement learning training & evaluation
evaluate/            Evaluation scripts and GO-MCTS bridge
gui/                 Web-based UI (React + Vite + Python backend)
data/                Dataset generation utilities
mlp/                 DoubleDummyMLP class (used by MCTS prior)
external/            DDS bridge (cdds / dds-bridge) C library + Python wrapper
Spades_AI_GO-MCTS/   GO-MCTS collaborator submodule (now a regular directory)
```

### [`trick_taking/`](trick_taking/) — Game Engine Foundation

| File | Purpose |
|------|---------|
| `card.py` | `Card`, `Suit`, `Rank` with `card_id` (0-51) encoding and bitmask helpers |
| `deck.py` | Standard 52-card deck (`STANDARD_52`) |
| `game_state.py` | Mutable `GameState` with hands, bids, tricks, phase, `get_player_view()` |
| `game_rules.py` | Abstract rules interface |
| `driver.py` | Game loop — deals, manages phases, calls players |
| `knowledge.py` | Knowledge state for imperfect-information players |
| `player.py` | `AIPlayer` abstract base class |
| `games/spades.py` | Spades rule implementation (`playable()` for legal moves) |
| `players/random_player.py` | Random legal-move player (baseline) |
| `players/human_player.py` | Interactive console player |
| `solvers/` | Double-dummy solver wrappers (DDS C library via ctypes) |
| `utils/feature_encoder.py` | Feature encoder for the old DoubleDummyMLP model |
| `utils/state_tools.py` | State serialization helpers |

**Run tests:**
```bash
python -m pytest trick_taking/tests/
```

### [`strategy/`](strategy/) — AI Player Strategies

| File | Purpose |
|------|---------|
| `truncated_mcts_strategy.py` | MCTS with exact-solver leaf evaluation + determinization |
| `spades_match_runner.py` | Match runner: configures and runs Spades games between AIs |
| `spades_player_programs.py` | Player factory — builds `TruncatedMCTSPlayer`, `RandomSpadesPlayer` |
| `rule_exact_player.py` | Player that uses rule-based bidding + exact solver for card play |
| `rule_exact_first4_player.py` | Rule-based first 4 tricks + exact solver for remaining cards |
| `rule_based_first4_player.py` | Rule-based first 4 tricks + rule-based play |
| `hand_strength.py` | Hand-strength estimator for bidding |

### [`rl/`](rl/) — Reinforcement Learning

| File | Purpose |
|------|---------|
| `rl_exact_player.py` | `RLExactPlayer` — policy network for first 4 tricks + exact solver |
| `rl_feature_encoder.py` | **264-dim feature encoder** for RL (Part 1: 164 dims, Part 2: 100 dims) |
| `policy_network.py` | `PolicyMLP` — the trainable 3-layer MLP policy network |
| `train_rl_multicpu.py` | Multi-CPU REINFORCE training loop with team-match reward |
| `pretrain_rl_multicpu.py` | Pre-training from random weights (faster, higher lr, entropy bonus) |
| `nil_rl_multicpu.py` | Nil-specific training variant |
| `eval_rl_multicpu.py` | Evaluation script (argmax, no exploration) |
| `both_eval.py` | Evaluation that auto-selects nil/non-nil checkpoint based on bids |

**Training commands:**
```bash
# Pre-train (fast, high entropy)
python rl/pretrain_rl_multicpu.py --num-games 20000 --seed 42 --lr 0.001 \
    --num-workers 32 --update-interval 120 --num-epochs 1 --entropy-coef 0.05

# Full training (from pretrained checkpoint)
python rl/train_rl_multicpu.py --num-games 100000 --seed 114514 --lr 0.000001 \
    --num-workers 30 --update-interval 300 --num-epochs 1 \
    --load-checkpoint ./rl_checkpoints/pretrain/pretrain_best.pt

# Evaluate
python rl/eval_rl_multicpu.py --num-games 1500 --seed 61 --num-workers 30 \
    --load-checkpoint ./rl_checkpoints/best.pt

# Both-checkpoint evaluation (auto nil/non-nil switching)
python rl/both_eval.py --num-games 1500 --seed 61 --num-workers 30 \
    --checkpoint-nonil ./55_2.pt --checkpoint-nil ./55_2nil.pt \
    --bid-checkpoint ./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt
```

**Important training parameters:**
- `--num-workers`: number of parallel game-playing processes
- `--update-interval`: games played between policy updates (larger = less variance)
- `--num-epochs`: number of gradient passes per update
- `--entropy-coef`: entropy bonus weight (encourages exploration in pre-training)
- `--load-checkpoint`: resume training from a `.pt` file

### [`evaluate/`](evaluate/) — Evaluation Scripts & GO-MCTS Bridge

| File | Purpose |
|------|---------|
| `dds_player.py` | Perfect-information DDS player (sees all 4 hands) |
| `dds_wrapper.py` | Ctypes wrapper for the DDS C library (SolveBoard) |
| `evaluate_dds_vs_rl.py` | Benchmark: DDS (optimal) vs RL-Exact |
| `evaluate_dds_vs_mcts.py` | Benchmark: DDS vs MCTS |
| `evaluate_cheat_mcts_vs_dds.py` | Benchmark: cheating MCTS vs DDS |
| `evaluate_our_mcts_vs_rule_v2.py` | Benchmark: our MCTS vs rule-based v2 players |
| `eval_rule_first4_multicpu.py` | Benchmark: rule-based-first4 vs DDS |
| `GO-MCTS/` | Bridge to the collaborator models: |
| `GO-MCTS/models.py` | `load_bid_mlp_model()`, `load_gpt2_policy_value_model()` + player exports |
| `GO-MCTS/bridge.py` | State/card conversion helpers (`to_go_state`, `normalize_bid`, etc.) |
| `GO-MCTS/adapters.py` | `GoPlayerAdapter` — wraps collaborator players for the local runner |

### [`gui/`](gui/) — Web Interface

| File | Purpose |
|------|---------|
| `backend.py` | Python HTTP server (no framework — uses `http.server`). Reconstructs `GameState` from frontend payload and calls `RLExactPlayer` |
| `src/game.js` | React-based card game UI with drag-to-play, animations, dual-mode (play alone or vs AI) |
| `src/styles.css` | Card table styling |
| `vite.config.js` | Vite dev server config |

**Run:**
```bash
# Terminal 1: AI backend
python gui/backend.py

# Terminal 2: frontend
cd gui && npm install && npm run dev
```

The backend is stateless — each HTTP request rebuilds the game state from the JSON payload. The frontend never sends hidden opponent cards to the backend; the AI only sees its own hand and public history.

### [`Spades_AI_GO-MCTS/`](Spades_AI_GO-MCTS/) — Collaborator Model Package

Contains the `spades_ai` Python package providing:

- **Bid models**: `BidMLP`, `BidEncoder` (used for bidding)
- **Rule-based players**: `rule_based.player.RuleBasedPlayer` (v1), `rule_based_v2.player.RuleBasedPlayer` (v2)
- **Search**: `GOMCTSConfig`, `GOMCTSPlayer`, `ArgmaxPlayer` (for the GO-MCTS evaluation)
- **Game types**: `Card`, `Suit`, `Rank`, `GameState`, `Bid`, `Trick`, etc. (used for state conversion in the bridge)

This was originally a git submodule of https://github.com/loddyluo/Spades_AI_GO-MCTS. It is now a regular directory with only the files actually needed by the main codebase.

### [`data/`](data/) — Dataset Generation

| File | Purpose |
|------|---------|
| `training_data.py` | Dataset loading, bucket management, state reconstruction utilities |
| `generate_dataset_multi_cpu.py` | Multi-CPU dataset generation (generates .pt files of state/action/value pairs) |

### [`mlp/`](mlp/) — DoubleDummyMLP (Prior Model)

| File | Purpose |
|------|---------|
| `mlp_model.py` | `DoubleDummyMLP` — a dual-head (value + policy) MLP used as an optional prior oracle by `TruncatedMCTSStrategy` |

This is a remnant of the old supervised-learning pipeline, kept only because `TruncatedMCTSStrategy` optionally loads a checkpoint for leaf-value priors during MCTS simulation.

### [`external/`](external/) — DDS Bridge Library

Contains the **dds-bridge** C library source (`external/dds/`) and a build script (`build_dds.sh`). The compiled library is accessed through `trick_taking/solvers/exact_double_dummy_cpp_*.py` wrappers using ctypes. The solver computes optimal play under double-dummy conditions (all hands visible).

---

## RL Feature Encoder Details

The **264-dim feature vector** ([rl/rl_feature_encoder.py](rl/rl_feature_encoder.py)) used by the RL policy network encodes:

| Dimensions | Content |
|-----------|---------|
| 0-3 | My bid, LHO's bid, Partner's bid, RHO's bid (integers; nil=0) |
| 4 | Cards played count (0-15, scalar) |
| 5-94 | Last 15 played cards × 6 dims each: [rank(2-14), spades?, hearts?, diamonds?, clubs?, seat(0-3)] |
| 95-146 | Hand bitmap (52 bits, 1=card in hand) |
| 147-150 | Per-suit count (S/H/D/C) |
| 151-163 | Led-suit tracking (13 cards of led suit, 1=already played) |
| 164-263 | Part 2: 10 base features × 10 repeats for emphasis: play position, partner status, legal follow options, nil bids |

The encoder only uses information available to the player (own hand + public history). It does **not** encode opponent hole cards.

---

## Legacy Player Implementations

The repository also contains these player types that may be useful as baselines:

- **RandomPlayer**: plays a random legal card
- **RuleBasedPlayer** (v1): heuristic rule-based strategy from the collaborator repo
- **RuleBasedPlayer** (v2): improved rule-based strategy with more nuanced heuristics
- **RuleExactPlayer**: rule-based bidding + exact solver for all card play
- **RuleExactFirst4Player**: rule-based first 4 tricks + exact solver for remaining
- **RuleBasedFirst4Player**: fully rule-based (no exact solver)
- **TruncatedMCTSPlayer**: MCTS with determinization and exact-solver leaf evaluation
- **DDSPlayer**: perfect-information AI that uses DDS to pick optimal cards (cheating)
- **MLPBidPlayer**: MLP-based bidder (wraps BidMLP) with rule-based card play
- **ArgmaxPlayer**: GPT-2 policy/value argmax player (from GO-MCTS)
- **GOMCTSPlayer**: full GO-MCTS player (from GO-MCTS)
