# Stochastic Hybrid Residual Bidder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and optionally deploy one reproducibly stochastic bidder whose local Q values are learned from four-trick-plus-DDS counterfactuals and whose final quality is decided by duplicate matches using the unchanged deployed card-play pipeline.

**Architecture:** A new `residual_bidder` package owns the frozen NSFP center, the five-member local-advantage ensemble, the calibrated 14-action distribution, and its deterministic random tape. CPU workers replay complete auctions, use the existing Nil/non-Nil rules for exactly four tricks, and call the native terminal solver once; PyTorch fits the ensemble on divided-by-100 targets. A separate real-play evaluator injects exact per-seat bid likelihoods into the existing `RuleExactFirst4NilPlayer` belief sampler and is the sole authority for calibration, promotion, and final testing.

**Tech Stack:** Python 3.10+; PyTorch with working RTX 5090 `sm_120` kernels; NumPy; PyYAML; pytest; existing `SpadesMatchRunner`, `RuleBasedFirst4NilPlayer`, `RuleExactFirst4NilPlayer`, and `ExactDoubleDummyCppFastestSolver`; JSON, gzip JSONL, compressed NPZ, and SQLite; an AutoDL RTX 5090 instance reached over SSH.

**Approved specification:** [`docs/superpowers/specs/2026-07-21-stochastic-hybrid-residual-bidder-design.md`](../specs/2026-07-21-stochastic-hybrid-residual-bidder-design.md)

## Global Constraints

- Freeze `Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt` at SHA-256 `b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c` and preserve the exact current `to_go_state -> BidEncoder.encode` observation path.
- Freeze the 149-dimensional encoder semantics and `configs/8.yaml` at SHA-256 `658d14e51a42bd340fd2bed039b584ff8dec20f92f00821e818710e8131e9445`; do not repair legacy position, auction, or derived-feature behavior inside this project.
- Legal acting actions are exactly `Nil, bid_1, ..., bid_13`; Blind Nil and normal zero remain unavailable.
- Learn Q values only for the two or three legal actions in the NSFP-centered local neighborhood. Nonlocal actions receive probability only from the geometric tail.
- Store raw score targets and train on `raw / 100`; never clip targets and never replace MSE with Huber loss.
- Use five independent residual MLP members with deal-level deterministic Poisson(1) bootstrap multiplicities and population standard deviation.
- Do not add REINFORCE, PPO, actor-critic, entropy-reward training, full-action Q learning, NSFP fine-tuning, or an argmax-imitation objective.
- Training branches play exactly four complete tricks with the existing Nil/non-Nil rules and call `ExactDoubleDummyCppFastestSolver.solve` exactly once. They must never enter `_exact_play`, construct an IS pool, or play the last nine tricks one by one.
- Evaluate each formal `(deal, state, forced local action, frozen continuation)` branch once. Repetition is permitted only in deterministic tests and never contributes multiple training rows.
- Real deployed card play is evaluation-only. It must not create Q labels, replay rows, or optimizer input.
- Both seats of one partnership share one bidder checkpoint and calibration tuple. Both opponent seats share one immutable league member selected for the whole duplicate deal.
- Acting and belief probabilities for every residual bidder must come from the same distribution implementation. Legacy NSFP and unknown external seats retain separately declared likelihood adapters.
- Shuffle seeds and policy seeds are independent. Sampling uses a hash-derived uniform variate keyed by policy seed, deal ID, room ID, canonical physical seat, and bid index; mutable global RNG state may not affect a bid.
- Formal generation, development, promotion, and final testing fail on missing artifacts, identity drift, invalid probabilities, NaN/Inf, solver unavailability, incorrect four-trick boundaries, or partition leakage. Only interactive runtime may fall back to deterministic NSFP argmax, and it must record that provenance.
- Use five disjoint deal namespaces: training, fast-hybrid development, complete-play development, per-candidate promotion, and one-time final test.
- Preserve all unrelated dirty-worktree files. Every task stages only its listed paths; never run `git add .`.
- Treat SSH credentials as ephemeral secrets: never place a password in the repository, command line, process listing, saved shell history, run manifest, or response text. Prefer an SSH key if the rented instance permits one.
- Do not start the first production data block until the AutoDL smoke run has reported measured deals/hour, CPU utilization, GPU fit throughput, storage growth, and all invariants, and the user has approved proceeding.

## Verified Baseline at Plan Time

- Local Python is `3.13.13`; local PyTorch is `2.12.0`; CUDA is unavailable locally, so GPU checks belong on the cloud runtime.
- The native fast solver is already present for local macOS and Linux builds; a long run still requires a fresh native build-ID and throughput report.
- A 20-state local benchmark previously measured one four-trick-boundary terminal solve at median `0.023 s`, mean `0.055 s`, and maximum `0.477 s`.
- The AutoDL instance has not yet been rented, so CPU count, RAM, disk, driver, CUDA runtime, image, SSH port, and persistent-workspace path are unknown. The remote preflight records them rather than assuming any previous cloud machine's values.

---

## File and Module Map

| Path | Responsibility |
|---|---|
| `requirements-bidder.txt` | Non-Torch Python dependencies; torch installation remains an explicit cloud-runtime decision. |
| `configs/residual_bidder/base.yaml` | Frozen hashes, architecture, policy grids, worker counts, partitions, training, stopping, promotion, and storage configuration. |
| `residual_bidder/config.py` | Typed configuration loading, exact validation, canonical serialization, and config hashing. |
| `residual_bidder/actions.py` | Canonical 14 actions, 16-to-14 aliasing, neighborhoods, masks, and bid conversion. |
| `residual_bidder/nsfp.py` | Hash-checked frozen NSFP loader and exact current observation bridge. |
| `residual_bidder/model.py` | 167-vector builder, residual blocks, one member, and the five-member ensemble. |
| `residual_bidder/checkpoint.py` | Atomic model/checkpoint schema, status, calibration tuple, hashes, and compatibility checks. |
| `residual_bidder/random_tape.py` | Domain-separated deterministic policy variates independent of shuffle RNG. |
| `residual_bidder/policy.py` | NSFP argmax, residual local values, geometric tail, sampling, strict failures, and runtime fallback. |
| `residual_bidder/likelihood.py` | Batched legacy, residual, and external likelihood adapters plus per-seat auction likelihoods. |
| `residual_bidder/player_proxy.py` | Bid-only policy override with transparent forwarding of all card-play callbacks. |
| `residual_bidder/hybrid.py` | Duplicate baseline capture, local forced branches, first-four play, one DDS solve, and target assembly. |
| `residual_bidder/seeds.py` | Disjoint deal namespaces and independent shuffle/policy seed derivation. |
| `residual_bidder/records.py` | Versioned deal/state/branch records and invariant validation. |
| `residual_bidder/shards.py` | Atomic gzip-JSONL/NPZ shards, hashes, manifests, and duplicate-deal admission. |
| `residual_bidder/reservoir.py` | Outcome-blind stratified Algorithm-R membership and natural-distribution corrections. |
| `residual_bidder/league.py` | Immutable policy manifests, exact opponent weights, deterministic selection, and acceptance. |
| `residual_bidder/dataset.py` | Deal-grouped fit splits, natural/reservoir mixture sampling, masks, and bootstrap counts. |
| `residual_bidder/training.py` | Five-member MSE fitting, resume identity, metrics, probes, and candidate artifacts. |
| `residual_bidder/evaluation.py` | Fast-hybrid and complete deployed-play duplicate evaluators using common random numbers. |
| `residual_bidder/calibration.py` | Grid canonicalization, successive halving, mandatory deterministic control, real-play shortlist, and tie rule. |
| `residual_bidder/promotion.py` | Fixed-size opponent-stratified deal bootstrap and all three promotion gates. |
| `residual_bidder/iteration.py` | Frozen-continuation state machine, block growth, stability stopping, and league advancement. |
| `residual_bidder/cli/*.py` | `preflight`, `generate`, `train`, `calibrate`, `evaluate`, and `iterate` commands. |
| `strategy/rule_exact_first4_player.py` | Inject per-seat likelihoods into the current deployed belief pool while preserving the default legacy adapter. |
| `strategy/rule_exact_first4_nil_player.py` | Forward the likelihood dependency through the deployed Nil subclass. |
| `gui/backend.py`, `gui/game_server.py`, `gui/src/game.js` | Optional promoted bidder, fixed deal policy tape, and runtime provenance; default behavior remains unchanged. |
| `scripts/autodl/bootstrap_bidder.sh` | Idempotent remote venv, dependency, native-solver, directory, and preflight setup without embedded credentials. |
| `scripts/autodl/run_bidder.sh` | `flock`-guarded resumable smoke/production launcher with PID, logs, and clean signal handling. |
| `tests/residual_bidder/*.py` | Unit, identity, hybrid, data, statistical, integration, CLI, and manifest tests. |

## Task 1: Establish the Package, Typed Configuration, and Runtime Gate

**Files:**
- Create: `requirements-bidder.txt`
- Create: `configs/residual_bidder/base.yaml`
- Create: `residual_bidder/__init__.py`
- Create: `residual_bidder/config.py`
- Create: `residual_bidder/cli/__init__.py`
- Create: `residual_bidder/cli/preflight.py`
- Create: `tests/residual_bidder/test_config.py`
- Create: `tests/residual_bidder/test_preflight.py`

**Interfaces:**
- Produces: `BidderConfig.load(path: Path) -> BidderConfig`, `BidderConfig.sha256() -> str`, and `inspect_runtime(config, require_cuda, require_sm120) -> dict[str, object]`.
- Consumed by: every later task and every CLI.

- [ ] **Step 1: Write the failing exact-config tests**

```python
def test_base_config_has_frozen_contract():
    cfg = BidderConfig.load(Path("configs/residual_bidder/base.yaml"))
    assert cfg.schema == "stochastic-hybrid-residual-bidder-v1"
    assert cfg.nsfp.sha256 == "b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c"
    assert cfg.play.exact_threshold == 36
    assert cfg.play.enable_nil is True
    assert cfg.play.enable_blind_nil is False
    assert cfg.model.members == 5
    assert cfg.model.input_dim == 167
    assert cfg.model.output_dim == 2
    assert cfg.targets.divisor == 100.0
    assert cfg.workers.outer == 0  # auto: max(1, affinity-aware CPU count - 2)
    assert cfg.workers.nested_exact == 1

def test_config_rejects_unknown_or_inconsistent_values(tmp_path):
    raw = yaml.safe_load(Path("configs/residual_bidder/base.yaml").read_text())
    raw["targets"]["clip"] = 5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown key.*clip"):
        BidderConfig.load(path)
```

- [ ] **Step 2: Run the config tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_config.py
```

Expected: collection fails because `residual_bidder.config` does not exist.

- [ ] **Step 3: Create the complete base configuration**

Use these exact initial grids and thresholds; they are configurable only by creating a new hashed run manifest before reading its development results:

```yaml
schema: stochastic-hybrid-residual-bidder-v1
nsfp:
  path: Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt
  sha256: b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c
play:
  config_path: configs/8.yaml
  config_sha256: 658d14e51a42bd340fd2bed039b584ff8dec20f92f00821e818710e8131e9445
  source_manifest:
    - strategy/spades_match_runner.py
    - strategy/rule_based_first4_player.py
    - strategy/rule_based_first4_nil_player.py
    - strategy/rule_exact_first4_player.py
    - strategy/rule_exact_first4_nil_player.py
    - trick_taking/games/spades.py
    - trick_taking/solvers/exact_double_dummy_cpp_fastest.py
    - trick_taking/solvers/exact_double_dummy_cpp_fastest_core.cpp
  exact_threshold: 36
  first_tricks: 4
  enable_nil: true
  enable_blind_nil: false
targets: {divisor: 100.0}
model:
  members: 5
  input_dim: 167
  hidden_dim: 256
  bottleneck_dim: 128
  output_dim: 2
  margin_divisor: 13.47
  init_seeds: [1701, 1702, 1703, 1704, 1705]
policy:
  policy_seed: 20260721
  lambda_grid: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
  temperature_grid: [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
  epsilon_grid: [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
  rho_grid: [0.25, 0.5, 0.75, 1.0]
workers: {outer: 0, nested_exact: 1}
data:
  block_deals: 256
  shard_deals: 64
  natural_fraction: 0.75
  reservoir_fraction: 0.25
  reservoir_capacity_per_stratum: 2048
  fixed_probe_states: 10000
  minimum_blocks: 3
training:
  batch_size: 4096
  learning_rate: 0.0003
  weight_decay: 0.0001
  max_epochs: 200
  early_stop_patience: 15
  gradient_norm_clip: 1.0
  precision: float32
calibration:
  halving_eta: 4
  round_deals: [256, 1024, 4096]
  real_play_shortlist: 8
  real_play_deals: 256
promotion:
  alpha_one_sided: 0.05
  power: 0.80
  minimum_detectable_points: 1.0
  minimum_deals: 4096
  round_to_deals: 256
  bootstrap_resamples: 10000
  protected_stratum_fraction: 0.05
  protected_behavior_strata:
    - incumbent_nil_presence
    - incumbent_tail_presence
    - incumbent_first_center_bucket
stopping:
  latest_blocks: 3
  probability_l1: 0.01
  changed_probe_fraction: 0.005
storage:
  run_dir: output/residual-bidder/main
```

- [ ] **Step 4: Create the non-Torch dependency file**

`requirements-bidder.txt` contains exactly:

```text
numpy>=1.26,<3
PyYAML>=6,<7
psutil>=5.9,<8
pytest>=8,<9
tqdm>=4.66,<5
```

Do not add a torch index or version here; the AutoDL base image's torch is accepted only if the capability preflight runs real `sm_120` forward/backward kernels successfully.

- [ ] **Step 5: Implement strict typed loading and canonical hashing**

Use frozen dataclasses, reject every unknown key recursively, validate all domains (`lambda >= 0`, `T >= 0`, `0 <= epsilon <= 1`, `0 < rho <= 1`), require the deterministic grid point, and hash canonical JSON:

```python
def canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class BidderConfig:
    schema: str
    nsfp: NSFPConfig
    play: PlayConfig
    targets: TargetConfig
    model: ModelConfig
    policy: PolicyGridConfig
    workers: WorkerConfig
    data: DataConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    promotion: PromotionConfig
    stopping: StoppingConfig
    storage: StorageConfig
```

- [ ] **Step 6: Write failing runtime-gate tests with fakes**

Require rejection of unavailable CUDA when requested, missing `sm_120`, non-finite forward/backward values, missing native solver, and frozen-file hash drift. A success report must include Python, torch, compiled CUDA, GPU name, compute capability, architecture list, native solver build ID, and every frozen hash.

```python
def test_preflight_rejects_missing_sm120(fake_torch, config):
    fake_torch.cuda.get_arch_list.return_value = ["sm_80", "sm_90"]
    with pytest.raises(PreflightError, match="sm_120"):
        inspect_runtime(config, require_cuda=True, require_sm120=True,
                        torch_module=fake_torch, solver_factory=FakeNativeSolver)
```

- [ ] **Step 7: Implement and verify the preflight CLI**

The GPU path runs a fixed-seed float32 matrix multiply and backward pass, synchronizes CUDA, and rejects any non-finite output or gradient. The solver path instantiates `ExactDoubleDummyCppFastestSolver`, requires `native_available`, and records the loaded native-library digest. Hash the ordered `play.source_manifest` as `(relative path, file SHA-256)` pairs plus `play.config_sha256`; every generator/evaluator/checkpoint records that resulting `play_pipeline_sha256`.

```bash
python -m residual_bidder.cli.preflight \
  --config configs/residual_bidder/base.yaml \
  --require-cuda --require-sm120 --require-native-solver
```

Expected on a valid 5090 runtime: JSON with `"compute_capability":[12,0]`, `"sm_120":true`, `"native_solver":true`, and `"ok":true`. Local unit tests use fakes and must pass without CUDA.

- [ ] **Step 8: Run Task 1 tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_config.py \
  tests/residual_bidder/test_preflight.py
git add requirements-bidder.txt configs/residual_bidder/base.yaml \
  residual_bidder/__init__.py residual_bidder/config.py \
  residual_bidder/cli/__init__.py residual_bidder/cli/preflight.py \
  tests/residual_bidder/test_config.py tests/residual_bidder/test_preflight.py
git commit -m "build: gate stochastic bidder runtime"
```

## Task 2: Freeze the NSFP Observation and Canonical 14-Action Space

**Files:**
- Create: `residual_bidder/actions.py`
- Create: `residual_bidder/nsfp.py`
- Create: `tests/residual_bidder/test_actions.py`
- Create: `tests/residual_bidder/test_nsfp.py`
- Reference unchanged: `evaluate/GO-MCTS/bridge.py`
- Reference unchanged: `Spades_AI_GO-MCTS/spades_ai/models/bid_encoder.py`
- Reference unchanged: `Spades_AI_GO-MCTS/spades_ai/models/bid_mlp.py`

**Interfaces:**
- Produces: `BidAction`, `LocalNeighborhood`, `legal_scores_14`, `FrozenNSFP.observe`, and `FrozenNSFP.observe_batch`.
- Consumed by: Tasks 3-13.

- [ ] **Step 1: Write failing action normalization tests**

```python
class BidAction(IntEnum):
    NIL = 0
    BID_1 = 1
    BID_2 = 2
    BID_3 = 3
    BID_4 = 4
    BID_5 = 5
    BID_6 = 6
    BID_7 = 7
    BID_8 = 8
    BID_9 = 9
    BID_10 = 10
    BID_11 = 11
    BID_12 = 12
    BID_13 = 13

@dataclass(frozen=True)
class LocalNeighborhood:
    center: BidAction
    lower: BidAction | None
    upper: BidAction | None

def legal_scores_14(raw_logits_16: torch.Tensor) -> torch.Tensor: ...
def choose_center(scores_14: torch.Tensor) -> BidAction: ...
def neighborhood(center: BidAction) -> LocalNeighborhood: ...
def to_local_bid(action: BidAction) -> str: ...
def from_local_bid(value: str) -> BidAction: ...
```

Tests must assert `Nil=max(raw[14],raw[15])`, `bid_1=max(raw[0],raw[1])`, bids 2-13 copy raw slots 2-13, stable ties choose the lowest canonical action index, and neighborhoods are exactly `{Nil,1}`, `{Nil,1,2}`, `{c-1,c,c+1}`, and `{12,13}` at the four boundaries.

- [ ] **Step 2: Write failing observation-equivalence and no-leak tests**

```python
@dataclass(frozen=True)
class NSFPObservation:
    encoded_149: torch.Tensor
    raw_logits_16: torch.Tensor
    legal_scores_14: torch.Tensor
    center: BidAction

class FrozenNSFP:
    @classmethod
    def load(cls, checkpoint: Path, expected_sha256: str,
             device: torch.device) -> "FrozenNSFP": ...
    def observe(self, state: GameState) -> NSFPObservation: ...
    def observe_batch(self, states: Sequence[GameState]) -> list[NSFPObservation]: ...
```

For dealer positions 0-3 and bidding positions 0-3, compare the first tensor bit-for-bit to:

```python
go_state = to_go_state(state)
expected = BidEncoder().encode(
    list(go_state.hands[go_state.current_player]),
    list(go_state.bids),
    len(go_state.bids),
)
```

Mutate all three non-acting hands while preserving the acting hand and public auction; `encoded_149`, logits, scores, and center must remain identical.

- [ ] **Step 3: Run Task 2 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_actions.py \
  tests/residual_bidder/test_nsfp.py
```

Expected: missing modules and interfaces.

- [ ] **Step 4: Implement the action mapping and frozen model**

Load `BidMLP` with `weights_only=True`, verify the checkpoint before deserialization, call `requires_grad_(False)` and `eval()`, validate all ranks/shapes/finite values, and reuse `to_go_state` without copying its logic into a corrected encoder. Do not expose hidden-layer activations.

```python
with torch.inference_mode():
    raw = self.model(encoded_149.unsqueeze(0).to(self.device)).squeeze(0).cpu()
scores = legal_scores_14(raw)
return NSFPObservation(encoded_149.cpu(), raw, scores, choose_center(scores))
```

- [ ] **Step 5: Verify GREEN, run a checkpoint characterization, and commit**

Run the Task 2 test command plus a 1,000-state fixed-seed characterization. For every state, alias-normalizing the raw NSFP argmax must match `choose_center(legal_scores_14(raw))`; exact constructed ties follow the canonical lower-index rule.

```bash
git add residual_bidder/actions.py residual_bidder/nsfp.py \
  tests/residual_bidder/test_actions.py tests/residual_bidder/test_nsfp.py
git commit -m "feat: freeze nsfp bidder observations"
```

## Task 3: Build the 167-Vector, Five Independent Members, and Atomic Checkpoint

**Files:**
- Create: `residual_bidder/model.py`
- Create: `residual_bidder/checkpoint.py`
- Create: `tests/residual_bidder/test_model.py`
- Create: `tests/residual_bidder/test_checkpoint.py`

**Interfaces:**
- Consumes: `NSFPObservation` and `LocalNeighborhood` from Task 2.
- Produces: `ResidualInput`, `ResidualQMember`, `ResidualQEnsemble`, `CalibrationTuple`, and atomic checkpoint load/save.

- [ ] **Step 1: Write failing exact-feature tests**

```python
@dataclass(frozen=True)
class ResidualInput:
    values: torch.Tensor       # shape (167,)
    neighborhood: LocalNeighborhood
    alternative_mask: torch.Tensor  # shape (2,), lower then upper

def build_residual_input(obs: NSFPObservation,
                         margin_divisor: float = 13.47) -> ResidualInput: ...
```

Assert `[0:149]` is bit-identical to the frozen tensor, `[149:163]` is the canonical center one-hot, `[163]` and `[164]` are center-minus-lower/upper score margins divided by `13.47`, and `[165:167]` are lower/upper masks. Missing alternatives must have margin and mask zero.

The builder accepts only `NSFPObservation`; its signature and tests must make it impossible to pass complete deals, opponent hands, future bids, played cards, DDS output, or NSFP hidden activations into the residual model.

- [ ] **Step 2: Write failing architecture and independence tests**

```python
class ResidualQMember(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (...,167)->(...,2)

class ResidualQEnsemble(nn.Module):
    members: nn.ModuleList
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (5,...,2)
    def mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Each member is exactly:

```text
Linear(167,256) -> LayerNorm(256) -> SiLU
-> ResidualBlock(256) -> ResidualBlock(256)
-> Linear(256,128) -> LayerNorm(128) -> SiLU -> Linear(128,2)
```

Each residual block is `Linear -> LayerNorm -> SiLU -> Linear -> add input -> SiLU`. Assert no dropout, no shared parameter storage, exact output shapes, deterministic eval output, and `std(unbiased=False)`.
Output index 0 is always the lower-action advantage and index 1 is always the upper-action advantage. A missing boundary alternative is masked from loss and inference rather than remapped to the other slot.

- [ ] **Step 3: Write failing checkpoint identity tests**

```python
@dataclass(frozen=True)
class CalibrationTuple:
    uncertainty_lambda: float
    temperature: float
    epsilon: float
    rho: float

@dataclass(frozen=True)
class BidderCheckpointMeta:
    schema: str
    status: Literal["candidate", "promoted"]
    model_id: str
    policy_id: str | None
    iteration: int
    nsfp_sha256: str
    play_pipeline_sha256: str
    config_sha256: str
    dataset_manifest_sha256: str
    member_init_seeds: tuple[int, int, int, int, int]
    calibration: CalibrationTuple | None
```

Round-trip all five state dicts and metadata. An uncalibrated candidate has a content-addressed `model_id`, `policy_id=None`, and `calibration=None`; calibration later fills a derived policy ID. Reject unknown schema, non-finite parameters, wrong dimensions/member count, changed frozen hashes, a promoted status without calibration/policy ID, and a candidate checkpoint passed to a promoted-only loader.

- [ ] **Step 4: Run Task 3 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_model.py \
  tests/residual_bidder/test_checkpoint.py
```

- [ ] **Step 5: Implement features, model, and recoverable checkpoint writes**

Seed and construct each member independently. Save under a sibling temporary file, flush and `os.fsync`, reopen and validate the full artifact, then `os.replace` and fsync the parent directory. Never pickle arbitrary Python objects; store a plain tensor state dict and JSON-compatible metadata.

- [ ] **Step 6: Verify GREEN and commit**

```bash
git add residual_bidder/model.py residual_bidder/checkpoint.py \
  tests/residual_bidder/test_model.py tests/residual_bidder/test_checkpoint.py
git commit -m "feat: add residual q ensemble checkpoint"
```

## Task 4: Construct and Sample the Reproducible 14-Action Policy

**Files:**
- Create: `residual_bidder/random_tape.py`
- Create: `residual_bidder/policy.py`
- Create: `tests/residual_bidder/test_random_tape.py`
- Create: `tests/residual_bidder/test_policy.py`

**Interfaces:**
- Consumes: frozen NSFP, 167-vector builder, ensemble, and `CalibrationTuple`.
- Produces: `BidSamplingKey`, `BidDistribution`, `BidDecision`, `ActingBidPolicy`, `NSFPArgmaxPolicy`, and `StochasticResidualPolicy`.

- [ ] **Step 1: Write failing policy-tape tests**

```python
@dataclass(frozen=True)
class BidSamplingKey:
    policy_seed: int
    deal_id: str
    room_id: str
    logical_seat: int
    bid_index: int

def policy_uniform(key: BidSamplingKey) -> float: ...
```

Derive a uniform in the open interval `(0,1)` from BLAKE2b over the domain `residual-bidder-policy-tape-v1` and the five canonical fields. Assert restart identity, separation of each field, independence from the shuffle seed, and independence from checkpoint/policy ID because neither is an input field.

Serialize the key as canonical UTF-8 JSON, request an 8-byte digest, and convert it exactly as follows so every platform consumes the same variate:

```python
integer = int.from_bytes(digest, byteorder="big", signed=False)
uniform = (integer + 0.5) / float(1 << 64)
```

- [ ] **Step 2: Write failing distribution-formula tests**

```python
@dataclass(frozen=True)
class BidDistribution:
    probabilities: tuple[float, ...]  # exactly 14
    center: BidAction
    local_values: tuple[float | None, float, float | None]  # lower, center, upper
    policy_id: str

def geometric_tail(center: BidAction, rho: float) -> torch.Tensor: ...
def stable_inverse_cdf(probabilities: Sequence[float], u: float) -> BidAction: ...
```

Use exact fixtures for:

```python
adjusted_lower = mean_lower - uncertainty_lambda * std_lower
adjusted_center = 0.0
adjusted_upper = mean_upper - uncertainty_lambda * std_upper
local = softmax(torch.tensor(valid_values) / temperature, dim=0)
tail[a] = rho ** abs(int(a) - int(center))
tail /= tail.sum()
final = (1.0 - epsilon) * local_scattered_to_14 + epsilon * tail
```

At `temperature=0`, select a stable local argmax with tie order center, lower, upper and emit a one-hot local core. Assert `rho=1` is uniform, all 14 probabilities are positive iff `epsilon>0`, the vector is finite/nonnegative/sums to one, and `temperature=0, epsilon=0` is deterministic.

- [ ] **Step 3: Write failing strict/fallback and empirical-sampling tests**

```python
@dataclass(frozen=True)
class BidDecision:
    action: BidAction
    distribution: BidDistribution
    uniform: float
    effective_policy_id: str
    fallback_reason: str | None

class StochasticResidualPolicy:
    def probabilities(self, state: GameState, *, strict: bool) -> BidDistribution: ...
    def probabilities_batch(self, states: Sequence[GameState], *,
                            strict: bool) -> list[BidDistribution]: ...
    def sample(self, state: GameState, legal_bids: Sequence[object],
               key: BidSamplingKey, *, strict: bool) -> BidDecision: ...
```

Both concrete policies satisfy this protocol:

```python
class ActingBidPolicy(Protocol):
    policy_id: str
    def probabilities(self, state: GameState, *, strict: bool) -> BidDistribution: ...
    def sample(self, state: GameState, legal_bids: Sequence[object],
               key: BidSamplingKey, *, strict: bool) -> BidDecision: ...
```

Wrong dimensions, invalid calibration, NaN/Inf, hash drift, or a missing expected legal bid must raise in strict mode. Runtime mode must return deterministic normalized NSFP argmax with `effective_policy_id="legacy-nsfp-fallback"` and a structured reason. Over at least 200,000 fixed-key variants, empirical frequencies must be within a precomputed binomial tolerance of the declared vector.

- [ ] **Step 4: Run Task 4 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_random_tape.py \
  tests/residual_bidder/test_policy.py
```

- [ ] **Step 5: Implement the shared distribution path once**

Both `probabilities` and `sample` must call one private `_distribution_from_observation` implementation. Scatter local mass by canonical `BidAction`, sum in float64, normalize once after the mixture, and convert to Python floats only at the API boundary. Sampling must consume the supplied uniform and must not call `random`, NumPy RNG, or torch RNG.

All configured temperature values are in divided-by-100 Q units; do not multiply Q back to raw score points before the local softmax.

- [ ] **Step 6: Verify deterministic reruns, probability calibration, and commit**

```bash
git add residual_bidder/random_tape.py residual_bidder/policy.py \
  tests/residual_bidder/test_random_tape.py tests/residual_bidder/test_policy.py
git commit -m "feat: sample calibrated stochastic bids"
```

## Task 5: Make Deployed Belief Weighting Policy-Aware Per Seat

**Files:**
- Create: `residual_bidder/likelihood.py`
- Create: `tests/residual_bidder/test_likelihood.py`
- Modify: `strategy/rule_exact_first4_player.py:128-181,833-968,1240-1300`
- Modify: `strategy/rule_exact_first4_nil_player.py:36-61`
- Modify: `tests/test_rule_exact_first4_player.py`
- Modify: `tests/test_rule_exact_first4_nil_player.py`

**Interfaces:**
- Consumes: `StochasticResidualPolicy.probabilities_batch` from Task 4.
- Produces: `AuctionEvent`, `ObservedAuction`, three likelihood adapters, `SeatLikelihoods`, and `AuctionLikelihoodEvaluator.batch_products`.
- Preserves: with no injected seat map, the deployed exact player uses the current all-NSFP softened likelihood behavior.

- [ ] **Step 1: Write failing auction/provenance tests**

```python
@dataclass(frozen=True)
class AuctionEvent:
    seat: int
    action: BidAction

@dataclass(frozen=True)
class ObservedAuction:
    dealer_seat: int
    events: tuple[AuctionEvent, ...]

    @property
    def max_bid_by_seat(self) -> tuple[BidAction, BidAction, BidAction, BidAction]:
        by_seat: list[BidAction | None] = [None, None, None, None]
        for event in self.events:
            by_seat[event.seat] = event.action
        if any(action is None for action in by_seat):
            raise ValueError("auction does not contain four actual bids")
        return tuple(by_seat)  # type: ignore[return-value]

@dataclass(frozen=True)
class SeatLikelihoods:
    adapters: tuple["BidLikelihoodAdapter", "BidLikelihoodAdapter",
                    "BidLikelihoodAdapter", "BidLikelihoodAdapter"]

class BidLikelihoodAdapter(Protocol):
    adapter_id: str
    def action_probabilities_batch(
        self,
        deals: Sequence[Sequence[Sequence[Card]]],
        auction: ObservedAuction,
        event_index: int,
    ) -> np.ndarray: ...  # shape (num_deals,14)
```

Reject repeated/non-canonical seats, missing one actual bid, invalid actions, and a seat-policy map of length other than four. Formal rules use no pass; runtime trace conversion may ignore the automatic Blind-Nil decline but must preserve the actual four bids and dealer.

- [ ] **Step 2: Characterize the existing legacy adapter before refactoring**

For 32 fixed proposed deals and auctions, call the current `_compute_batch_bid_prods` and save only the expected numeric products in the test fixture. Implement `LegacyNSFPLikelihoodAdapter` to preserve these exact semantics:

```python
raw16 = torch.softmax(bid_mlp(encoded_batch), dim=-1)
softened16 = 0.99 * raw16 + 0.01 * (1.0 / 14.0)
observed_raw_index = (
    14 if action == BidAction.NIL
    else 1 if action == BidAction.BID_1
    else int(action)
)
```

For this legacy adapter only, previous bids and `position=min(seat,2)` remain in seat-index order exactly as in the current code. Do not silently renormalize the 16 softened values.

- [ ] **Step 3: Write residual and external adapter consistency tests**

`ResidualLikelihoodAdapter` must reconstruct each hypothetical bidding snapshot from the proposed full deal, dealer, and chronological auction prefix, then call the same policy `probabilities_batch` used for acting. Assert bit-identical 14-vectors for an actual deal snapshot and its reconstructed hypothesis. `ExternalUniformLikelihoodAdapter` returns exactly `1/14` for every action and is used only when provenance explicitly says human/unknown.

For event `i`, initialize a local `GameState` with all four proposed hands, set its dealer/current bidder, replay exactly events `[0:i]` into both `state.bids` and the appropriate `state.max_bid` seat slots, then invoke the ordinary residual policy on that state. This deliberately preserves the current bridge's seat-index layout while reconstructing precisely which bid slots were populated at the acting decision.

```python
class AuctionLikelihoodEvaluator:
    def batch_products(self, deals, auction: ObservedAuction,
                       seats: SeatLikelihoods) -> np.ndarray:
        products = np.ones(len(deals), dtype=np.float64)
        for event_index, event in enumerate(auction.events):
            probs = seats.adapters[event.seat].action_probabilities_batch(
                deals, auction, event_index)
            products *= probs[:, int(event.action)]
        return products
```

- [ ] **Step 4: Run the likelihood and existing exact-player tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_likelihood.py \
  tests/test_rule_exact_first4_player.py \
  tests/test_rule_exact_first4_nil_player.py
```

- [ ] **Step 5: Inject one likelihood dependency into `RuleExactFirst4Player`**

Add one optional constructor argument and preserve default behavior:

```python
def __init__(self, ..., bid_likelihoods: SeatLikelihoods | None = None) -> None:
    ...
    self._bid_likelihoods = bid_likelihoods

def _compute_batch_bid_prods(self, proposals, auction):
    if self._bid_likelihoods is None:
        return self._legacy_batch_bid_prods(proposals, auction.max_bid_by_seat)
    return self._auction_likelihood_evaluator.batch_products(
        proposals, auction, self._bid_likelihoods).tolist()
```

Build `ObservedAuction` from `state.bids` plus `state.dealer_seat` inside `_build_is_pool`. If an old caller provides only four `state.max_bid` values, permit that representation only for the default all-legacy adapter; an injected residual adapter requires the chronological trace and fails closed. Forward the constructor argument unchanged through `RuleExactFirst4NilPlayer`.

- [ ] **Step 6: Verify default equivalence and mixed-seat behavior**

Require:

1. the 32 legacy fixture products are bit-identical;
2. fixed default-player card choices remain identical on existing tests;
3. a mixed map `(residual, external, residual, external)` calls the correct adapter for each observed bid;
4. changing a residual checkpoint changes only that seat's likelihood factor;
5. invalid probabilities abort formal evaluation rather than reverting to all-NSFP.

- [ ] **Step 7: Commit the belief integration**

```bash
git add residual_bidder/likelihood.py tests/residual_bidder/test_likelihood.py \
  strategy/rule_exact_first4_player.py strategy/rule_exact_first4_nil_player.py \
  tests/test_rule_exact_first4_player.py tests/test_rule_exact_first4_nil_player.py
git commit -m "feat: weight bids by actual seat policies"
```

## Task 6: Implement Bid-Only Proxies and the Four-Trick Hybrid Evaluator

**Files:**
- Create: `residual_bidder/player_proxy.py`
- Create: `residual_bidder/hybrid.py`
- Create: `tests/residual_bidder/test_player_proxy.py`
- Create: `tests/residual_bidder/test_hybrid.py`
- Reference unchanged: `strategy/spades_match_runner.py`
- Reference unchanged: `strategy/rule_based_first4_player.py`
- Reference unchanged: `strategy/rule_based_first4_nil_player.py`
- Reference unchanged: `trick_taking/solvers/exact_double_dummy_cpp_fastest.py`

**Interfaces:**
- Consumes: acting policies and random tape from Task 4.
- Produces: transparent proxy, captured baseline states, forced directives, hybrid room/duplicate results, and local raw/scaled targets.

- [ ] **Step 1: Write proxy characterization tests**

```python
@dataclass(frozen=True)
class ProxySamplingContext:
    deal_id: str
    room_id: str
    logical_seat: int
    bid_index: int

class ActingBidPlayerProxy(AIPlayer):
    def __init__(self, play_player: AIPlayer, acting_policy: ActingBidPolicy,
                 context: ProxySamplingContext,
                 forced_action: BidAction | None = None) -> None: ...
```

The proxy overrides only `place_bid`. It forwards `start_game`, `bid_placed`, `set_teams`, `play_card`, and `card_played` exactly once and in order. With no forced action it samples the acting policy using the supplied tape key; with a forced action it verifies that action is legal and records it without sampling. A spy player must receive object-identical callback arguments.

- [ ] **Step 2: Write failing baseline-capture and replay tests**

```python
@dataclass(frozen=True)
class CapturedBidState:
    state_id: str
    room_id: str
    bidder: int
    candidate_ordinal: int
    observation: NSFPObservation
    residual_input: ResidualInput
    baseline_action: BidAction
    baseline_from_tail: bool
    auction_prefix: tuple[AuctionEvent, ...]

@dataclass(frozen=True)
class ForcedBidDirective:
    state_id: str
    bidder: int
    candidate_ordinal: int
    action: BidAction
    observation_sha256: str
```

Replay always starts from the original shuffle seed with fresh players. Before forcing, assert bidder, candidate ordinal, public auction prefix, center/neighborhood, and 149-vector digest. Later seats reuse their original hash-derived variates; no mutable RNG state is copied.

Only the target candidate call is forced. Every earlier and later candidate-team call uses the frozen incumbent policy, and every opponent-team call uses the one league member selected for the entire duplicate deal; both rooms and all branches preserve those identities.

- [ ] **Step 3: Write the one-solver-call and score-perspective tests**

```python
@dataclass(frozen=True)
class HybridRoomResult:
    room_id: str
    candidate_team: int
    candidate_margin: float
    terminal_team0_minus_team1: float
    captured_states: tuple[CapturedBidState, ...]
    auction: ObservedAuction
    solver_calls: int
```

Construct all four play players as `RuleBasedFirst4NilPlayer` with fresh state and the bid proxy. Run `SpadesMatchRunner(..., max_tricks=4, rules=SpadesRules(enable_nil=True, enable_blind_nil=False))`. At return assert empty table, `tricks_played==4`, each hand has 9 cards, 36 total cards remain, and call `solver.solve(runner.state)` once. The solver value is team-0 minus team-1; negate it when the candidate partnership is team 1.

- [ ] **Step 4: Write duplicate branch-count and target tests**

Room 0 assigns candidate seats `(0,2)` and room 1 assigns candidate seats `(1,3)` on the identical deal/dealer. Each room has two captured candidate observations. For every state, evaluate every local action and use:

```python
raw_target[action] = candidate_margin[action] - candidate_margin[center]
scaled_target[action] = raw_target[action] / 100.0
```

The center target is exactly zero. Reuse a local baseline result; if the baseline came from the nonlocal tail, run all local actions separately. Assert the normal maximum is 10 hybrid games per duplicate and the hard maximum is 14 when all four baselines are nonlocal.

- [ ] **Step 5: Run Task 6 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_player_proxy.py \
  tests/residual_bidder/test_hybrid.py
```

- [ ] **Step 6: Implement the evaluator without calling deployed exact play**

Import neither `RuleExactFirst4Player._exact_play` nor its IS-pool helpers. Use one native solver object per outer worker process. Record the full terminal state summary before solving and fail if the native solver is unavailable; never substitute a Python solver.

- [ ] **Step 7: Verify two real deals twice and commit**

Run two fixed duplicate deals through the native solver twice. Require identical auctions, first-four card sequences, targets, solver values, and branch manifests. Instrument the solver and assert one call per branch.

```bash
git add residual_bidder/player_proxy.py residual_bidder/hybrid.py \
  tests/residual_bidder/test_player_proxy.py tests/residual_bidder/test_hybrid.py
git commit -m "feat: generate four-trick bid counterfactuals"
```

## Task 7: Add Disjoint Seeds and Atomic Deal-Grouped Storage

**Files:**
- Create: `residual_bidder/seeds.py`
- Create: `residual_bidder/records.py`
- Create: `residual_bidder/shards.py`
- Create: `tests/residual_bidder/test_seeds.py`
- Create: `tests/residual_bidder/test_records.py`
- Create: `tests/residual_bidder/test_shards.py`

**Interfaces:**
- Consumes: hybrid results from Task 6.
- Produces: five partition namespaces, validated records, atomic shards, manifests, and duplicate-deal admission.

- [ ] **Step 1: Write failing partition and seed tests**

```python
class Partition(str, Enum):
    TRAIN = "train"
    FAST_DEV = "fast-hybrid-development"
    REAL_DEV = "complete-play-development"
    PROMOTION = "promotion"
    FINAL = "final-test"

@dataclass(frozen=True)
class DealNamespace:
    partition: Partition
    scope_id: str  # continuation ID, fixed dev/test manifest ID, or candidate policy ID

@dataclass(frozen=True)
class DealSeed:
    namespace: DealNamespace
    deal_index: int
    deal_id: str
    shuffle_seed: int

def deal_seed(namespace: DealNamespace, deal_index: int) -> DealSeed: ...
```

Derive `shuffle_seed` from BLAKE2b domain `residual-bidder-shuffle-v1`, partition, scope ID, and index. The policy seed is not an input. Training scope is the frozen continuation ID; fast/real development and final test use fixed manifest IDs; promotion scope is the candidate policy ID. Test one million `(partition,scope,index)` keys for unique deal IDs, fixed restart identity, and distinct shuffle seeds across scopes and partitions.

- [ ] **Step 2: Write failing branch/state schema tests**

Use schema `stochastic-hybrid-residual-data-v1` and exact immutable types:

```python
@dataclass(frozen=True)
class BranchRecord:
    deal_id: str
    partition: Partition
    room_id: str
    state_id: str
    logical_seat: int
    acting_policy_id: str
    seat_policy_ids: tuple[str, str, str, str]
    forced_action: BidAction
    continuation_auction: tuple[AuctionEvent, ...]
    terminal_team0_minus_team1: float
    acting_partnership_margin: float
    raw_target: float
    scaled_target: float
    solver_build_id: str

@dataclass(frozen=True)
class TrainingRow:
    deal_id: str
    partition: Partition
    state_id: str
    features_167: np.ndarray
    raw_targets_2: np.ndarray
    scaled_targets_2: np.ndarray
    masks_2: np.ndarray
    center: BidAction
    baseline_action: BidAction
    baseline_from_tail: bool
    stratum: str
    natural_frequency: float
    sampling_probability: float
    importance_weight: float
```

The full JSON branch record additionally stores every spec-required hash, chronological observed auction, exact 149-vector, 14 legal scores, neighborhood, first-four terminal summary, independent shuffle/policy identifiers, and sampling metadata. Reject wrong shapes/dtypes, non-finite data, `scaled != raw/100`, nonzero masked targets, duplicated branch actions, changed policy/solver hashes, children in another partition, or a nonzero center target.

- [ ] **Step 3: Run record tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_seeds.py \
  tests/residual_bidder/test_records.py
```

- [ ] **Step 4: Implement deterministic IDs and lossless serialization**

State IDs are SHA-256 over schema, deal, room, bidder, candidate ordinal, public auction, and exact 149-vector bytes. Serialize floats with JSON's round-trippable representation and arrays in NPZ; on read, reconstruct and run the same validation before admitting data.

- [ ] **Step 5: Write failing atomic-shard and duplicate-admission tests**

Each completed shard directory is:

```text
manifest.json
branches.jsonl.gz
states.npz
COMPLETE
```

`states.npz` contains numeric arrays and fixed-width Unicode IDs only, with `allow_pickle=False`. Inject failures after every write/fsync/rename boundary and assert no invalid directory can be opened as a shard. `dataset-index.sqlite3` enforces `UNIQUE(partition, deal_id)` and admits all deals from a validated shard in one transaction.

- [ ] **Step 6: Implement atomic shard creation and validation**

Write to a sibling directory named `.partial-{uuid}`, fsync each file, write a manifest containing file hashes/counts/config and policy identities, fsync the directory, rename to `shard-{partition}-{start_index:012d}-{end_index:012d}-{digest16}`, then fsync the parent. Readers require `COMPLETE`, exact hashes, exact counts, and successful record validation.

- [ ] **Step 7: Run Task 7 tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_seeds.py \
  tests/residual_bidder/test_records.py tests/residual_bidder/test_shards.py
git add residual_bidder/seeds.py residual_bidder/records.py residual_bidder/shards.py \
  tests/residual_bidder/test_seeds.py tests/residual_bidder/test_records.py \
  tests/residual_bidder/test_shards.py
git commit -m "feat: store hybrid bidder datasets atomically"
```

## Task 8: Build the Immutable League, Outcome-Blind Reservoir, and Parallel Generator

**Files:**
- Create: `residual_bidder/league.py`
- Create: `residual_bidder/reservoir.py`
- Create: `residual_bidder/cli/generate.py`
- Create: `tests/residual_bidder/test_league.py`
- Create: `tests/residual_bidder/test_reservoir.py`
- Create: `tests/residual_bidder/test_generate_cli.py`

**Interfaces:**
- Consumes: policies, hybrid evaluator, seeds, and shard writer.
- Produces: frozen opponent selection, reservoir membership/weights, and resumable CPU generation blocks.

- [ ] **Step 1: Write failing league schedule tests**

```python
@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    kind: Literal["legacy-nsfp", "residual"]
    checkpoint_path: str
    checkpoint_sha256: str
    calibration: CalibrationTuple | None
    accepted_iteration: int | None

class League:
    def weights(self) -> dict[str, float]: ...
    def select(self, deal_id: str) -> PolicyManifest: ...
```

Exact weights are NSFP `1.0` before acceptance; NSFP/latest `0.5/0.5` with one residual snapshot; and NSFP/latest/older `0.5/0.25/0.25` with the older share uniform when two or more exist. Selection is BLAKE2b-keyed by deal ID, immutable across rooms/branches/restarts, and both opponent seats use the returned member. Reject changed bytes behind a manifest ID.

- [ ] **Step 2: Write outcome-blind reservoir tests**

```python
@dataclass(frozen=True)
class StratumKey:
    center_bucket: str       # Nil,1,2,3,4,5,6,7+
    bidding_position: int    # 0..3
    nil_already_seen: bool
    partner_bid_visible: bool
    opponent_policy_id: str
    baseline_from_tail: bool

class StratifiedReservoir:
    def consider_before_outcome(self, state: CapturedBidState,
                                key: StratumKey) -> bool: ...
```

Use deterministic per-stratum Algorithm R keyed by schema, stratum, and `seen_count`. Call it after the baseline auction but before any local branch return is available. Tests mutate every branch reward and require identical membership. For stratum `h`, store inclusion probability `min(1, capacity_h/seen_h)` and natural frequency `seen_h/seen_total`.

- [ ] **Step 3: Define and test the natural/reservoir training correction**

All training rows remain in the unfiltered natural pool; selected reservoir rows are an additional sampling view, not a second label. With mixture fractions `0.75/0.25`, compute the actual row/stratum sampling probability `q_train(h)` and use:

```python
importance_weight_h = p_natural_h / q_train_h
normalized_weight = importance_weight_h / mean_importance_in_loss_batch
```

A synthetic population distorted by the reservoir mixture must recover the known conditional mean after weighting.

- [ ] **Step 4: Write generation CLI failure/resume tests**

```bash
CUDA_VISIBLE_DEVICES='' python -m residual_bidder.cli.generate \
  --config configs/residual_bidder/base.yaml \
  --run-dir output/residual-bidder/smoke \
  --iteration 0 --partition train \
  --start-deal-index 0 --num-deals 8 --outer-workers 2
```

Reject visible CUDA in a formal generator, `nested_exact != 1`, artifact drift, mutable league/incumbent manifests, overlapping deal IDs, invalid partitions, and a missing native solver. A terminated worker may leave only ignored partial directories; resume regenerates only unadmitted deals.

- [ ] **Step 5: Run Task 8 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_league.py \
  tests/residual_bidder/test_reservoir.py \
  tests/residual_bidder/test_generate_cli.py
```

- [ ] **Step 6: Implement spawn-based outer deal parallelism**

Resolve `outer_workers=0` to `max(1, available_cpu_count()-2)`, where `available_cpu_count()` uses `len(os.sched_getaffinity(0))` when supported and otherwise `(os.cpu_count() or 1)`. Use multiprocessing `spawn`; each worker loads immutable CPU policies once, constructs one native solver, owns complete duplicate deals, and writes worker-local staging shards. Never create a process pool inside a deal worker.

- [ ] **Step 7: Emit and verify operational metrics**

Every block reports unique deals, two baseline rooms, total hybrid games, games/deal, states, local/tail baseline counts, solver median/p95/p99/max, deals/hour, CPU utilization, failure counts, raw target percentiles, center histogram, reservoir composition, and projected storage per 10,000 deals.

- [ ] **Step 8: Run an 8-deal CPU smoke and commit**

Require eight admitted unique deal IDs, no more than 112 hybrid games, valid reopened hashes, no CUDA context, one solver call per game, and identical semantic shard digests on rerun after excluding timestamps.

```bash
git add residual_bidder/league.py residual_bidder/reservoir.py \
  residual_bidder/cli/generate.py tests/residual_bidder/test_league.py \
  tests/residual_bidder/test_reservoir.py tests/residual_bidder/test_generate_cli.py
git commit -m "feat: parallelize hybrid bidder generation"
```

## Task 9: Train and Resume the Five-Member Local-Advantage Ensemble

**Files:**
- Create: `residual_bidder/dataset.py`
- Create: `residual_bidder/training.py`
- Create: `residual_bidder/cli/train.py`
- Create: `tests/residual_bidder/test_dataset.py`
- Create: `tests/residual_bidder/test_training.py`

**Interfaces:**
- Consumes: validated training shards and Task 3 ensemble.
- Produces: deterministic grouped batches, weighted MSE, resumable fit state, metrics, fixed probes, and an uncalibrated candidate checkpoint.

- [ ] **Step 1: Write deal grouping and Poisson bootstrap tests**

```python
def fit_split(deal_id: str) -> Literal["fit", "fit-validation"]: ...

def bootstrap_multiplicity(member_index: int, deal_id: str) -> int:
    """Deterministic Poisson(1) draw from a member/deal BLAKE2b seed."""
```

Every row from both rooms and all states of one deal must share the split and one bootstrap count per member. Assert restart identity, differing member/deal keys, and approximate Poisson(1) frequencies over 100,000 synthetic deals.

- [ ] **Step 2: Write exact scaled-MSE and gradient tests**

```python
def masked_weighted_mse(
    predictions: torch.Tensor,      # (5,B,2), scaled units
    raw_targets: torch.Tensor,      # (B,2), raw score points
    masks: torch.Tensor,            # (B,2)
    importance: torch.Tensor,       # (B,)
    bootstrap: torch.Tensor,        # (5,B)
) -> tuple[torch.Tensor, torch.Tensor]:  # scalar total, per-member losses
    scaled = raw_targets / 100.0
    weights = masks.unsqueeze(0) * importance[None, :, None] * bootstrap[:, :, None]
    squared = (predictions - scaled.unsqueeze(0)).square()
    per_member = (weights * squared).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)
    return per_member.mean(), per_member
```

Test hand-computed arithmetic, center-relative negative/positive extremes, no clipping, zero illegal-slot gradients, one member receiving zero bootstrap mass, and synthetic conditional-mean recovery. Assert no `SmoothL1Loss`, Huber loss, target clamp, or return normalization exists.

- [ ] **Step 3: Write deterministic dataset-mixture tests**

The loader draws 75% natural rows and 25% reservoir-view rows, groups sampling decisions by deal, applies the recorded natural correction, and returns `(features, raw_targets, masks, importance, deal_ids)`. A restart from saved RNG state must yield the identical next batch. No promotion/final partition may be opened by the training dataset.

- [ ] **Step 4: Write checkpoint-resume identity tests**

Training state contains epoch, optimizer/scheduler states, scaler if present, CPU/CUDA/NumPy RNG states, admitted shard hashes, config/incumbent/league hashes, all five model states, best fit-validation metric, and fixed-probe IDs. Interrupted-plus-resumed training must produce bit-identical next batches and parameters to uninterrupted training on CPU deterministic mode.

- [ ] **Step 5: Run Task 9 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_dataset.py \
  tests/residual_bidder/test_training.py
```

- [ ] **Step 6: Implement the fit loop and diagnostics**

Use AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size `4096`, gradient norm clip `1.0`, float32, at most 200 epochs, and patience 15 on weighted fit-validation MSE. Log raw-score calibration by predicted bucket, lower/upper sign accuracy, offline local regret, ensemble disagreement, per-stratum errors, Q stability on fixed probes, gradient norms, and effective bootstrap/sample weights.

- [ ] **Step 7: Implement the training CLI and atomic candidate artifact**

```bash
python -m residual_bidder.cli.train \
  --config configs/residual_bidder/base.yaml \
  --run-dir output/residual-bidder/smoke \
  --iteration 0 --device cpu --resume
```

The result is `status="candidate"` with no calibration tuple. The checkpoint records exact dataset, incumbent, league, config, git, torch, and initialization identities. Changed inputs make `--resume` fail rather than restart silently.

- [ ] **Step 8: Run CPU overfit/GPU smoke tests and commit**

Overfit 256 deterministic rows to near-zero MSE on CPU. On the later AutoDL preflight, run at least two GPU optimizer steps over all five members, reload the candidate, and require identical evaluation predictions with no `sm_120` warning.

```bash
git add residual_bidder/dataset.py residual_bidder/training.py \
  residual_bidder/cli/train.py tests/residual_bidder/test_dataset.py \
  tests/residual_bidder/test_training.py
git commit -m "feat: train local bid advantage ensemble"
```

## Task 10: Calibrate on Fast Hybrid Deals, Then Select on Frozen Real Play

**Files:**
- Create: `residual_bidder/evaluation.py`
- Create: `residual_bidder/calibration.py`
- Create: `residual_bidder/cli/calibrate.py`
- Create: `tests/residual_bidder/test_evaluation.py`
- Create: `tests/residual_bidder/test_calibration.py`

**Interfaces:**
- Consumes: candidate ensemble, policies, league, hybrid evaluator, real deployed player, and per-seat likelihoods.
- Produces: common-random duplicate results, a frozen shortlist, one selected calibration tuple, and a calibrated candidate checkpoint.

- [ ] **Step 1: Write duplicate evaluation identity tests**

```python
@dataclass(frozen=True)
class DuplicateResult:
    deal_id: str
    opponent_policy_id: str
    room_margins: tuple[float, float]
    duplicate_margin: float
    policy_ids_by_room: tuple[tuple[str, str, str, str],
                              tuple[str, str, str, str]]
    policy_tape_sha256: str

def duplicate_margin(room0_candidate_margin: float,
                     room1_candidate_margin: float) -> float:
    return 0.5 * (room0_candidate_margin + room1_candidate_margin)
```

Swapping candidate ownership between `(0,2)` and `(1,3)` on one deal must cancel a constructed hand-strength offset. Candidate, incumbent, and every calibration tuple consume the same deal IDs, opponent selections, room IDs, and policy uniforms.

- [ ] **Step 2: Write the complete deployed-play evaluator tests**

`DeployedPlayEvaluator` wraps fresh `RuleExactFirst4NilPlayer` instances with `ActingBidPlayerProxy`, runs all 13 tricks through `SpadesMatchRunner`, and injects a `SeatLikelihoods` map matching the actual policy assigned to every seat. Assert:

1. first four card decisions still route through existing Nil/non-Nil rules;
2. later decisions still route through `_exact_play` and its current control flow;
3. residual seats use their exact adapter, legacy seats use the preserved softened adapter, and no formal seat uses an undeclared fallback;
4. complete games never emit training rows;
5. candidate and opponent partnership seats share their respective policy objects.

- [ ] **Step 3: Write fast-grid canonicalization tests**

```python
@dataclass(frozen=True, order=True)
class GridPoint:
    uncertainty_lambda: float
    temperature: float
    epsilon: float
    rho: float

def canonical_grid(config: PolicyGridConfig) -> tuple[GridPoint, ...]: ...
```

When `epsilon==0`, canonicalize `rho=1` because rho has no effect. Include every configured lambda/T/epsilon/rho combination otherwise, including a `rho=1` uniform-tail control for every nonzero epsilon. Require at least one fully deterministic point `(T=0, epsilon=0)`.

- [ ] **Step 4: Write successive-halving and deterministic-control tests**

At round sizes `[256,1024,4096]` and eta 4, rank by paired mean fast-hybrid duplicate margin, retain `ceil(n/4)`, and reuse earlier deal results rather than replaying them. Each grid point/deal plays only one sampled policy game per room through four tricks plus one DDS solve; it does not expand local counterfactual branches or emit labels. Regardless of rank, carry the best deterministic point into the final real-play shortlist so stochasticity is tested rather than assumed. Freeze the final eight-or-fewer point IDs before reading any real-play result.

- [ ] **Step 5: Write the real-play selection and tie-rule tests**

Evaluate the frozen shortlist and incumbent on one fixed `REAL_DEV` manifest. Select maximum paired mean margin relative to the shared incumbent baseline. Exact numeric ties use this order:

```python
tie_key = (
    point.epsilon,                 # lower first
    point.temperature,             # lower first
    -point.uncertainty_lambda,     # larger first
    point.rho,                     # smaller first
)
```

Real-play results may select parameters and diagnostics only; attempting to serialize them as Q labels must raise.

- [ ] **Step 6: Run Task 10 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_evaluation.py \
  tests/residual_bidder/test_calibration.py
```

- [ ] **Step 7: Implement evaluation and calibration CLIs**

```bash
CUDA_VISIBLE_DEVICES='' python -m residual_bidder.cli.calibrate \
  --config configs/residual_bidder/base.yaml \
  --run-dir output/residual-bidder/smoke --iteration 0 \
  --fast-dev-manifest output/residual-bidder/smoke/manifests/fast-dev.json \
  --real-dev-manifest output/residual-bidder/smoke/manifests/real-dev.json \
  --outer-workers 2 --resume
```

Persist every round result, frozen shortlist, common random-tape manifest, and selected tuple atomically. A changed candidate, incumbent, opponent league, deal list, play hash, or seed invalidates resume.

The uncalibrated ensemble retains its immutable `model_id`. Saving the selected tuple creates a new content-addressed `policy_id = SHA256(model_id, NSFP hash, lambda, T, epsilon, rho)`; changing any calibration value cannot reuse an existing policy ID.

- [ ] **Step 8: Verify no-label real play and commit**

Run fake fast/real evaluators plus two real duplicate deals. Require that real-play output contains scores, bids, likelihood ESS, Nil/contract/bag diagnostics, and policy identities but no `TrainingRow`, raw target, or scaled target field.

```bash
git add residual_bidder/evaluation.py residual_bidder/calibration.py \
  residual_bidder/cli/calibrate.py tests/residual_bidder/test_evaluation.py \
  tests/residual_bidder/test_calibration.py
git commit -m "feat: calibrate stochastic bidder by duplicate play"
```

## Task 11: Promote Only Statistically Superior Candidates and Advance the League

**Files:**
- Create: `residual_bidder/promotion.py`
- Create: `residual_bidder/cli/evaluate.py`
- Modify: `residual_bidder/league.py`
- Create: `tests/residual_bidder/test_promotion.py`
- Modify: `tests/residual_bidder/test_league.py`

**Interfaces:**
- Consumes: calibrated candidate, incumbent, frozen league, and complete deployed-play evaluator.
- Produces: a predeclared promotion manifest, deal-level confidence intervals, a pass/fail decision, immutable league acceptance, and one-time final evaluation.

- [ ] **Step 1: Write fixed-sample-size tests**

From development paired differences with standard deviation `s`, compute:

```python
raw_n = math.ceil(((NormalDist().inv_cdf(0.95)
                    + NormalDist().inv_cdf(0.80)) * s / 1.0) ** 2)
n = max(4096, round_up(raw_n, multiple=256))
```

Write `n`, promotion deal IDs, opponent allocations, strata, candidate/incumbent/checkpoint hashes, calibration tuple, all thresholds, and bootstrap seed to `promotion-manifest.json` before any promotion game. There is no result-dependent extension or early stopping.

- [ ] **Step 2: Write paired-statistic and bootstrap tests**

For every deal/opponent:

```python
z = candidate_duplicate_margin - incumbent_duplicate_margin
```

Use 10,000 deterministic resamples of whole deal bundles within opponent strata; all four underlying candidate/incumbent room games stay together. Combine stratum means with the frozen league weights. Reject code paths that resample rooms, states, branches, or individual games.

- [ ] **Step 3: Encode all three promotion gates in tests**

Promotion passes iff:

1. the one-sided 95% lower bound of league-weighted mean `Z` is strictly greater than zero;
2. the one-sided 95% upper bound in the NSFP-opponent stratum is at least zero;
3. the one-sided 95% upper bound in every predeclared opponent/behavior stratum with at least 5% promotion weight or natural frequency is at least zero.

Equal policies must fail gate 1. A known superior synthetic policy must pass with adequate sample size. A candidate with positive overall gain but established protected-stratum regression must fail. Nil success, contract success, overtricks, and bags remain diagnostics and cannot veto promotion.

Behavior strata are marginal, overlapping deal groups computed before candidate outcomes from the frozen incumbent's two-room baseline: Nil absent/present, no tail/at least one tail action, and the first candidate observation's NSFP center bucket (`Nil,1,2,3,4,5,6,7+`). Combine these with each opponent member; apply a gate only when the group reaches the configured 5% promotion weight or estimated natural frequency. Always resample the parent deal once even when it belongs to multiple reported groups.

- [ ] **Step 4: Write immutable league-acceptance tests**

Only a passing, fully completed promotion result may change the league. Copy the promoted checkpoint into a content-addressed immutable directory, verify its hash, atomically append a `PolicyManifest`, and recompute the exact league weights. Failed or interrupted candidates leave incumbent and league byte-identical.

- [ ] **Step 5: Run Task 11 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_promotion.py \
  tests/residual_bidder/test_league.py
```

- [ ] **Step 6: Implement promotion and final-test modes**

```bash
CUDA_VISIBLE_DEVICES='' python -m residual_bidder.cli.evaluate \
  --config configs/residual_bidder/base.yaml \
  --run-dir output/residual-bidder/main --iteration 0 \
  --mode promotion --outer-workers 0 --resume
```

`--mode final` requires an explicit final-test manifest, a promoted checkpoint, and a run state declaring training complete. It reads the one-time `FINAL` namespace once and cannot update calibration, checkpoint, league, or stopping state.

- [ ] **Step 7: Verify the promotion decision artifact and commit**

```bash
git add residual_bidder/promotion.py residual_bidder/cli/evaluate.py \
  residual_bidder/league.py tests/residual_bidder/test_promotion.py \
  tests/residual_bidder/test_league.py
git commit -m "feat: promote bidders with paired team matches"
```

## Task 12: Orchestrate Frozen Iterations and the Approved Data-Stability Stop

**Files:**
- Create: `residual_bidder/iteration.py`
- Create: `residual_bidder/cli/iterate.py`
- Create: `tests/residual_bidder/test_iteration.py`

**Interfaces:**
- Consumes: all generation, training, calibration, promotion, and league commands.
- Produces: a recoverable frozen-continuation state machine and auditable block/iteration stopping decisions.

- [ ] **Step 1: Write state-transition and identity tests**

```python
class IterationPhase(str, Enum):
    CREATED = "created"
    GENERATING = "generating"
    TRAINING = "training"
    CALIBRATING_FAST = "calibrating-fast"
    EVALUATING_REAL_DEV = "evaluating-real-dev"
    PROMOTION_READY = "promotion-ready"
    PROMOTING = "promoting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PAUSED = "paused"
```

Every transition atomically records command, start/end time, input/output hashes, continuation/incumbent/league IDs, deal ranges, and reason. Restart may resume only the exact incomplete transition. Reject incumbent mutation during generation, old-continuation labels, overlapping deal blocks, promotion seed reuse, or final-test access before completion.

- [ ] **Step 2: Write the three-part block-stopping tests**

After at least three independent admitted blocks, compare each block-trained/calibrated candidate with its immediate predecessor on the same fixed probe and development manifests. Stop growing the current iteration only when all hold:

```python
all(delta_fast_margin[i] <= deal_bootstrap_se[i] for i in latest_three)
changed_probe_fraction_l1_gt_0_01 < 0.005
latest_real_dev_paired_margin <= predecessor_real_dev_paired_margin
```

Compute a probe change from the full 14-vector L1 distance, not argmax alone. A promotion failure without all three stability conditions does not declare convergence.

- [ ] **Step 3: Write accepted-policy iteration tests**

An accepted candidate becomes the next incumbent, freezes a new continuation ID, and starts a fresh training namespace range. Labels from the preceding continuation remain auditable but are excluded unless their local branches are rerun under the new continuation. A rejected stable candidate ends that iteration without changing the league; an unstable candidate requests another unique block.

- [ ] **Step 4: Run Task 12 tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_iteration.py
```

- [ ] **Step 5: Implement subprocess isolation and clean interruption**

The iterator launches CPU generation/evaluation children with `CUDA_VISIBLE_DEVICES=""` and GPU training children with the configured CUDA device. It waits for validated manifests rather than trusting exit code alone. On SIGINT/SIGTERM, let atomic writers finish or discard their partial directory, persist `PAUSED`, terminate children cleanly, and exit nonzero so the remote launcher can report interruption.

- [ ] **Step 6: Implement the resumable CLI and commit**

```bash
python -m residual_bidder.cli.iterate \
  --config configs/residual_bidder/base.yaml \
  --run-dir output/residual-bidder/main --resume
```

```bash
git add residual_bidder/iteration.py residual_bidder/cli/iterate.py \
  tests/residual_bidder/test_iteration.py
git commit -m "feat: orchestrate stochastic bidder iterations"
```

## Task 13: Integrate an Optional Promoted Bidder and Runtime Provenance

**Files:**
- Modify: `gui/backend.py:123-240,376-520,592-615`
- Modify: `gui/game_server.py:242-450,947-1019`
- Modify: `gui/src/game.js:193-218,313-318,403-440`
- Modify: `tests/test_gui_backend.py`
- Modify: `tests/test_game_server_showdown.py`
- Modify: `gui/src/game.test.js`
- Create: `tests/residual_bidder/test_runtime_integration.py`

**Interfaces:**
- Consumes: promoted policy loader, sampling tape, proxy, and per-seat likelihood adapters.
- Produces: optional stochastic acting bids in both GUI paths, deterministic per-deal replay, and exact observed-bid provenance for later belief sampling.

- [ ] **Step 1: Freeze the default-path characterization**

With no `--residual-bid-checkpoint`, fixed payloads must produce exactly the current NSFP bid, card choice, `last_bid_info`, `last_play_info`, API response shape, and game-server behavior. Record these expectations before adding any new argument.

- [ ] **Step 2: Add deal/tape/provenance payload tests**

`buildAiPayload(state)` must include:

```javascript
{
  dealId: String(state.seed),
  roomId: 'single-player',
  dealerSeat: (state.firstSeat + 3) % 4,
  policyProvenance: [...state.bidPolicyIds],
}
```

An AI bid response includes `policyId` and `fallbackReason`; `applyBid` stores `policyId` by seat for subsequent play requests. Human bids store `external-human`. Remote `GameRoom` uses room code plus seed as deal identity and tracks the two human and two AI seat policies internally.

`build_local_state` must set `dealer_seat` and rebuild chronological non-pass `state.bids` in opener order from the seat-indexed bid array; it must continue populating `state.max_bid`. This trace is required by residual likelihood reconstruction during later exact play. Reject a payload whose non-null bids are inconsistent with the declared dealer/current bidder.

- [ ] **Step 3: Write promoted acting-only integration tests**

Add CLI arguments:

```text
--residual-bid-checkpoint PATH
--policy-seed INTEGER
```

Reject candidate/unpromoted or incompatible checkpoints. With a synthetic promoted checkpoint, assert all configured AI seats share one policy object/checkpoint, the sampled action matches the declared 14-vector and supplied tape, the same deal/room/seat reproduces the same action after restart, and changing only deal ID can change the sampled action.

- [ ] **Step 4: Write per-seat runtime belief tests**

For single-player mode, the human seat uses `ExternalUniformLikelihoodAdapter` and AI seats use residual or recorded fallback adapters. For remote mode, the two human seats are external and the AI partnership shares residual. An interactive residual failure returns deterministic NSFP argmax, records `legacy-nsfp-fallback` for that seat, and causes later exact play to use the legacy adapter for its observed bid. Formal evaluation remains strict and never follows this branch.

- [ ] **Step 5: Run runtime tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/test_gui_backend.py \
  tests/test_game_server_showdown.py \
  tests/residual_bidder/test_runtime_integration.py
(cd gui && npm test)
```

- [ ] **Step 6: Implement one shared promoted-policy loader**

Do not duplicate model definitions in `gui`. `RuleExactProvider` and `GameRoom` receive a shared `StochasticResidualPolicy`; only bid calls are proxied. Card calls still enter the same `RuleExactFirst4NilPlayer`, with the intentional per-seat likelihood injection from Task 5. Health/startup diagnostics expose policy ID, checkpoint hash, calibration tuple, policy seed hash, and cumulative fallback counts without exposing the seed itself if configured as a secret.

- [ ] **Step 7: Verify default compatibility, stochastic reproducibility, and commit**

```bash
git add gui/backend.py gui/game_server.py gui/src/game.js \
  tests/test_gui_backend.py tests/test_game_server_showdown.py \
  gui/src/game.test.js tests/residual_bidder/test_runtime_integration.py
git commit -m "feat: serve promoted stochastic bidder"
```

## Task 14: Package a Secret-Safe, Resumable AutoDL Run

**Files:**
- Create: `scripts/autodl/bootstrap_bidder.sh`
- Create: `scripts/autodl/run_bidder.sh`
- Create: `tests/residual_bidder/test_autodl_scripts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: committed repository snapshot and all bidder CLIs.
- Produces: idempotent capability preflight, content-addressed remote release, locked resumable launcher, and operator commands without embedded credentials.

- [ ] **Step 1: Write script-safety and syntax tests**

```python
@pytest.mark.parametrize("path", [
    Path("scripts/autodl/bootstrap_bidder.sh"),
    Path("scripts/autodl/run_bidder.sh"),
])
def test_remote_script_has_no_secret_or_provider_assumptions(path):
    text = path.read_text()
    assert "password=" not in text.lower()
    assert "sshpass" not in text
    assert "AUTODL_PASSWORD" not in text
    subprocess.run(["bash", "-n", str(path)], check=True)
```

Also require `set -euo pipefail`, quoted path variables, an explicit run directory argument, no destructive recursive deletion, and a `flock` lock around the iterator.

- [ ] **Step 2: Implement an idempotent bootstrap script**

Exact interface:

```bash
: "${AUTODL_RELEASE_DIR:?set the inspected release directory}"
: "${AUTODL_VENV_DIR:?set the isolated venv directory}"
: "${AUTODL_RUN_DIR:?set the persistent run directory}"
bash scripts/autodl/bootstrap_bidder.sh \
  --release-dir "$AUTODL_RELEASE_DIR" \
  --venv-dir "$AUTODL_VENV_DIR" \
  --run-dir "$AUTODL_RUN_DIR"
```

The script:

1. records `uname`, CPU affinity/count, RAM, filesystems, Python, driver, `nvidia-smi`, torch, CUDA, and GPU details;
2. creates a venv with `--system-site-packages` so a verified AutoDL image torch can be reused;
3. installs only `requirements-bidder.txt` into the venv;
4. imports/compiles the native solver and records its digest;
5. runs `residual_bidder.cli.preflight --require-cuda --require-sm120 --require-native-solver`;
6. writes environment JSON and `pip freeze` beneath the run directory atomically;
7. stops with an actionable report if the image torch lacks working `sm_120`, rather than automatically replacing it with an unverified wheel.

- [ ] **Step 3: Implement the locked launcher and signal behavior**

Exact interface:

```bash
: "${AUTODL_RELEASE_DIR:?set the inspected release directory}"
: "${AUTODL_VENV_DIR:?set the isolated venv directory}"
: "${AUTODL_RUN_DIR:?set the smoke run directory}"
bash scripts/autodl/run_bidder.sh \
  --release-dir "$AUTODL_RELEASE_DIR" \
  --venv-dir "$AUTODL_VENV_DIR" \
  --run-dir "$AUTODL_RUN_DIR" \
  --mode smoke
```

`--mode smoke` runs preflight plus the Task 15 smoke sequence. `--mode production` runs `python -m residual_bidder.cli.iterate --resume`. The script obtains `$run_dir/run.lock` with `flock -n`, writes PID/start/commit/config hashes, forwards TERM/INT to the Python child, waits for atomic state persistence, and appends stdout/stderr to a timestamped log. A second launcher must fail without disturbing the first.

- [ ] **Step 4: Document committed-snapshot transfer over interactive SSH**

Do not copy the dirty working tree. From the local repository, create and transfer only `git archive HEAD`; let normal `ssh`/`scp` prompt interactively or use an SSH key. Never put the password in a shell variable or command argument.

```bash
set -u
: "${AUTODL_HOST:?set AUTODL_HOST}"
: "${AUTODL_PORT:?set AUTODL_PORT}"
: "${AUTODL_USER:?set AUTODL_USER}"
: "${AUTODL_WORKSPACE:?set AUTODL_WORKSPACE after remote disk inspection}"
release_commit="$(git rev-parse HEAD)"
archive_path="$(mktemp -t residual-bidder-release.XXXXXX.tar.gz)"
git archive --format=tar.gz -o "$archive_path" HEAD
ssh -p "$AUTODL_PORT" "$AUTODL_USER@$AUTODL_HOST" \
  "mkdir -p '$AUTODL_WORKSPACE/incoming' \
   '$AUTODL_WORKSPACE/releases/${release_commit}'"
scp -P "$AUTODL_PORT" "$archive_path" \
  "$AUTODL_USER@$AUTODL_HOST:$AUTODL_WORKSPACE/incoming/${release_commit}.tar.gz"
ssh -p "$AUTODL_PORT" "$AUTODL_USER@$AUTODL_HOST" \
  "tar -xzf '$AUTODL_WORKSPACE/incoming/${release_commit}.tar.gz' \
   -C '$AUTODL_WORKSPACE/releases/${release_commit}' && \
   printf '%s\n' '$release_commit' > \
   '$AUTODL_WORKSPACE/releases/${release_commit}/RELEASE_COMMIT'"
rm -f "$archive_path"
```

The four `AUTODL_*` connection/path values are supplied only in the operator's current shell; the password is entered at the SSH prompt and is not an environment variable. Verify the remote release by comparing `release_commit`, config SHA-256, NSFP SHA-256, and an archive file manifest.

- [ ] **Step 5: Document safe background start and observability**

After bootstrap succeeds:

```bash
release_dir="$AUTODL_WORKSPACE/releases/$release_commit"
venv_dir="$AUTODL_WORKSPACE/venvs/residual-bidder"
smoke_dir="$AUTODL_WORKSPACE/runs/smoke"
mkdir -p "$smoke_dir"
nohup bash "$release_dir/scripts/autodl/run_bidder.sh" \
  --release-dir "$release_dir" --venv-dir "$venv_dir" \
  --run-dir "$smoke_dir" --mode smoke \
  > "$smoke_dir/launcher.log" 2>&1 < /dev/null &
```

The runbook includes `tail -F` for logs, `nvidia-smi dmon`, `ps`/CPU-affinity inspection, run-state JSON, shard validation, `df -h`, and clean TERM via the recorded PID. Explain that CPU/native solver utilization dominates generation while the 5090 is expected to be busy mainly during fitting.

- [ ] **Step 6: Run script tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_autodl_scripts.py
git add scripts/autodl/bootstrap_bidder.sh scripts/autodl/run_bidder.sh \
  tests/residual_bidder/test_autodl_scripts.py README.md
git commit -m "ops: run bidder training on autodl"
```

## Task 15: End-to-End Verification and the Production Launch Gate

**Files:**
- Review: every file created or modified in Tasks 1-14.
- Do not modify or stage unrelated dirty-worktree files.

**Interfaces:**
- Consumes: complete implementation and an available AutoDL connection.
- Produces: local regression evidence, remote smoke artifacts, measured capacity, and a user decision point before production generation.

- [ ] **Step 1: Run the complete local Python and GUI regression suites**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests
(cd gui && npm test)
```

Expected: zero failed tests. Report exact passed/skipped counts; do not summarize an interrupted or partial run as success.

- [ ] **Step 2: Run frozen-artifact and source-boundary checks**

```bash
git diff --check
shasum -a 256 Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt configs/8.yaml
git status --short
rg -n "bid_dds\.pt|_exact_play|_build_is_pool" residual_bidder configs/residual_bidder
```

The hashes must match the Global Constraints. `bid_dds.pt` must not appear in new runtime/training configuration. `_exact_play`/IS references are permitted only in the complete deployed-play evaluator and likelihood integration tests, never in `hybrid.py` or generation workers.

- [ ] **Step 3: Transfer the committed snapshot and run AutoDL capability preflight**

Using the interactive SSH procedure from Task 14, record remote release hash, CPU/RAM/disk, driver, GPU, torch/CUDA architecture list, native solver build ID, forward/backward checksums, and available workspace capacity. Stop if `sm_120`, native solver, artifact hashes, or disk checks fail.

- [ ] **Step 4: Run the eight-deal hybrid and two-step GPU smoke**

Under a run directory permanently marked `non_promotable_smoke=true`:

1. generate eight training duplicate deals with CUDA hidden;
2. reopen and validate every shard/hash;
3. fit all five members for at least two GPU optimizer steps;
4. reload the candidate and require identical evaluation predictions;
5. run a tiny fast calibration containing the deterministic control;
6. run two complete deployed-play duplicate deals for candidate and incumbent twice.

- [ ] **Step 5: Assert every smoke invariant programmatically**

Require:

- exactly four candidate observations per duplicate baseline;
- no duplicate uses more than 14 hybrid games;
- every hybrid game ends after four complete tricks with 36 cards and one native solve;
- no hybrid branch enters deployed belief play;
- every center target is zero and every alternative equals forced margin minus center margin, with scaled value exactly raw/100;
- five member parameter sets are independent and predictions are finite;
- every 14-vector is normalized, tail decay is monotone, and acting/belief residual vectors are bit-identical;
- the same policy tape reproduces auctions after restart;
- both partnership seats share one policy, both opponent seats share one selected league member, and all children retain the parent partition;
- complete real-play evaluation creates no training row and has no formal fallback.

- [ ] **Step 6: Report measured capacity rather than estimating from the GPU model**

Report end-to-end deals/hour, hybrid games/deal, solver median/p95/p99/max, CPU utilization, GPU fit examples/second and memory, real-play seconds/game, bytes/deal, projected storage for 10k/100k/1m deals, and projected wall time for the first 256-deal block and 4,096-deal promotion floor.

- [ ] **Step 7: Review high-risk correctness boundaries**

Review specifically for hidden-hand leakage, 16-to-14 alias errors, center versus baseline confusion, team-1 sign mistakes, duplicate score double counting, policy/shuffle seed coupling, checkpoint-dependent policy uniforms, branch random-tape drift, target clipping, member parameter sharing, outcome-conditioned reservoir selection, nested multiprocessing, acting/belief mismatch, runtime fallback provenance loss, room-level bootstrap, calibration leakage, and final-test reuse. Fix every validated finding with its own failing regression test and focused commit.

- [ ] **Step 8: Stop and request the production decision**

Present the smoke invariant/throughput report and the proposed first production block command. Do not start production mode until the user explicitly approves it. The approval authorizes the first block only; later automatic continuation follows the already approved iteration state machine unless a capability failure, identity change, or unexpected cost requires another decision.
