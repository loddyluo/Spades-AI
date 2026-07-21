# Residual Bidder Training Implementation Plan

> Superseded before implementation by
> [Stochastic Hybrid Residual Bidder Training Design](../specs/2026-07-21-stochastic-hybrid-residual-bidder-design.md).
> Do not execute this plan. A replacement implementation plan will be written
> after the new specification is reviewed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全冻结当前出牌流程与残局 belief bidder 的前提下，实现并训练一个由 `bid_nsfp.pt` 定位、最多修正 ±1、以 duplicate/team-match 分差为目标的五成员 residual-Q acting bidder。

**Architecture:** 新代码放入独立的 `residual_bidder` 包。冻结 NSFP 负责生成 149 维输入、14 动作合法分数与中心动作；五个独立 MLP 只预测 lower/upper 相对中心的期望分差。acting-bid proxy 只覆盖 `place_bid`，其余方法原样转发到当前部署使用的 `RuleExactFirst4NilPlayer`，从结构上保证出牌与内部 NSFP belief 路径不变。数据生成用可重建的整局分支而不是复制可变玩家状态；每副 duplicate 在两个 room 中复用基线结果，只补跑缺失的局部动作，最多 10 局。训练、league、promotion 和云端运行均由带哈希与原子状态文件的可恢复 CLI 驱动。

**Tech Stack:** Python 3.10+、PyTorch 2.12/CUDA 13（RTX 5090）、NumPy、PyYAML、pytest、现有 `SpadesMatchRunner` / `RuleExactFirst4NilPlayer` / native C++ exact solver、Kubernetes Job、JSON + gzip JSONL + compressed NPZ。

## Global Constraints

- `bid_nsfp.pt` 既是 acting center，也是残局 importance-sampling belief bidder；后者的加载代码、checkpoint 和调用语义不可替换。
- acting bidder 只可输出 `Nil, bid_1, ..., bid_13`，且相对 NSFP center 最多移动一格。不得增加全动作搜索或 DDS audit。
- residual 输入的前 149 维必须逐元素等于当前 `to_go_state -> BidEncoder.encode` 路径送给 NSFP 的 tensor；不得“修正”旧 encoder 的 position、auction 或 derived-feature 语义。
- 候选队两席共享同一 acting policy/checkpoint；一个对手队的两席共享同一个不可变 league member。
- 每个训练 iteration 的 incumbent 和 opponent league 在标签生成期间冻结。旧 continuation 的标签不能无标记混入当前 iteration。
- 同一 deal 的两 room、所有分支、所有状态只能属于同一 partition；promotion/final seeds 永不进入训练或调参。
- 每个 action/deal 只做一次正式 rollout。重复同一分支仅允许在 determinism test/debug 中出现，不能写入训练集。
- 正式 rollout 使用外层多进程、`RuleExactFirst4NilPlayer(num_workers=1)`，禁止形成“deal worker × exact-solver worker”的嵌套进程爆炸。
- rollout 子进程设置 `CUDA_VISIBLE_DEVICES=""`，全部 play/belief/acting inference 在 CPU；RTX 5090 只用于 residual ensemble 拟合与 GPU smoke test。
- 训练与评估遇到 checkpoint/schema/NaN/Inf 不匹配必须失败；实际对局 inference 遇到同类问题记录原因并退回 NSFP center。
- 保留当前 dirty worktree 中用户的修改。每次 commit 只 stage 本 task 列出的文件，不使用 `git add .`。
- 本计划阶段只允许只读 Kubernetes 检查。正式执行中，任何 scale Deployment、创建 Job、写 PVC 或启动长训练前都要单独确认运行窗口。

## Verified Environment Baseline

- Kubernetes context: `cs-luohaoren-mwku@yw-k8s`
- Namespace: `cs-luohaoren-mwku`
- Current Deployment/Pod: `my-task-dev` / `my-task-dev-76f5f8b6fc-lm26d`
- Node selector: `gpu-type=5090`; GPU: RTX 5090, 32,607 MiB; driver: `595.71.05`
- Pod resources: 8 CPU, 32 GiB RAM, one `nvidia.com/gpu`
- Persistent storage: `/root` on `pvc-workspace-home` (200 GiB RWX); `/scratch` on a 100 GiB RWO PVC
- The namespace permits `create jobs.batch` and `patch deployment`; the running Deployment currently holds the only GPU allocation.
- Current pod Python is 3.10.12. Its preinstalled `torch 2.5.0a0+nv24.10` reports CUDA available but lacks `sm_120`, so it is not an accepted training runtime.
- Runtime choice: the official [PyTorch 2.12 release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/) specify CUDA 13.0 as the default PyPI wheel, recommend CUDA 13.0+ for Blackwell, and require Linux driver 580.65.06 or newer; the observed driver 595.71.05 clears that gate, subject to the mandatory real-kernel preflight below.
- Frozen artifact hashes at plan time:
  - `bid_nsfp.pt`: `b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c`
  - `configs/8.yaml`: `658d14e51a42bd340fd2bed039b584ff8dec20f92f00821e818710e8131e9445`
- The fixed play identity is the SHA-256 of an ordered manifest containing `RuleExactFirst4NilPlayer`, its parent/non-Nil/Nil rule players, exact-solver wrapper/native source, Spades rules, and `configs/8.yaml`; resolve and freeze this manifest when a run is created.

---

## File and Module Map

| Path | Responsibility |
|---|---|
| `requirements-bidder.txt` | Non-Torch training/runtime dependencies; torch is installed separately so the CUDA wheel source is explicit. |
| `configs/residual_bidder/base.yaml` | Frozen paths/hashes, architecture, rollout parallelism, optimization, lambda grid, promotion statistics, and storage defaults. |
| `residual_bidder/actions.py` | 14-action representation, 16→14 score aliases, neighborhoods, local slots, stable tie order. |
| `residual_bidder/nsfp.py` | Frozen checkpoint loader and exact current encoder/bridge observation path. |
| `residual_bidder/model.py` | Residual block, one ensemble member, five-member ensemble. |
| `residual_bidder/checkpoint.py` | Composite checkpoint schema, hashes, atomic save/load, compatibility validation. |
| `residual_bidder/policy.py` | NSFP-only and composite acting policies, conservative decision, runtime fallback. |
| `residual_bidder/player_proxy.py` | `place_bid` override with all play callbacks forwarded unchanged. |
| `residual_bidder/play_factory.py` | Frozen play-player construction, shared immutable acting policies, explicit `num_workers=1`. |
| `residual_bidder/seeds.py` | Partitioned deal IDs and deterministic corresponding seed bundles. |
| `residual_bidder/counterfactual.py` | Baseline capture, forced-action replay, duplicate rooms, target assembly, ≤10-game invariant. |
| `residual_bidder/records.py` | Versioned branch/state records, JSON serialization, validation. |
| `residual_bidder/shards.py` | Atomic gzip-JSONL/NPZ shards, manifests, SHA-256, duplicate-deal index. |
| `residual_bidder/reservoir.py` | Outcome-blind stratified reservoir and natural-distribution weights. |
| `residual_bidder/dataset.py` | Deal-grouped splits, mixture sampling, masks, deterministic bootstrap multiplicities. |
| `residual_bidder/training.py` | MSE fitting, resume, metrics, candidate artifact creation. |
| `residual_bidder/league.py` | Immutable member manifest and exact league sampling weights. |
| `residual_bidder/promotion.py` | Duplicate evaluation, opponent-stratified deal bootstrap, promotion decision. |
| `residual_bidder/iteration.py` | Frozen-iteration state machine and convergence/stop rules. |
| `residual_bidder/cli/*.py` | `preflight`, `generate`, `train`, `evaluate`, and `iterate` entry points. |
| `gui/backend.py`, `gui/game_server.py` | Optional promoted-checkpoint acting-bid integration; default NSFP and all play dispatch stay compatible. |
| `deploy/k8s/residual-bidder-job.yaml` | Resumable cloud job using the known namespace, image, GPU, and RWX PVC. |
| `tests/residual_bidder/*.py` | Unit, determinism, integration, data, statistical, and CLI tests. |

## Task 1: Establish a 5090-Compatible, Reproducible Runtime Gate

**Files:**
- Create: `requirements-bidder.txt`
- Create: `configs/residual_bidder/base.yaml`
- Create: `residual_bidder/__init__.py`
- Create: `residual_bidder/cli/__init__.py`
- Create: `residual_bidder/cli/preflight.py`
- Create: `tests/residual_bidder/test_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

Test a pure `inspect_runtime(torch_module, require_cuda, require_solver) -> dict[str, object]` function with fakes. Require it to reject:

1. `torch.cuda.is_available() == False` when CUDA is required;
2. an architecture list without `sm_120`;
3. non-finite forward or backward results;
4. an unavailable native exact solver;
5. checkpoint hashes that differ from the config.

Also test that a successful report contains Python, torch, compiled CUDA, GPU name, compute capability, architecture list, forward/backward checksums, solver availability, both frozen artifact hashes, and the resolved fixed-play source-manifest hash.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_preflight.py
```

Expected: import failure because `residual_bidder.cli.preflight` does not exist.

- [ ] **Step 3: Implement the runtime inspector and CLI**

The CLI interface is:

```bash
python -m residual_bidder.cli.preflight \
  --config configs/residual_bidder/base.yaml \
  --require-cuda \
  --require-sm sm_120 \
  --require-native-solver
```

On GPU, run a fixed-seed `4096 x 4096` matrix multiply plus backward pass in float32, synchronize CUDA, and fail unless outputs and gradients are finite. Instantiate `ExactDoubleDummyCppFastestSolver` and require `native_available=True`. Emit one JSON object and exit nonzero on any failed gate.

Keep torch out of `requirements-bidder.txt`; list `numpy>=1.26,<3`, `pyyaml>=6,<7`, `tqdm>=4.66,<5`, and `pytest>=8,<9`. Install torch independently in a persistent venv:

```bash
python3 -m venv /root/venvs/residual-bidder
/root/venvs/residual-bidder/bin/python -m pip install --upgrade pip
/root/venvs/residual-bidder/bin/python -m pip install 'torch==2.12.*'
/root/venvs/residual-bidder/bin/python -m pip install -r /root/Spades-AI/requirements-bidder.txt
```

Do not modify the image's system torch.

Create `configs/residual_bidder/base.yaml` in this task with the complete, final sections enumerated in Task 10: verified paths/hashes, fixed play configuration, worker counts, seed namespaces, data mixture, model/training values, lambda grid, promotion statistics, and persistent paths. Early modules read only the sections they own; later tasks add validation and behavior, not ad-hoc config files.

- [ ] **Step 4: Verify GREEN locally and on the pod**

Local test command: repeat Step 2; expect all tests to pass.

After the implementation commit has been synchronized to `/root/Spades-AI`, run the CLI inside the current pod. Expected report includes `"gpu_name": "NVIDIA GeForce RTX 5090"`, `"compute_capability": [12, 0]`, `"sm_120_supported": true`, `"native_solver": true`, and `"ok": true`. Do not proceed to a training smoke run if any field fails.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add requirements-bidder.txt configs/residual_bidder/base.yaml \
  residual_bidder/__init__.py residual_bidder/cli/__init__.py \
  residual_bidder/cli/preflight.py tests/residual_bidder/test_preflight.py
git commit -m "build: add residual bidder runtime gate"
```

## Task 2: Implement the Frozen NSFP Observation and Hard Local Action Space

**Files:**
- Create: `residual_bidder/actions.py`
- Create: `residual_bidder/nsfp.py`
- Create: `tests/residual_bidder/test_actions.py`
- Create: `tests/residual_bidder/test_nsfp.py`
- Reference unchanged: `evaluate/GO-MCTS/bridge.py`
- Reference unchanged: `Spades_AI_GO-MCTS/spades_ai/models/bid_encoder.py`
- Reference unchanged: `Spades_AI_GO-MCTS/spades_ai/models/bid_mlp.py`

- [ ] **Step 1: Write action-alias and neighborhood tests**

Define `BidAction(IntEnum)` with `NIL=0` through `BID_13=13`, and:

```python
@dataclass(frozen=True)
class LocalNeighborhood:
    center: BidAction
    lower: BidAction | None
    upper: BidAction | None

def legal_action_scores(raw_logits: torch.Tensor) -> torch.Tensor: ...  # (..., 16) -> (..., 14)
def local_neighborhood(center: BidAction) -> LocalNeighborhood: ...
def to_local_bid(action: BidAction) -> str: ...
```

Tests must prove:

- Nil score is `max(raw[14], raw[15])`;
- bid-1 score is `max(raw[0], raw[1])`;
- bid-2..13 scores are raw slots 2..13;
- neighborhoods are exactly `{Nil,1}`, `{Nil,1,2}`, `{b-1,b,b+1}`, and `{12,13}` at the four boundaries;
- blind nil and normal zero can never be returned as acting actions;
- stable score ties select the lowest `BidAction` index at center construction.

- [ ] **Step 2: Write exact encoder-reuse and no-leak tests**

Define:

```python
@dataclass(frozen=True)
class NSFPObservation:
    encoded_149: torch.Tensor
    raw_logits_16: torch.Tensor
    legal_scores_14: torch.Tensor
    center: BidAction

class FrozenNSFP:
    @classmethod
    def load(cls, checkpoint: Path, device: str, expected_sha256: str) -> "FrozenNSFP": ...
    def observe(self, local_state: GameState) -> NSFPObservation: ...
```

Build a bidding state with a nonzero dealer. Compare `encoded_149` bit-for-bit against the existing current path:

```python
go_state = to_go_state(local_state)
features = BidEncoder().encode(
    list(go_state.hands[go_state.current_player]),
    list(go_state.bids),
    len(go_state.bids),
)
```

Mutate every opponent hand while preserving the acting hand and public auction; assert `encoded_149`, logits, and center are unchanged. This test prevents full-deal leakage without changing the legacy bridge semantics.

- [ ] **Step 3: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_actions.py tests/residual_bidder/test_nsfp.py
```

Expected: missing modules/classes.

- [ ] **Step 4: Implement actions and frozen observation**

Load `BidMLP` with `weights_only=True`, freeze parameters, call `eval()`, and preserve the exact bridge/encoder path above. Validate tensor ranks, exact dimensions, finite values, checkpoint hash, and device. The module must not expose or reuse NSFP hidden activations.

- [ ] **Step 5: Verify GREEN and legacy equivalence**

Run Step 3 and a 1,000-state randomized equivalence test comparing `FrozenNSFP.center` converted to a local bid against current `MLPBidPlayer.choose_bid` plus `normalize_bid_for_legal_options`. Exact-logit tie fixtures follow the new explicit legal-action tie rule; ordinary checkpoint states must match 100%.

- [ ] **Step 6: Commit**

```bash
git add residual_bidder/actions.py residual_bidder/nsfp.py \
  tests/residual_bidder/test_actions.py tests/residual_bidder/test_nsfp.py
git commit -m "feat: freeze nsfp residual observations"
```

## Task 3: Build the 167-D Feature Contract, Five-Member Model, and Deterministic Policy

**Files:**
- Create: `residual_bidder/model.py`
- Create: `residual_bidder/checkpoint.py`
- Create: `residual_bidder/policy.py`
- Create: `tests/residual_bidder/test_model.py`
- Create: `tests/residual_bidder/test_policy.py`

- [ ] **Step 1: Write feature-contract tests**

Define:

```python
MARGIN_TEMPERATURE = 13.47

@dataclass(frozen=True)
class ResidualInput:
    values: torch.Tensor       # (167,)
    neighborhood: LocalNeighborhood
    masks: torch.Tensor        # (2,), lower then upper

def build_residual_input(observation: NSFPObservation) -> ResidualInput: ...
```

Assert exact slices:

- `[0:149]`: unchanged `encoded_149`;
- `[149:163]`: one-hot NSFP center;
- `[163:165]`: `(score(center)-score(lower_or_upper))/13.47`, zero when absent;
- `[165:167]`: lower/upper masks.

Test Nil, bid-1, ordinary, and bid-13 boundary rows.

- [ ] **Step 2: Write architecture, decision, and fallback tests**

Required public interfaces:

```python
class ResidualQMember(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (..., 167) -> (..., 2)

class ResidualQEnsemble(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (5, ..., 2)

@dataclass(frozen=True)
class BidDecision:
    action: BidAction
    center: BidAction
    mean: tuple[float, float]
    std: tuple[float, float]
    conservative_values: tuple[float, float]
    fallback_reason: str | None

class CompositeBidPolicy:
    def decide(self, local_state: GameState, legal_bids: Sequence[Any]) -> BidDecision: ...
```

Use five wholly independent members with the approved 167→256→two residual blocks→128→2 architecture, LayerNorm and SiLU, no dropout. Tests assert separate parameter storage, exact shapes, and no stochastic output in eval mode.

Decision fixtures must prove `V=mean-lambda*std`, invalid alternatives are masked, and ties choose center, then lower, then upper. NaN, Inf, wrong dimensions, checkpoint mismatch, or unavailable residual model return center with a nonempty fallback reason only in runtime mode; `strict=True` raises.

- [ ] **Step 3: Write checkpoint compatibility tests**

Checkpoint metadata must contain:

```text
schema_version, architecture_id, member_count, input_dim, output_dim,
nsfp_sha256, belief_nsfp_sha256, play_pipeline_sha256, play_config_sha256,
iteration, policy_id, dataset_manifest_sha256, lambda,
model_init_seeds, training_seed, git_commit, torch_version
```

Save to a temporary file, fsync, then `os.replace`. Reject changed hashes, wrong member count/dimensions, a missing member, non-finite parameters, and unknown schema versions.

- [ ] **Step 4: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_model.py tests/residual_bidder/test_policy.py
```

- [ ] **Step 5: Implement feature construction, models, checkpoint IO, and policy**

Use population standard deviation consistently (`unbiased=False`). Keep predictions in target/100 units through decision time. Convert the chosen `BidAction` to an exact member of `legal_bids`; if the expected local bid is absent, strict mode fails and runtime mode returns the normalized center.

- [ ] **Step 6: Verify GREEN and commit**

```bash
git add residual_bidder/model.py residual_bidder/checkpoint.py residual_bidder/policy.py \
  tests/residual_bidder/test_model.py tests/residual_bidder/test_policy.py
git commit -m "feat: add residual bid ensemble policy"
```

## Task 4: Isolate Acting Bids with a Transparent Player Proxy

**Files:**
- Create: `residual_bidder/player_proxy.py`
- Create: `residual_bidder/play_factory.py`
- Create: `tests/residual_bidder/test_player_proxy.py`
- Modify only if a characterization test requires it: none of `strategy/rule_exact_first4_player.py`, `trick_taking/games/spades.py`, or the play models

- [ ] **Step 1: Write proxy-forwarding characterization tests**

Define:

```python
class ActingBidPlayerProxy(AIPlayer):
    def __init__(self, play_player: AIPlayer, acting_policy: ActingBidPolicy): ...
```

The proxy overrides only `place_bid`. It forwards `start_game`, every `bid_placed`, `set_teams`, `play_card`, and every `card_played` call exactly once with object identity/order preserved. `play_card` must be byte-for-byte behaviorally equivalent to calling the wrapped fixed player directly on the same deterministic state.

- [ ] **Step 2: Write fixed-factory tests**

Define `FrozenPlayAssets` and:

```python
def build_fixed_play_partnership(
    seats: tuple[int, int],
    acting_policy: ActingBidPolicy,
    assets: FrozenPlayAssets,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    exact_workers: int = 1,
) -> dict[int, ActingBidPlayerProxy]: ...
```

Assert:

- candidate seats reference the same `acting_policy` object;
- opponent seats reference the same selected opponent policy object;
- all four wrapped play players are fresh instances;
- all use the currently deployed `RuleExactFirst4NilPlayer(num_workers=1)`, `exact_threshold=36`, and `configs/8.yaml`;
- no `55_2nil.pt` policy is loaded, because the deployed Nil subclass uses `RuleBasedFirst4NilPlayer` for the first four tricks;
- the wrapped player's acting `_bid_model` is not used;
- calling its `_ensure_bid_model_loaded()` still resolves the original `bid_nsfp.pt` hash, proving the belief bidder was not replaced.

- [ ] **Step 3: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_player_proxy.py
```

- [ ] **Step 4: Implement without editing the frozen play path**

Share the native solver and immutable acting-policy assets within one process, but instantiate fresh stateful `RuleExactFirst4NilPlayer` and proxy objects for every replayed game. The subclass's existing rule-based Nil path remains untouched. Keep acting inference on CPU during rollouts.

- [ ] **Step 5: Run integration equivalence**

On 100 fixed seeds, compare an NSFP-only proxy against the existing acting-bid path. Require identical bids, card sequence, trick counts, and final scores. Separately assert that two composite-proxy seats use one residual checkpoint while their internal belief bidder remains NSFP.

- [ ] **Step 6: Commit**

```bash
git add residual_bidder/player_proxy.py residual_bidder/play_factory.py \
  tests/residual_bidder/test_player_proxy.py
git commit -m "feat: isolate acting bidder from fixed play"
```

## Task 5: Implement Deterministic Duplicate Counterfactual Rollouts

**Files:**
- Create: `residual_bidder/seeds.py`
- Create: `residual_bidder/counterfactual.py`
- Create: `tests/residual_bidder/test_seeds.py`
- Create: `tests/residual_bidder/test_counterfactual.py`

- [ ] **Step 1: Write partition and seed-bundle tests**

Define:

```python
class Partition(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    PROMOTION = "promotion"
    FINAL = "final"
    RESERVOIR = "reservoir"

@dataclass(frozen=True)
class SeedBundle:
    schema_version: str
    partition: Partition
    deal_index: int
    deal_seed: int
    room: int
    python_seed: int
    numpy_seed: int
    torch_seed: int

def seed_bundle(partition: Partition, deal_index: int, room: int) -> SeedBundle: ...
```

Derive all values from BLAKE2b over `residual-bidder-seeds-v1|partition|deal_index|room`. Assert namespaces never overlap in a million generated test keys, room 0/1 have the same `deal_seed`, and repeated construction is identical.

- [ ] **Step 2: Write baseline/branch scheduling tests**

Public data types:

```python
@dataclass(frozen=True)
class CapturedBidState:
    candidate_ordinal: int
    bidder: int
    observation: NSFPObservation
    residual_input: ResidualInput
    neighborhood: LocalNeighborhood
    baseline_action: BidAction
    chronological_auction: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class BranchDirective:
    target_candidate_ordinal: int
    forced_action: BidAction

def run_room(..., directive: BranchDirective | None) -> RoomRollout: ...
def run_duplicate_deal(...) -> DuplicateRollout: ...
```

Use a fast deterministic fake player first. In each room, run one incumbent baseline and capture only the two candidate-team bidding states on that baseline path. For each captured state, schedule exactly `neighborhood actions - {baseline_action}`. The replay starts from the original deal seed with fresh players; it follows incumbent before and after the target call, and forces only that call.

Critical later-iteration fixture: if incumbent selected lower, reuse baseline return as `R(lower)`, run center and upper once each, and form both `R(lower)-R(center)` and `R(upper)-R(center)`. Never treat incumbent's selected action as the NSFP center.

- [ ] **Step 3: Write duplicate and reward tests**

Room 0 uses candidate seats `(0,2)` and room 1 uses candidate seats `(1,3)` on the identical deal/dealer. Opponent ownership swaps accordingly. Compute each room's candidate margin as the mean of its two candidate-seat values from `SpadesRules.score`; because those values are already own-team minus opponent-team, do not subtract the opponent a second time.

Assert:

- affected-room label is `margin(forced action)-margin(forced center)`;
- the constant other room cancels from that difference;
- two room margins average to the duplicate match margin;
- all branches retain deal, room, opponent, incumbent, and continuation version;
- one duplicate produces two baselines plus at most eight additional games;
- boundary states produce fewer games;
- target arrays always use `[lower-center, upper-center]` and masks.

- [ ] **Step 4: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_seeds.py \
  tests/residual_bidder/test_counterfactual.py
```

- [ ] **Step 5: Implement replay-based rollouts**

Set Python/NumPy/torch seeds at each fresh room replay. Do not clone a player, RNG, native solver state, or partially played `SpadesMatchRunner`. Identify a target by candidate-team bid ordinal plus an assertion on bidder, public auction, encoded-149 checksum, and legal neighborhood; fail if replay diverges before forcing.

For real rollouts instantiate `SpadesRules(enable_nil=True, enable_blind_nil=False)`. Record both team payoff values (`mean(scores[0],scores[2])`, `mean(scores[1],scores[3])`), candidate margin, tricks, bids, and the full continuation auction.

- [ ] **Step 6: Verify with real fixed play**

Run two duplicate seeds twice in strict debug mode. Require identical branch manifests and scores. Run 100 fake fast deals to prove the ≤10 invariant and action-target arithmetic.

- [ ] **Step 7: Commit**

```bash
git add residual_bidder/seeds.py residual_bidder/counterfactual.py \
  tests/residual_bidder/test_seeds.py tests/residual_bidder/test_counterfactual.py
git commit -m "feat: generate duplicate bid counterfactuals"
```

## Task 6: Add Auditable, Atomic, Deal-Grouped Dataset Storage

**Files:**
- Create: `residual_bidder/records.py`
- Create: `residual_bidder/shards.py`
- Create: `tests/residual_bidder/test_records.py`
- Create: `tests/residual_bidder/test_shards.py`

- [ ] **Step 1: Write schema and validation tests**

Use schema `residual-bidder-data-v1`. A branch record contains every field required by design section 20, including frozen hashes, exact 149-vector, 14 legal scores, center, masks, forced action, continuation, seed bundle, tricks, both team payoff values, candidate margin, target, stratum, sampling probability, and importance metadata.

A state tensor row combines the branches for one captured state:

```python
@dataclass(frozen=True)
class TrainingRow:
    deal_id: str
    state_id: str
    partition: Partition
    features_167: np.ndarray
    targets_2: np.ndarray
    masks_2: np.ndarray
    natural_frequency: float
    sampling_probability: float
    importance_weight: float
    opponent_id: str
    policy_id: str
```

Reject wrong dimensions/dtypes, non-finite values, target values in a masked slot, missing hashes, branch/policy mismatches, duplicated forced actions, and children assigned to a different partition.

- [ ] **Step 2: Write atomic shard tests**

One shard directory contains:

```text
manifest.json
branches.jsonl.gz
states.npz
```

`states.npz` uses only numeric arrays and fixed-width Unicode IDs; no pickled object arrays. Write all files under a sibling temporary directory, fsync files and directory, validate by reopening, compute SHA-256/row counts, then rename atomically with format `shard-{partition}-{start_index:012d}-{end_index:012d}-{digest[:16]}`. Inject failures before rename and assert no admissible shard remains.

Maintain `dataset-index.sqlite3` with `UNIQUE(partition, deal_id)` and admit a shard in one transaction after filesystem validation. A duplicate deal ID is a hard error.

- [ ] **Step 3: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_records.py tests/residual_bidder/test_shards.py
```

- [ ] **Step 4: Implement records, shard writer/reader, and index**

Manifest fields include schema, partition, deal/state/branch counts, deal range, incumbent and league manifest hashes, every frozen asset hash, file hashes, creation command, git commit, and completion marker. Readers ignore any directory without a valid completion marker and matching hashes.

- [ ] **Step 5: Verify GREEN and commit**

```bash
git add residual_bidder/records.py residual_bidder/shards.py \
  tests/residual_bidder/test_records.py tests/residual_bidder/test_shards.py
git commit -m "feat: store residual bidder rollout shards"
```

## Task 7: Implement Natural/Reservoir Sampling and Parallel Generation

**Files:**
- Create: `residual_bidder/league.py`
- Create: `residual_bidder/reservoir.py`
- Create: `residual_bidder/cli/generate.py`
- Create: `tests/residual_bidder/test_league.py`
- Create: `tests/residual_bidder/test_reservoir.py`
- Create: `tests/residual_bidder/test_generate_cli.py`

- [ ] **Step 1: Write immutable league tests needed by generation**

Represent NSFP as a policy member with residual disabled. Accepted snapshots are content-addressed and immutable. Exact weights are NSFP 1.00 before any acceptance; NSFP/latest 0.50/0.50 with one accepted snapshot; and NSFP/latest/older 0.50/0.25/0.25 with the older share uniform when there are at least two.

Hash league manifests and sample one opponent from a deterministic deal key. Both opponent seats and every branch/room for a deal use that same member. Reject a changed file behind an existing member ID. These interfaces are consumed by the generation CLI in this task and extended only with promotion results in Task 9.

- [ ] **Step 2: Write outcome-blind reservoir tests**

Define strata as the cross-product of center bucket (`Nil,1,2,3,4,5,6,7+`), bidding position (`0..3`), nil already seen, partner visible, and opponent member. Reservoir acceptance receives only `CapturedBidState` and opponent metadata—never branch returns.

Use deterministic per-stratum Algorithm R keyed by `(reservoir-v1, stratum, seen_count)`. After a scan, each retained row records inclusion probability `min(1, capacity_h/seen_h)` and observed natural frequency `seen_h/seen_total`.

Test that changing rewards cannot change membership, restart reproduces membership, and a synthetic imbalanced population is reconstructed after weighting.

- [ ] **Step 3: Define training-mixture weights exactly**

The base configuration samples 75% of rows from the natural pool and 25% from the reservoir. For stratum `h`, compute the actual minibatch stratum probability `q_train(h)` from that mixture and store/use:

```text
w(h) = p_natural(h) / q_train(h)
```

Normalize weights to mean 1 inside the legal-alternative loss denominator. The natural pool has inclusion probability 1. Test exact recovery of known conditional means under deliberately distorted `q_train`.

- [ ] **Step 4: Write generation CLI tests**

CLI:

```bash
CUDA_VISIBLE_DEVICES='' python -m residual_bidder.cli.generate \
  --config configs/residual_bidder/base.yaml \
  --run-dir /root/residual-bidder/runs/main \
  --iteration 0 \
  --partition train \
  --start-deal-index 0 \
  --num-deals 256 \
  --outer-workers 6 \
  --exact-workers 1
```

Reject an environment with visible CUDA in formal generation, `exact_workers != 1`, changed asset hashes, a mutable league manifest, overlapping admitted deal IDs, or a non-frozen incumbent.

Each outer worker owns complete duplicate deals and writes a worker-local staging shard. It loads immutable assets once, creates fresh players for every replay, and never lets branches from one deal cross workers.

- [ ] **Step 5: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_reservoir.py \
  tests/residual_bidder/test_league.py tests/residual_bidder/test_generate_cli.py
```

- [ ] **Step 6: Implement league sampling plus natural and reservoir generation**

Natural generation branches every captured candidate state. Reservoir generation uses the separate `reservoir` seed namespace, scans unfiltered baseline rooms, selects states before seeing any counterfactual outcome, and only then runs the required local branches for retained states.

Emit per-block throughput metrics: baseline/branch games, states, games/deal, seconds/game, center histogram, correction slots, solver failures, and estimated time/storage per 10,000 unique deals.

- [ ] **Step 7: Run an 8-deal CPU smoke block**

Expected: 8 unique duplicate IDs, no more than 80 full games, all shard hashes valid, no GPU context, and deterministic rerun hashes after excluding creation timestamps.

- [ ] **Step 8: Commit**

```bash
git add residual_bidder/league.py residual_bidder/reservoir.py residual_bidder/cli/generate.py \
  tests/residual_bidder/test_league.py tests/residual_bidder/test_reservoir.py \
  tests/residual_bidder/test_generate_cli.py
git commit -m "feat: parallelize residual rollout generation"
```

## Task 8: Train and Resume the Five-Member Residual Ensemble

**Files:**
- Create: `residual_bidder/dataset.py`
- Create: `residual_bidder/training.py`
- Create: `residual_bidder/cli/train.py`
- Create: `tests/residual_bidder/test_dataset.py`
- Create: `tests/residual_bidder/test_training.py`

- [ ] **Step 1: Write deal-grouping and bootstrap tests**

All rows from a duplicate deal stay in one split and minibatch group. Define deterministic bootstrap multiplicity:

```python
def bootstrap_multiplicity(member: int, deal_id: str) -> int:
    """Poisson(1) draw from a BLAKE2b-derived local NumPy seed."""
```

Assert restart identity, independence across member/deal keys, approximate Poisson(1) frequencies on 100,000 synthetic deals, and one multiplicity shared by all rows/rooms/branches of a deal.

- [ ] **Step 2: Write loss and synthetic recovery tests**

Implement:

```python
def masked_weighted_mse(
    predictions: torch.Tensor,  # (5, B, 2)
    targets: torch.Tensor,      # (B, 2), raw score points
    masks: torch.Tensor,        # (B, 2)
    importance: torch.Tensor,   # (B,)
    bootstrap: torch.Tensor,    # (5, B)
) -> torch.Tensor: ...          # scalar
```

Targets are divided by 100 inside the loss and never clipped. Weight each member/row/slot by legality × importance × bootstrap multiplicity and divide by the sum of those weights. Test exact arithmetic, extreme Nil/contract tails, zero illegal-slot gradients, and recovery of known conditional means. Do not use Huber loss.

- [ ] **Step 3: Write resume and identity tests**

A training state stores epoch, optimizer/scheduler states, RNG states, admitted shard manifest hashes, incumbent/league/config hashes, best development metrics, and all five model states. Resume must produce the same next minibatch and parameters as uninterrupted training. Refuse any dataset or policy identity drift.

- [ ] **Step 4: Verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_dataset.py tests/residual_bidder/test_training.py
```

- [ ] **Step 5: Implement training**

Base settings:

```text
members=5, batch_size=4096, optimizer=AdamW,
learning_rate=3e-4, weight_decay=1e-4,
max_epochs=200, early_stop_patience=15,
gradient_norm_clip=1.0, precision=float32,
model_init_seeds=[1701,1702,1703,1704,1705]
```

Set deterministic algorithms where supported. Log weighted MSE, sign accuracy, calibration by predicted-advantage bucket, offline regret, ensemble disagreement, per-stratum metrics, and fixed-probe selected actions. Save last and best states atomically.

CLI:

```bash
python -m residual_bidder.cli.train \
  --config configs/residual_bidder/base.yaml \
  --run-dir /root/residual-bidder/runs/main \
  --iteration 0 \
  --device cuda:0 \
  --resume
```

- [ ] **Step 6: Run CPU synthetic and 5090 smoke fits**

CPU synthetic test must overfit 256 deterministic rows to near-zero MSE. GPU smoke uses the 8-deal shard, performs at least two optimizer steps on all five members, reloads the checkpoint, and obtains identical eval predictions. Monitor with `nvidia-smi`; any `sm_120` compatibility warning fails the gate.

- [ ] **Step 7: Commit**

```bash
git add residual_bidder/dataset.py residual_bidder/training.py residual_bidder/cli/train.py \
  tests/residual_bidder/test_dataset.py tests/residual_bidder/test_training.py
git commit -m "feat: train residual bid ensemble"
```

## Task 9: Implement Immutable League Sampling and Duplicate Promotion

**Files:**
- Modify: `residual_bidder/league.py`
- Create: `residual_bidder/promotion.py`
- Create: `residual_bidder/cli/evaluate.py`
- Modify: `tests/residual_bidder/test_league.py`
- Create: `tests/residual_bidder/test_promotion.py`

- [ ] **Step 1: Extend league-manifest tests for acceptance**

Retain the generation-time guarantees from Task 7 and add atomic acceptance tests. Exact weights remain:

- no accepted snapshot: NSFP 1.00;
- one accepted snapshot: NSFP 0.50, latest 0.50;
- two or more: NSFP 0.50, latest 0.25, older accepted members uniformly share 0.25.

Hash league manifests and sample opponents from a deterministic deal key. Both opponent seats and every branch/room for a deal use the same selected member. Reject a changed file behind an existing member ID.

- [ ] **Step 2: Write promotion-statistics tests**

For each deal/opponent compute:

```text
Z = M(new, opponent, deal) - M(incumbent, opponent, deal)
```

Keep all four games and every opponent stratum for a deal together in 10,000 deterministic bootstrap resamples. Bootstrap deals within opponent strata, then combine stratum means with frozen league weights.

Required tests:

- swapping partnerships cancels deal strength in a constructed example;
- bootstrapping rooms or branches instead of deals is rejected;
- equal policies do not pass the strict `lower_bound > 0` rule;
- a known superior synthetic policy passes with adequate power;
- NSFP-anchor and every predeclared ≥5% stratum use the one-sided upper-bound non-regression check;
- promotion data cannot tune lambda or any veto.

- [ ] **Step 3: Freeze sample size before promotion results**

Use development `Z` standard deviation `s`, one-sided alpha 0.05, power 0.80, and a predeclared minimum detectable improvement of 1.0 raw score point:

```text
n = ceil(((z_0.95 + z_0.80) * s / 1.0)^2)
```

Round up to a multiple of 256 and require at least 4,096 unique duplicate deals. Write seeds, opponent allocations, strata, lambda, all thresholds, and the exact candidate/incumbent hashes to `promotion-manifest.json` before playing any promotion game. There is no upper cap; budget exhaustion pauses the run but does not weaken the criterion.

- [ ] **Step 4: Tune lambda only on development deals**

Evaluate fixed grid `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]` on the same frozen development seed/opponent manifest. Choose maximum paired duplicate margin; exact ties choose the larger lambda, then record it in candidate checkpoint metadata. Promotion never revisits this choice.

- [ ] **Step 5: Implement the evaluation CLI**

```bash
CUDA_VISIBLE_DEVICES='' python -m residual_bidder.cli.evaluate \
  --config configs/residual_bidder/base.yaml \
  --run-dir /root/residual-bidder/runs/main \
  --iteration 0 \
  --mode promotion \
  --outer-workers 6 \
  --exact-workers 1
```

Promotion passes only when the overall one-sided 95% lower bound is >0, the NSFP anchor one-sided upper bound is ≥0, and every predeclared ≥5% stratum upper bound is ≥0. Contract/Nil/overtrick metrics are diagnostics, never undeclared vetoes.

- [ ] **Step 6: Verify and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_league.py tests/residual_bidder/test_promotion.py
git add residual_bidder/league.py residual_bidder/promotion.py residual_bidder/cli/evaluate.py \
  tests/residual_bidder/test_league.py tests/residual_bidder/test_promotion.py
git commit -m "feat: promote bidders by duplicate margin"
```

## Task 10: Orchestrate Frozen Iterations and Data-Growth Stopping

**Files:**
- Create: `residual_bidder/iteration.py`
- Create: `residual_bidder/cli/iterate.py`
- Modify: `configs/residual_bidder/base.yaml`
- Create: `tests/residual_bidder/test_iteration.py`

- [ ] **Step 1: Write the state-machine tests**

States are `CREATED -> GENERATING -> TRAINING -> DEV_EVALUATION -> PROMOTION_READY -> PROMOTING -> ACCEPTED|REJECTED|PAUSED`. Every transition atomically records the command, input/output hashes, timestamps, and reason. Restart resumes the incomplete transition without changing incumbent or league.

Tests reject label generation after incumbent mutation, unmarked old-continuation data, fewer than three completed blocks before stopping, promotion seed reuse, and final-test access before the explicitly requested final evaluation.

- [ ] **Step 2: Encode the approved stopping rule**

Production block size is 256 unique duplicate deals; an 8-deal block is permitted only with `--smoke`. After at least three blocks, stop adding data to the iteration only when both hold across the last three additions:

1. development duplicate-margin improvement is no more than one deal-bootstrap standard error each time;
2. fewer than 0.5% of fixed probe states change selected action between consecutive candidates.

Otherwise generate another unique block without changing the incumbent.

- [ ] **Step 3: Validate and finalize the base config without placeholders**

Validate that the Task 1 config includes the verified artifact paths/hashes, exact threshold 36, nil enabled/blind nil disabled, 6 outer workers, 1 exact worker, block size 256, natural/reservoir mixture 0.75/0.25, model/training values from Task 8, lambda grid, bootstrap count 10,000, promotion alpha/power/MDE/minimum deals, and persistent path `/root/residual-bidder/runs/main`.

The config must not mention `bid_dds.pt` and must identify `bid_nsfp.pt` separately as `acting_center_checkpoint` and `belief_checkpoint`, with the same expected hash.

- [ ] **Step 4: Implement orchestration with subprocess isolation**

The iterator invokes generation/evaluation in a child environment with `CUDA_VISIBLE_DEVICES=""`, waits for admitted shard/result manifests, then invokes training with `CUDA_VISIBLE_DEVICES=0`. This prevents CUDA initialization in forked rollout workers. On SIGTERM, finish or discard the current temporary shard, save run state, and exit so Kubernetes can resume safely.

Only accepted candidates are copied into the immutable league and become the next incumbent. A rejection leaves incumbent/league unchanged and ends that iteration.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_iteration.py
git add residual_bidder/iteration.py residual_bidder/cli/iterate.py \
  configs/residual_bidder/base.yaml tests/residual_bidder/test_iteration.py
git commit -m "feat: orchestrate fitted bidder iterations"
```

## Task 11: Integrate Only the Promoted Acting Bidder into Runtime Entry Points

**Files:**
- Modify: `gui/backend.py`
- Modify: `gui/game_server.py`
- Modify: `tests/test_gui_backend.py`
- Create: `tests/residual_bidder/test_runtime_integration.py`

- [ ] **Step 1: Write default-path characterization tests**

Before changing either runtime entry point, capture fixed bidding and playing payloads. With no residual checkpoint configured, require exactly the current NSFP bid, card action, `last_play_info`, and response payload. This is the backward-compatibility baseline.

- [ ] **Step 2: Write acting-only integration tests**

Add an optional CLI/config value:

```text
--residual-bid-checkpoint PATH
```

The default is absent. When present, load a promoted `CompositeBidPolicy` and use it only for the AI seats' `place_bid`; card actions still call the same `RuleExactFirst4NilPlayer.play_card`. The two AI partnership seats in `gui/game_server.py` share one policy object/checkpoint.

Test with a synthetic valid checkpoint that forces lower or upper:

- the bid changes only within the approved neighborhood;
- the same play payload returns the same card with and without residual bidding;
- the underlying player's internal belief loader still resolves the frozen NSFP hash;
- opponent placeholder hand identities in `gui/backend.py` cannot affect the bid;
- invalid/NaN/mismatched runtime checkpoints log a structured fallback and return the NSFP center;
- repeated identical requests return the same final action;
- default/no-argument behavior is bit-for-bit unchanged.

- [ ] **Step 3: Implement one shared loader and acting-policy dispatch**

Use `residual_bidder.checkpoint` and `residual_bidder.policy`; do not duplicate model definitions in `gui`. `RuleExactProvider` holds an optional shared acting policy and calls it in `_choose_bid`; `_choose_play` remains unchanged. `gui/game_server.py` wraps only its AI players' bid calls with `ActingBidPlayerProxy` or an equivalent direct acting-policy dispatch while preserving every play callback.

Expose `acting_policy_id`, center checkpoint hash, residual checkpoint hash, and cumulative fallback count in backend health/startup diagnostics. A promoted checkpoint hash is required; a development or smoke checkpoint is rejected and falls back to NSFP.

- [ ] **Step 4: Verify RED, implement, then verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/test_gui_backend.py \
  tests/residual_bidder/test_runtime_integration.py
```

Run before implementation to observe the missing argument/dispatch failures, then repeat after implementation and require all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add gui/backend.py gui/game_server.py tests/test_gui_backend.py \
  tests/residual_bidder/test_runtime_integration.py
git commit -m "feat: serve promoted residual acting bidder"
```

## Task 12: Package the Kubernetes Run Without Hiding the CPU Bottleneck

**Files:**
- Create: `deploy/k8s/residual-bidder-job.yaml`
- Create: `tests/residual_bidder/test_k8s_manifest.py`
- Modify: `README.md` (add a concise residual-bidder runbook link/section only)

- [ ] **Step 1: Write manifest validation tests**

The manifest must use:

```text
namespace: cs-luohaoren-mwku
image: harbor.xa.hqzyai.com:19443/llm-course/lab:v2
nodeSelector: gpu-type=5090
requests/limits: cpu=8, memory=32Gi, nvidia.com/gpu=1
PVC: pvc-workspace-home mounted at /root
restartPolicy: Never
backoffLimit: 0
terminationGracePeriodSeconds: 60
metadata.name: residual-bidder
```

The command sources `/root/venvs/residual-bidder`, changes to `/root/residual-bidder/code/current`, sets `PYTHONPATH=Spades_AI_GO-MCTS:.`, and runs `python -m residual_bidder.cli.iterate --config configs/residual_bidder/base.yaml --run-dir /root/residual-bidder/runs/main --resume`.

Do not mount the RWO `/scratch` PVC into the Job; persistent state stays on the 200 GiB RWX `/root` PVC. The current Deployment already holds the GPU, so the runbook must show a guarded handoff rather than creating a permanently Pending second claimant.

- [ ] **Step 2: Add exact code-sync and environment commands**

After all implementation commits and tests pass, synchronize committed content only into a new content-addressed directory; never copy the dirty working tree or overlay an older release:

```bash
release_commit="$(git rev-parse HEAD)"
release_dir="/root/residual-bidder/code/${release_commit}"
kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  exec deploy/my-task-dev -- mkdir -p "${release_dir}"
git archive --format=tar HEAD | \
  kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  exec -i deploy/my-task-dev -- tar -xf - -C "${release_dir}"
kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  exec deploy/my-task-dev -- ln -sfn "${release_dir}" /root/residual-bidder/code/current
```

Create the persistent venv and run Task 1 preflight from the `current` symlink. Record `git rev-parse HEAD`, config hash, `pip freeze`, preflight JSON, and `nvidia-smi` into `/root/residual-bidder/environment/`.

- [ ] **Step 3: Require an explicit launch confirmation**

Only after the user approves the GPU handoff window:

1. record `kubectl get deploy my-task-dev -o yaml` and current replica count;
2. scale `my-task-dev` from 1 to 0;
3. wait until its GPU pod terminates;
4. apply `deploy/k8s/residual-bidder-job.yaml`;
5. wait for the Job pod to pass preflight and begin/resume the state machine;
6. on completion/pause, delete only that named Job and restore the Deployment to its recorded replica count.

If any handoff step fails, restore the original Deployment replica count before proceeding. Never delete the PVC or run directory.

- [ ] **Step 4: Add observability commands**

The runbook includes these exact log/GPU checks:

```bash
kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  logs -f job/residual-bidder
training_pod="$(kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  get pod -l job-name=residual-bidder -o jsonpath='{.items[0].metadata.name}')"
kubectl --context cs-luohaoren-mwku@yw-k8s -n cs-luohaoren-mwku \
  exec "${training_pod}" -- nvidia-smi
```

Also include run-state inspection, shard validation, estimated games/second, and disk-usage checks. Explicitly explain that the 5090 will be lightly used while the 8 CPU cores generate exact-play rollouts and heavily used only during short fitting phases; low GPU utilization during generation is not a malfunction.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. \
  python3 -m pytest -q tests/residual_bidder/test_k8s_manifest.py
git add deploy/k8s/residual-bidder-job.yaml tests/residual_bidder/test_k8s_manifest.py README.md
git commit -m "ops: run residual bidder training on 5090"
```

## Task 13: End-to-End Verification Before Any Production Deals

**Files:**
- Review all files created or modified in Tasks 1-12.
- Do not edit unrelated dirty-worktree files.

- [ ] **Step 1: Run the complete local regression suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Spades_AI_GO-MCTS:. python3 -m pytest -q tests
```

Expected: all existing bug-fix tests and all residual-bidder tests pass.

- [ ] **Step 2: Run static artifact checks**

```bash
git diff --check
git status --short
```

Confirm no edit to `BidEncoder`, `BidMLP`, the bridge, fixed play semantics, `bid_nsfp.pt`, or `configs/8.yaml`; confirm no reference to `bid_dds.pt` in `residual_bidder`, its config, runtime integration, or deployment.

- [ ] **Step 3: Run a real deterministic end-to-end smoke**

On the cloud runtime:

1. preflight passes;
2. generate 8 natural duplicate deals with CUDA hidden;
3. validate/reopen shards;
4. fit all five members for two GPU optimizer steps;
5. tune lambda on a tiny smoke-only development set, clearly marked non-promotable;
6. run two candidate-vs-incumbent duplicate seeds twice;
7. require identical bids, plays, margins, records, and checkpoint predictions across reruns.

Smoke artifacts live under `/root/residual-bidder/runs/smoke` and can never be admitted to production manifests.

- [ ] **Step 4: Run invariants over the smoke artifacts**

Assert programmatically:

- every final bid is within one of NSFP center;
- both candidate seats use one composite policy ID;
- both opponent seats use one league member ID;
- belief checkpoint hash remains NSFP;
- no deal exceeds 10 games;
- every target equals forced margin minus center margin;
- one action/deal has exactly one formal rollout;
- all deal children share a partition;
- training/evaluation contain no NaN/Inf or runtime fallbacks.

- [ ] **Step 5: Review throughput before choosing the first production duration**

Use the 8-deal measured wall time to report unique deals/day, full games/day, expected storage/10,000 deals, CPU saturation, and GPU fitting time. Keep the semantic production block at 256 deals, but use these measurements to estimate wall-clock duration; do not guess from the 5090 alone.

- [ ] **Step 6: Request final code review and address findings**

Review specifically for hidden-hand leakage, center/incumbent confusion, duplicate score double-counting, branch RNG reuse, nested multiprocessing, belief-bidder replacement, data leakage between partitions, non-atomic resume, and statistically invalid room-level bootstraps.

- [ ] **Step 7: Commit any review-only fixes, then stop at the launch gate**

Do not start the first 256-deal production block until the user has seen the smoke throughput/invariant report and explicitly authorizes the cloud training run.
