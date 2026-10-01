"""「完整复盘模式」的分析引擎：后 9 墩每个动作的期望遗憾。

做什么
------
给定一份 GUI 复盘记录（`spades-ai-replay` v1），对**后 9 墩**（即剩余牌数
≤ 36 的精确求解阶段，`exact_threshold=36`）里的每一次出牌：

1. 按**出手那一家的视角**重建 `GameState`：
   - 该家自己的手牌是真实的；
   - 另外三家按「人手牌数正确、具体牌面未知」填充占位牌；
   - 已完成的墩、当前桌面、叫牌、大小王是否已破、每墩归属全部来自公开历史。
   这一步完全复用 `gui.backend.build_local_state`，也就是人机对战模式里
   前端每次请求 AI 时后端做的事，因此视角转换与线上完全一致。
2. 把它交给**同一条 AI pipeline**（`RuleExactProvider` →
   `SolverLeafMLPExactPlayer` → `RuleExactFirst4Player._exact_play`），
   因此重要性采样提案、bid 似然权重、坏动作惩罚、预算表、求解器 Q 值
   与线上逐字一致。
3. 记下每个合法动作的**期望 Q**（用与选牌完全相同的那批提案和归一化权重
   聚合出来的未裁剪 Q 均值，单位是「0 队分数 − 1 队分数」），并换算成
   **期望遗憾**：
   - 0 队（座位 0/2）：`regret(a) = max_a' E[Q(a')] − E[Q(a)]`
   - 1 队（座位 1/3）：`regret(a) = E[Q(a)] − min_a' E[Q(a')]`
   遗憾 ≥ 0，0 表示这是 AI 眼中的最优动作。

求解器会在根节点做「等大牌张」过滤（同一花色中，与上方邻牌之间所有中间
点数都已打出的那张会被合并掉），所以个别合法牌拿不到 Q。这些牌与它所在
组的代表牌在双明手意义下严格等值，我们用同一套分组规则把代表牌的 Q 补回
去，并在结果里标出 `source: "equivalent:<代表牌>"`。

怎么跑
------
作为子进程（由 `gui/backend.py` 的作业接口拉起）：

    python -m gui.regret_analysis --job-id <hex> [backend 的 CLI 参数...]

手动调试：

    python -m gui.regret_analysis --record replay.json --out result.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

from trick_taking.card import Card, Suit  # noqa: E402

import gui.regret_jobs as jobs  # noqa: E402

CARD_CODE_RE = re.compile(r"^[2-9TJQKA][CDHS]$")

# 完整复盘自己导出的文件是「复盘记录 + 遗憾分析」的打包格式。
REGRET_BUNDLE_FORMAT = "spades-ai-regret-replay"

# 后 9 墩 = 剩余牌数不超过 exact_threshold 的阶段（13 墩 × 4 张 = 52；
# 打完前 4 墩剩 36 张，正好进入精确求解阶段）。
DEFAULT_EXACT_THRESHOLD = 36

# 期望遗憾是两个期望 Q 之差；两者都做过浮点求和，真零值可能带 ~1e-13 的噪声。
# 这个容差把噪声压成精确的 0，避免「AI 自己选的那张牌」显示成伪遗憾。
REGRET_EPSILON = 1e-9

RANK_VALUE_BY_CODE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
SUIT_VALUE_BY_CODE = {"S": 0, "H": 1, "D": 2, "C": 3}


class ReplayRecordError(ValueError):
    """复盘记录不合法（结构、牌张、轮转、赢家或计分不一致）。"""


# ────────────────────────────────────────────────────────────────────────
# 牌码工具
# ────────────────────────────────────────────────────────────────────────
def card_code_to_id(code: str) -> int:
    """把 "AS" 转成求解器使用的 0..51 card_id。"""
    return SUIT_VALUE_BY_CODE[code[-1]] * 13 + RANK_VALUE_BY_CODE[code[:-1]] - 2


def card_id_to_code(card_id: int) -> str:
    """把 0..51 card_id 转回 "AS" 形式。"""
    suit = "SHDC"[card_id // 13]
    rank = "23456789TJQKA"[card_id % 13]
    return f"{rank}{suit}"


def card_code_to_card(code: str) -> Card:
    """把 "AS" 转成 `trick_taking.card.Card`。"""
    from trick_taking.card import Rank

    return Card(Suit.from_short(code[-1]), Rank.from_short(code[:-1]))


def _card_priority_key(card: Card) -> tuple:
    """与 `RuleExactFirst4Player._exact_play` 的并列取舍完全一致。"""
    return (card.suit != Suit.SPADES, -card.rank.value, card.suit.value)


# ────────────────────────────────────────────────────────────────────────
# 复盘记录的解析与校验
# ────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReplayBoard:
    """一份校验通过的复盘记录。"""

    seed: int
    view_seat: int
    seat_names: list[str]
    bids: list[Any]
    initial_hands: list[list[str]]
    tricks: list[list[tuple[int, str]]]
    tricks_won: list[int]
    first_seat: int

    @property
    def plays(self) -> list[tuple[int, str, int]]:
        """展平后的出牌序列：`(seat, code, trick_number)`。"""
        flat: list[tuple[int, str, int]] = []
        for index, trick in enumerate(self.tricks):
            for seat, code in trick:
                flat.append((seat, code, index + 1))
        return flat


def _fail(message: str) -> None:
    raise ReplayRecordError(message)


def _require_card_code(value: Any, context: str) -> str:
    if not isinstance(value, str) or not CARD_CODE_RE.match(value):
        _fail(f"{context} 包含无效牌码 {value!r}")
    return value


def _require_seat(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
        _fail(f"{context} 必须是 0-3 的座位编号")
    return value


def _normalize_bid(raw: Any, context: str) -> Any:
    """把复盘记录里的叫牌统一成 `build_local_state` 认识的本地形式。"""
    if raw == "nil":
        return "nil"
    if isinstance(raw, str):
        match = re.match(r"^bid_(\d+)$", raw)
        if match and 1 <= int(match.group(1)) <= 13:
            return f"bid_{int(match.group(1))}"
        _fail(f"{context} 包含无效叫牌 {raw!r}")
    if isinstance(raw, dict):
        bid_type = str(raw.get("type", "normal")).lower()
        if bid_type in ("nil", "blind_nil"):
            return bid_type
        value = raw.get("value")
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 13:
            return f"bid_{value}"
    _fail(f"{context} 包含无效叫牌 {raw!r}")


def _trick_winner(plays: Sequence[tuple[int, str]]) -> int:
    """按黑桃为王的花色规则算出赢家。"""
    lead_suit = plays[0][1][-1]
    best_seat, best_code = plays[0]
    best_suit, best_rank = best_code[-1], RANK_VALUE_BY_CODE[best_code[:-1]]
    for seat, code in plays[1:]:
        suit, rank = code[-1], RANK_VALUE_BY_CODE[code[:-1]]
        if suit == "S":
            if best_suit != "S" or rank > best_rank:
                best_seat, best_suit, best_rank = seat, suit, rank
        elif suit == lead_suit and best_suit != "S" and rank > best_rank:
            best_seat, best_suit, best_rank = seat, suit, rank
    return best_seat


def team_scores(bids: Sequence[Any], tricks_won: Sequence[int]) -> dict[str, float]:
    """复刻前端 `computeScores` 的队式计分（队 0 = 座位 0/2）。"""

    def score_for(team_seats: Sequence[int]) -> float:
        score = 0.0
        bid_total = 0
        trick_total = 0
        for seat in team_seats:
            bid = bids[seat]
            if not bid:
                continue
            trick_total += tricks_won[seat]
            if bid in ("nil", "blind_nil"):
                score += 50.0 if tricks_won[seat] == 0 else -50.0
            else:
                bid_total += int(str(bid).split("_")[1])
        if bid_total == 0:
            return score
        if trick_total >= bid_total:
            return score + bid_total * 10 - (trick_total - bid_total) * 9
        return score - bid_total * 10

    return {"northSouth": score_for([0, 2]), "eastWest": score_for([1, 3])}


def unwrap_replay_document(document: Any) -> Any:
    """把「完整复盘」导出的打包文件拆回内层的复盘记录。

    同时接受带版本号的新格式和早期版本写出的裸 `{replay, analysis}`，
    这样已经导出过的文件仍然能直接喂给 `--record`。
    """
    if not isinstance(document, dict):
        return document
    if document.get("format") == REGRET_BUNDLE_FORMAT:
        if document.get("version") != 1:
            _fail(f"不支持的完整复盘版本 {document.get('version')!r}")
        inner = document.get("replay")
        if not isinstance(inner, dict):
            _fail("完整复盘文件缺少 replay 记录")
        return inner
    if "format" not in document and isinstance(document.get("replay"), dict):
        return document["replay"]
    return document


def parse_replay_record(record: Any) -> ReplayBoard:
    """校验一份 GUI 复盘记录，返回可用的 `ReplayBoard`。

    输入也可以是「完整复盘」导出的 `{replay, analysis}` 打包文件，这里会自动
    拆出内层记录；本函数只关心对局本身，不校验 analysis。

    校验范围刻意与前端 `parseReplayImport` 对齐：格式/版本、四家 13 张互不
    重复的标准牌、13 墩 × 4 次出牌的轮转顺序、跟牌合法性、赢家、以及
    `tricksWon` 是否自洽。任何不一致都会抛 `ReplayRecordError`，绝不静默修正。
    """
    record = unwrap_replay_document(record)
    if not isinstance(record, dict) or isinstance(record, list):
        _fail("顶层必须是 JSON 对象")
    if record.get("format") != "spades-ai-replay" or record.get("version") != 1:
        _fail(
            "不支持的复盘格式 "
            f"{record.get('format')!r} / 版本 {record.get('version')!r}"
        )

    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail("seed 必须是非负整数")

    view_seat = _require_seat(record.get("viewSeat", 0), "viewSeat")

    raw_names = record.get("seats", ["North", "East", "South", "West"])
    if (
        not isinstance(raw_names, list)
        or len(raw_names) != 4
        or any(not isinstance(name, str) or not name.strip() for name in raw_names)
    ):
        _fail("seats 必须包含四个非空名称")
    seat_names = [name.strip() for name in raw_names]

    raw_bids = record.get("bids")
    if not isinstance(raw_bids, list) or len(raw_bids) != 4:
        _fail("bids 必须包含四个叫牌")
    bids = [_normalize_bid(raw, f"座位 {seat}") for seat, raw in enumerate(raw_bids)]

    raw_hands = record.get("initialHands")
    if not isinstance(raw_hands, list) or len(raw_hands) != 4:
        _fail("initialHands 必须包含四家手牌")
    initial_hands: list[list[str]] = []
    for seat, raw_hand in enumerate(raw_hands):
        if not isinstance(raw_hand, list) or len(raw_hand) != 13:
            _fail(f"座位 {seat} 的初始手牌必须恰好有 13 张")
        initial_hands.append(
            [_require_card_code(code, f"座位 {seat} 第 {i + 1} 张牌")
             for i, code in enumerate(raw_hand)]
        )
    all_codes = [code for hand in initial_hands for code in hand]
    if len(set(all_codes)) != 52:
        _fail("四家初始手牌必须无重复地组成标准 52 张牌")

    raw_tricks = record.get("tricks")
    if not isinstance(raw_tricks, list) or len(raw_tricks) != 13:
        _fail("tricks 必须包含 13 墩")

    remaining = [list(hand) for hand in initial_hands]
    tricks: list[list[tuple[int, str]]] = []
    calculated_tricks_won = [0, 0, 0, 0]
    previous_winner: int | None = None
    spades_broken = False

    for index, raw_trick in enumerate(raw_tricks):
        trick_number = index + 1
        if not isinstance(raw_trick, dict):
            _fail(f"第 {trick_number} 墩必须是对象")
        if raw_trick.get("trickNumber") != trick_number:
            _fail(f"第 {trick_number} 墩的 trickNumber 不连续")
        leader = _require_seat(raw_trick.get("leader"), f"第 {trick_number} 墩 leader")
        if previous_winner is not None and leader != previous_winner:
            _fail(
                f"第 {trick_number} 墩应由上一墩赢家座位 {previous_winner} 首攻"
            )
        raw_plays = raw_trick.get("plays")
        if not isinstance(raw_plays, list) or len(raw_plays) != 4:
            _fail(f"第 {trick_number} 墩必须包含四次出牌")

        trick_plays: list[tuple[int, str]] = []
        for offset, raw_play in enumerate(raw_plays):
            if not isinstance(raw_play, dict):
                _fail(f"第 {trick_number} 墩第 {offset + 1} 次出牌必须是对象")
            expected_seat = (leader + offset) % 4
            seat = _require_seat(
                raw_play.get("seat"), f"第 {trick_number} 墩第 {offset + 1} 次出牌 seat"
            )
            if seat != expected_seat:
                _fail(
                    f"第 {trick_number} 墩第 {offset + 1} 次应由座位 {expected_seat} 出牌"
                )
            code = _require_card_code(
                raw_play.get("card"), f"第 {trick_number} 墩第 {offset + 1} 次出牌"
            )
            if code not in remaining[seat]:
                _fail(f"座位 {seat} 在第 {trick_number} 墩并不持有 {code}")

            # 跟牌合法性：有领出花色必须跟，否则任意；未破黑桃时不能首攻黑桃。
            if offset == 0:
                if not spades_broken:
                    non_spades = [c for c in remaining[seat] if c[-1] != "S"]
                    if non_spades and code[-1] == "S":
                        _fail(f"黑桃未破，座位 {seat} 在第 {trick_number} 墩不能首攻 {code}")
            else:
                lead_suit = trick_plays[0][1][-1]
                followers = [c for c in remaining[seat] if c[-1] == lead_suit]
                if followers and code[-1] != lead_suit:
                    _fail(
                        f"座位 {seat} 在第 {trick_number} 墩必须跟 {lead_suit}，却出了 {code}"
                    )

            remaining[seat].remove(code)
            trick_plays.append((seat, code))
            if code[-1] == "S":
                spades_broken = True

        winner = _trick_winner(trick_plays)
        recorded_winner = _require_seat(
            raw_trick.get("winner"), f"第 {trick_number} 墩 winner"
        )
        if recorded_winner != winner:
            _fail(
                f"第 {trick_number} 墩赢家应为座位 {winner}，记录为 {recorded_winner}"
            )
        calculated_tricks_won[winner] += 1
        previous_winner = winner
        tricks.append(trick_plays)

    if any(hand for hand in remaining):
        _fail("13 墩结束后仍有未打出的牌")

    raw_tricks_won = record.get("tricksWon")
    if (
        not isinstance(raw_tricks_won, list)
        or len(raw_tricks_won) != 4
        or any(
            not isinstance(value, int) or value != calculated_tricks_won[seat]
            for seat, value in enumerate(raw_tricks_won)
        )
    ):
        _fail(
            "tricksWon 与逐墩结果不一致，应为 "
            + ",".join(str(value) for value in calculated_tricks_won)
        )

    raw_score = record.get("score")
    if raw_score is not None:
        expected = team_scores(bids, calculated_tricks_won)
        if (
            not isinstance(raw_score, dict)
            or raw_score.get("northSouth") != expected["northSouth"]
            or raw_score.get("eastWest") != expected["eastWest"]
        ):
            _fail(
                "score 与叫牌/墩数不一致，应为 "
                f"NS={expected['northSouth']:g}, EW={expected['eastWest']:g}"
            )

    return ReplayBoard(
        seed=seed,
        view_seat=view_seat,
        seat_names=seat_names,
        bids=bids,
        initial_hands=initial_hands,
        tricks=tricks,
        tricks_won=calculated_tricks_won,
        first_seat=tricks[0][0][0],
    )


# ────────────────────────────────────────────────────────────────────────
# 决策点枚举：把复盘记录变成一串「当时 AI 看到的 payload」
# ────────────────────────────────────────────────────────────────────────
def iter_decision_payloads(
    board: ReplayBoard,
    exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
) -> list[dict[str, Any]]:
    """枚举后 9 墩的每一个决策点，产出与前端 `buildAiPayload` 同构的 payload。

    每个 payload 都是「轮到 `currentPlayer` 出牌、之前的历史全部公开」那一刻
    的快照，因此喂给 `build_local_state` 后得到的正是那一家当时的视角。
    """
    flat = board.plays
    total_plays = len(flat)
    payloads: list[dict[str, Any]] = []

    played_by_seat: list[list[str]] = [[] for _ in range(4)]
    spades_broken = False
    tricks_won = [0, 0, 0, 0]
    completed_tricks: list[dict[str, Any]] = []
    current_trick: list[dict[str, Any]] = []
    current_leader = board.first_seat

    for index in range(total_plays):
        remaining_before = total_plays - index
        seat, code, trick_number = flat[index]

        if remaining_before <= exact_threshold:
            hands_now = [
                [c for c in board.initial_hands[s] if c not in played_by_seat[s]]
                for s in range(4)
            ]
            payloads.append(
                {
                    "playIndex": index,
                    "trickNumber": trick_number,
                    "posInTrick": index % 4,
                    "seat": seat,
                    "actualCard": code,
                    "remainingBefore": remaining_before,
                    "payload": {
                        "seed": board.seed,
                        "firstSeat": board.first_seat,
                        "phase": "playing",
                        "currentPlayer": seat,
                        "leader": current_leader,
                        "trickNumber": trick_number,
                        "spadesBroken": spades_broken,
                        "humanSeat": seat,
                        "remainingHand": list(hands_now[seat]),
                        "bids": [serialize_bid(bid) for bid in board.bids],
                        "completedTricks": list(completed_tricks),
                        "currentTrick": list(current_trick),
                        "tricksWon": list(tricks_won),
                    },
                }
            )

        # 推进时间线（用真实历史，与上面构造 payload 的顺序无关）。
        played_by_seat[seat].append(code)
        if code[-1] == "S":
            spades_broken = True
        current_trick.append({"seat": seat, "card": code})
        if len(current_trick) == 4:
            winner = _trick_winner([(entry["seat"], entry["card"]) for entry in current_trick])
            tricks_won[winner] += 1
            completed_tricks.append({"cards": list(current_trick)})
            current_trick = []
            current_leader = winner

    return payloads


def serialize_bid(bid: Any) -> Any:
    """把本地叫牌转成前端 payload 里的 `{value,type}` 形式。"""
    if bid is None:
        return None
    if bid == "nil":
        return {"value": 0, "type": "nil"}
    if bid == "blind_nil":
        return {"value": 0, "type": "blind_nil"}
    if isinstance(bid, str) and bid.startswith("bid_"):
        return {"value": int(bid.split("_")[1]), "type": "normal"}
    return None


# ────────────────────────────────────────────────────────────────────────
# 等大牌张：把求解器过滤掉的合法牌补回 Q
# ────────────────────────────────────────────────────────────────────────
def equivalent_representatives(
    hand_codes: Sequence[str],
    other_codes: Iterable[str],
) -> dict[str, str]:
    """复刻 C++ `filter_equivalent` 的分组，返回 `被合并牌 -> 代表牌`。

    求解器在根节点会删掉「与更大的手牌紧邻、且两者之间所有点数都已不在任何
    人手里」的牌。这些牌与代表牌在双明手意义下完全等值，因此可以直接复用
    代表牌的 Q 值。
    """
    suit_names = "SHDC"
    my_ranks = [0, 0, 0, 0]
    other_ranks = [0, 0, 0, 0]
    for code in hand_codes:
        my_ranks[SUIT_VALUE_BY_CODE[code[-1]]] |= 1 << (RANK_VALUE_BY_CODE[code[:-1]] - 2)
    for code in other_codes:
        other_ranks[SUIT_VALUE_BY_CODE[code[-1]]] |= 1 << (RANK_VALUE_BY_CODE[code[:-1]] - 2)

    mapping: dict[str, str] = {}
    for suit_index in range(4):
        mine = my_ranks[suit_index]
        if mine == 0:
            continue
        theirs = other_ranks[suit_index]
        previous = -1
        for bit in range(12, -1, -1):
            if not mine & (1 << bit):
                continue
            if previous < 0:
                previous = bit
                continue
            between = ((1 << previous) - 1) ^ ((1 << (bit + 1)) - 1)
            if theirs & between:
                previous = bit
            else:
                keep = f"{'23456789TJQKA'[previous]}{suit_names[suit_index]}"
                drop = f"{'23456789TJQKA'[bit]}{suit_names[suit_index]}"
                mapping[drop] = keep
    return mapping


# ────────────────────────────────────────────────────────────────────────
# 遗憾计算
# ────────────────────────────────────────────────────────────────────────
def build_action_table(
    *,
    seat: int,
    legal_codes: Sequence[str],
    expected_q: dict[str, float],
    equivalence: dict[str, str],
    has_nil: bool = False,
    preferred_card: str | None = None,
) -> dict[str, Any]:
    """把一家的一次决策换算成动作表 + 期望遗憾。"""
    team = 0 if seat in (0, 2) else 1

    actions: list[dict[str, Any]] = []
    for code in legal_codes:
        if code in expected_q:
            actions.append({"card": code, "q": float(expected_q[code]), "source": "solver"})
            continue
        representative = equivalence.get(code)
        if representative is not None and representative in expected_q:
            actions.append(
                {
                    "card": code,
                    "q": float(expected_q[representative]),
                    "source": f"equivalent:{representative}",
                }
            )
            continue
        actions.append({"card": code, "q": None, "source": "unavailable"})

    scored = [action for action in actions if action["q"] is not None]
    if not scored:
        # 没有任何动作拿到 Q（求解器整批失败）：如实报告，不编造数字。
        for action in actions:
            action["regret"] = None
        return {
            "team": team,
            "bestCard": None,
            "bestQ": None,
            "maxRegret": None,
            "actions": actions,
        }

    if team == 0:
        best_q = max(action["q"] for action in scored)
    else:
        best_q = min(action["q"] for action in scored)

    # 并列时的取舍：优先用 pipeline 自己实际会出的那张（等大牌张硬约束会让
    # 它在若干张同 Q 的牌里挑组内最大的那张），否则退回与 `_exact_play` 完全
    # 相同的规则——有人叫 nil 挑优先级最高的（黑桃 > 大牌），否则挑最低的。
    tied = [action["card"] for action in scored if action["q"] == best_q]
    if preferred_card is not None and preferred_card in tied:
        best_card = preferred_card
    else:
        chooser = min if has_nil else max
        best_card = chooser(tied, key=lambda code: _card_priority_key(card_code_to_card(code)))

    def regret_of(value: float) -> float:
        # 负数只可能来自浮点噪声（best_q 就是同一集合的极值），容差内一律归零。
        return value if value > REGRET_EPSILON else 0.0

    for action in actions:
        if action["q"] is None:
            action["regret"] = None
        elif team == 0:
            action["regret"] = regret_of(best_q - action["q"])
        else:
            action["regret"] = regret_of(action["q"] - best_q)

    regrets = [action["regret"] for action in actions if action["regret"] is not None]
    return {
        "team": team,
        "bestCard": best_card,
        "bestQ": float(best_q),
        "maxRegret": max(regrets) if regrets else None,
        "actions": sorted(
            actions,
            key=lambda action: (
                action["regret"] is None,
                action["regret"] if action["regret"] is not None else 0.0,
            ),
        ),
    }


# ────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────
ProgressCallback = Callable[[dict[str, Any]], None]


def analyze_board(
    board: ReplayBoard,
    analyze_decision: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
    on_progress: ProgressCallback | None = None,
    collect_proposals: bool = True,
) -> dict[str, Any]:
    """对一份校验通过的复盘跑完整分析。

    `analyze_decision` 接受一个 payload 字典，返回
    `{"seat": int, "legalCards": [...], "chosenCard": str, "info": {...}}`，
    由 `RuleExactProvider.analyze_play_action` 提供——也就是线上那条 pipeline。
    """
    decision_points = iter_decision_payloads(board, exact_threshold)
    total = len(decision_points)
    decisions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, point in enumerate(decision_points):
        play_index = point["playIndex"]
        seat = point["seat"]
        actual_card = point["actualCard"]

        record: dict[str, Any] = {
            "playIndex": play_index,
            "trickNumber": point["trickNumber"],
            "posInTrick": point["posInTrick"],
            "seat": seat,
            "remainingBefore": point["remainingBefore"],
            "actualCard": actual_card,
            "forced": False,
            "degraded": False,
            "uniformDeterminization": False,
            "mode": None,
            "samples": None,
            "legalCards": [],
            "chosenCard": None,
            "matchesActual": None,
            "team": 0 if seat in (0, 2) else 1,
            "bestCard": None,
            "bestQ": None,
            "maxRegret": None,
            "playedQ": None,
            "playedRegret": None,
            "actions": [],
            "proposals": [],
        }

        def report() -> None:
            if on_progress is not None:
                on_progress(
                    {"done": position + 1, "total": total, "current": _progress_point(point)}
                )

        try:
            outcome = analyze_decision(point["payload"])
        except Exception as error:  # 单个决策失败不应毁掉整份复盘
            record["mode"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            failures.append(
                {
                    "playIndex": play_index,
                    "seat": seat,
                    "error": record["error"],
                }
            )
            decisions.append(record)
            report()
            continue

        legal_codes = list(outcome.get("legalCards") or [])
        chosen_card = outcome.get("chosenCard")
        info = outcome.get("info") or {}

        record["mode"] = info.get("mode")
        record["samples"] = info.get("samples")
        record["legalCards"] = legal_codes
        record["chosenCard"] = chosen_card
        record["matchesActual"] = chosen_card == actual_card

        if len(legal_codes) <= 1:
            # 唯一合法动作：没有可选项，遗憾必然是 0。
            record["forced"] = True
            record["bestCard"] = chosen_card or (legal_codes[0] if legal_codes else None)
            record["playedRegret"] = 0.0
            record["actions"] = [
                {"card": code, "q": None, "regret": 0.0, "source": "forced"}
                for code in legal_codes
            ]
            decisions.append(record)
            report()
            continue

        if info.get("mode") != "exact_is_determinized":
            # 有多个合法动作却没走成精确求解：遗憾未知，如实标记而不是谎报 0。
            record["degraded"] = True
            record["actions"] = [
                {"card": code, "q": None, "regret": None, "source": "unavailable"}
                for code in legal_codes
            ]
            decisions.append(record)
            report()
            continue

        hands_now = _remaining_hands(board, play_index)
        others = [code for s in range(4) if s != seat for code in hands_now[s]]
        equivalence = equivalent_representatives(hands_now[seat], others)

        table = build_action_table(
            seat=seat,
            legal_codes=legal_codes,
            expected_q=info.get("expected_q") or {},
            equivalence=equivalence,
            has_nil=any(
                isinstance(bid, dict) and bid.get("type") in ("nil", "blind_nil")
                for bid in point["payload"]["bids"]
            ),
            preferred_card=chosen_card,
        )
        record["team"] = table["team"]
        record["bestCard"] = table["bestCard"]
        record["bestQ"] = table["bestQ"]
        record["maxRegret"] = table["maxRegret"]
        record["actions"] = table["actions"]
        for action in table["actions"]:
            if action["card"] == actual_card:
                record["playedQ"] = action.get("q")
                record["playedRegret"] = action.get("regret")
        if collect_proposals:
            samples = info.get("proposal_samples") or []
            record["proposals"] = samples
            # IS 池为空时 pipeline 会退回均匀 determinization；如实标出来，
            # 免得把这些遗憾当成加权信念下的结论。
            record["uniformDeterminization"] = is_uniform_determinization(samples)

        decisions.append(record)
        report()

    return {
        "ok": True,
        "exactThreshold": exact_threshold,
        "qUnit": "team0_score_minus_team1_score",
        "decisions": decisions,
        "failures": failures,
        "summary": summarize(decisions),
    }


def is_uniform_determinization(samples: Sequence[dict[str, Any]]) -> bool:
    """判断这批采样是不是「IS 池为空 → 均匀 determinization」兜底出来的。

    重要性采样的权重正比于每个世界在叫牌似然下的后验，一般各不相等；
    兜底路径下每份采样的权重都恰好是 1/K。IS 池为空意味着这些遗憾数字
    建立在均匀信念而不是加权信念上，值得在界面上说明。
    """
    if len(samples) < 2:
        return False
    first = samples[0].get("weight")
    if not isinstance(first, (int, float)) or isinstance(first, bool) or first <= 0:
        return False
    if any(abs(float(sample.get("weight", 0.0)) - float(first)) > 1e-12 for sample in samples):
        return False
    return abs(float(first) * len(samples) - 1.0) < 1e-9


def _progress_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "playIndex": point["playIndex"],
        "trickNumber": point["trickNumber"],
        "seat": point["seat"],
        "remainingBefore": point["remainingBefore"],
    }


def _remaining_hands(board: ReplayBoard, play_index: int) -> list[list[str]]:
    """全局第 `play_index` 手之前，四家各自还剩什么牌。"""
    played = [set() for _ in range(4)]
    for index, (seat, code, _) in enumerate(board.plays):
        if index >= play_index:
            break
        played[seat].add(code)
    return [
        [code for code in board.initial_hands[seat] if code not in played[seat]]
        for seat in range(4)
    ]


def summarize(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """汇总整份复盘的遗憾，给出最亏的几手。"""
    analyzed = [
        decision
        for decision in decisions
        if not decision.get("forced") and decision.get("playedRegret") is not None
    ]
    total_regret = sum(float(decision["playedRegret"]) for decision in analyzed)
    per_seat: list[dict[str, Any]] = []
    for seat in range(4):
        seat_decisions = [d for d in analyzed if d["seat"] == seat]
        per_seat.append(
            {
                "seat": seat,
                "actions": len(seat_decisions),
                "totalRegret": sum(float(d["playedRegret"]) for d in seat_decisions),
                "worstRegret": max(
                    (float(d["playedRegret"]) for d in seat_decisions), default=0.0
                ),
            }
        )

    worst = sorted(
        analyzed, key=lambda decision: float(decision["playedRegret"]), reverse=True
    )[:8]
    return {
        "analyzedActions": len(analyzed),
        "totalDecisions": len(decisions),
        "totalRegret": total_regret,
        "meanRegret": (total_regret / len(analyzed)) if analyzed else 0.0,
        "perSeat": per_seat,
        "worst": [
            {
                "playIndex": decision["playIndex"],
                "trickNumber": decision["trickNumber"],
                "seat": decision["seat"],
                "actualCard": decision["actualCard"],
                "bestCard": decision["bestCard"],
                "regret": decision["playedRegret"],
            }
            for decision in worst
            if float(decision["playedRegret"]) > 0
        ],
    }


# ────────────────────────────────────────────────────────────────────────
# 子进程入口
# ────────────────────────────────────────────────────────────────────────
def _strip_host_port(argv: Sequence[str]) -> list[str]:
    """Drop `--host`/`--port` (and their values) from the backend argv.

    The GUI backend parses its own CLI, so the analysis child is handed that
    argv verbatim; only the two listen options are meaningless here and must go
    — together with their values, which `argparse` would otherwise read back as
    positional junk.
    """
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in ("--host", "--port"):
            skip_next = True
            continue
        if token.startswith("--host=") or token.startswith("--port="):
            continue
        cleaned.append(token)
    return cleaned


def _build_provider(extra_argv: Sequence[str]):
    """用与 GUI 后端完全相同的 CLI 参数构造生产 pipeline。"""
    from gui import backend

    original = sys.argv
    try:
        sys.argv = ["gui.regret_analysis", *_strip_host_port(extra_argv)]
        args = backend.parse_args()
    finally:
        sys.argv = original
    return backend.RuleExactProvider(args), args


def run_job(job_id: str, job_root: Path | None, extra_argv: Sequence[str]) -> int:
    """执行一个后台分析作业，把进度写进 job.json、结果写进 result.json。"""
    started = time.time()

    def report(progress: dict[str, Any]) -> None:
        jobs.set_job_status(job_id, "running", progress=progress, root=job_root)

    jobs.set_job_status(
        job_id,
        "running",
        progress={"done": 0, "total": 0, "current": None},
        root=job_root,
    )

    try:
        record = jobs.read_job_record(job_id, job_root)
        if record is None:
            raise RuntimeError("作业目录里没有 record.json")

        board = parse_replay_record(record)
        provider, _args = _build_provider(extra_argv)

        expected_total = len(iter_decision_payloads(board, provider.exact_threshold))
        jobs.set_job_status(
            job_id,
            "running",
            progress={"done": 0, "total": expected_total, "current": None},
            root=job_root,
            job_extra={"ai": provider.ai_name},
        )

        result = analyze_board(
            board,
            provider.analyze_play_action,
            exact_threshold=provider.exact_threshold,
            on_progress=report,
        )
        result["seed"] = board.seed
        result["seatNames"] = board.seat_names
        result["ai"] = provider.ai_name
        result["config"] = _describe_provider_config(provider)
        result["startedAt"] = started
        result["finishedAt"] = time.time()
        result["elapsedSeconds"] = time.time() - started

        jobs.write_job_result(job_id, result, job_root)
        jobs.set_job_status(
            job_id,
            "done",
            progress={
                "done": expected_total,
                "total": expected_total,
                "current": None,
            },
            root=job_root,
            job_extra={"ai": provider.ai_name},
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - 必须把失败写回作业文件
        # BaseException, not Exception: a bad CLI argument makes argparse call
        # sys.exit(), and a job that dies on SystemExit must still be reported
        # as failed instead of hanging the GUI on "running" forever.
        traceback.print_exc()
        jobs.set_job_status(
            job_id,
            "error",
            error=f"{type(error).__name__}: {error}",
            root=job_root,
        )
        return 1


def _describe_provider_config(provider: Any) -> dict[str, Any]:
    """Which hyperparameters produced these numbers, with a file hash.

    A regret value only means something next to its config: with
    ``multiplier_clip_factor != 1`` the AI optimises a rescaled objective, so
    the report must say which one was used instead of leaving it to memory.
    """
    return {
        "path": str(getattr(provider, "config_path", "")),
        "sha256": getattr(provider, "config_sha256", None),
        "effective": asdict(provider.hyperparam_config),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析分析引擎自己的参数；其余参数原样转发给 GUI 后端解析。"""
    parser = argparse.ArgumentParser(
        description="完整复盘：后 9 墩每个动作的期望遗憾",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--job-id", default="", help="由 GUI 后端创建的作业 id")
    parser.add_argument("--job-root", default="", help="作业目录根路径")
    parser.add_argument("--record", default="", help="直接指定复盘 JSON（手动调试）")
    parser.add_argument("--out", default="", help="直接指定结果输出路径（手动调试）")
    known, remainder = parser.parse_known_args(argv)
    if known.job_id and known.record:
        parser.error("--job-id 与 --record 只能二选一")
    if not known.job_id and not known.record:
        parser.error("必须提供 --job-id 或 --record")
    known.forwarded = remainder
    return known


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    job_root = Path(args.job_root) if args.job_root else None

    if args.job_id:
        return run_job(args.job_id, job_root, args.forwarded)

    with open(args.record, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    board = parse_replay_record(record)
    provider, _backend_args = _build_provider(args.forwarded)
    result = analyze_board(
        board,
        provider.analyze_play_action,
        exact_threshold=provider.exact_threshold,
        on_progress=lambda progress: print(
            f"  [{progress['done']}/{progress['total']}] "
            f"第 {progress['current']['trickNumber']} 墩 座位 {progress['current']['seat']}",
            flush=True,
        ),
    )
    result["ai"] = provider.ai_name
    result["config"] = _describe_provider_config(provider)
    result["seed"] = board.seed
    result["seatNames"] = board.seat_names
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"结果已写入 {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
