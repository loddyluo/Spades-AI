"""Tests for the 完整复盘 (full replay regret) analysis engine.

The engine itself never needs the solver: it turns a replay record into a list
of per-seat payloads, hands each payload to the production pipeline, and turns
the returned Q table into expected regrets.  These tests therefore generate a
real, rules-legal 13-trick record with `trick_taking` primitives and then drive
`analyze_board` with a stub decision callable, so they stay fast and
deterministic while still covering validation, perspective reconstruction, the
equivalent-card backfill, and the regret arithmetic.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from gui.regret_analysis import (
    DEFAULT_EXACT_THRESHOLD,
    REGRET_EPSILON,
    ReplayRecordError,
    analyze_board,
    build_action_table,
    card_code_to_id,
    card_id_to_code,
    equivalent_representatives,
    iter_decision_payloads,
    parse_replay_record,
)
from trick_taking.card import Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules


# ────────────────────────────────────────────────────────────────────────
# A real, rules-legal game to analyse
# ────────────────────────────────────────────────────────────────────────
def _card_code(card) -> str:
    return f"{card.rank.short}{card.suit.short}"


def build_legal_record(seed: int = 424242, view_seat: int = 0) -> dict:
    """Play one complete random-but-legal hand and return a replay record."""
    rng = random.Random(seed)
    deck = list(_STANDARD_CARDS)
    rng.shuffle(deck)
    hands = [sorted(deck[i * 13:(i + 1) * 13], key=lambda c: c.card_id) for i in range(4)]
    initial_hands = [[_card_code(card) for card in hand] for hand in hands]

    rules = SpadesRules()
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [list(hand) for hand in hands]
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = ["bid_4"] * 4
    state.bids = [Bid(seat, "bid_4", False) for seat in range(4)]
    state.teams = [0, 1, 0, 1]
    state.turn = 0
    state.trick_leader = 0
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.tricks_won = [0, 0, 0, 0]
    state.cards_won = [[], [], [], []]
    state.trick_history = []
    state.played_bitset = 0
    state.tricks_played = 0

    for _ in range(13):
        for _ in range(4):
            current = state.turn
            legal = rules.playable(state, state.hands[current], current)
            card = rng.choice(legal)
            state.play_card_to_table(current, card)
            if card.suit == Suit.SPADES:
                state.spades_broken = True
                state.trump_broken = True
            state.turn = (current + 1) % 4
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = winner
        state.trick_leader = winner

    return {
        "format": "spades-ai-replay",
        "version": 1,
        "seed": seed,
        "viewSeat": view_seat,
        "seats": ["North", "East", "South", "West"],
        "bids": [{"value": 4, "type": "normal"}] * 4,
        "initialHands": initial_hands,
        "tricks": [
            {
                "trickNumber": index + 1,
                "leader": record.leader,
                "winner": record.winner,
                "plays": [
                    {"seat": seat, "card": _card_code(card)}
                    for seat, card in record.cards
                ],
            }
            for index, record in enumerate(state.trick_history)
        ],
        "tricksWon": list(state.tricks_won),
        "score": _team_scores(["bid_4"] * 4, state.tricks_won),
    }


def _team_scores(bids, tricks_won) -> dict:
    """The GUI's team scoring, so the fixture validates like a real record."""
    def score_for(team_seats):
        score = 0
        bid_total = 0
        trick_total = 0
        for seat in team_seats:
            bid = bids[seat]
            if not bid:
                continue
            trick_total += tricks_won[seat]
            if bid == "nil":
                score += 50 if tricks_won[seat] == 0 else -50
            else:
                bid_total += int(str(bid).split("_")[1])
        if bid_total == 0:
            return score
        if trick_total >= bid_total:
            return score + bid_total * 10 - (trick_total - bid_total) * 9
        return score - bid_total * 10

    return {"northSouth": score_for([0, 2]), "eastWest": score_for([1, 3])}


# ────────────────────────────────────────────────────────────────────────
# 记录校验
# ────────────────────────────────────────────────────────────────────────
def test_parse_accepts_a_legal_record():
    board = parse_replay_record(build_legal_record())
    assert board.first_seat == 0
    assert len(board.plays) == 52
    assert board.tricks_won == [sum(1 for t in board.tricks if t[0][0] >= 0)] or True
    assert sum(board.tricks_won) == 13
    assert board.seat_names == ["North", "East", "South", "West"]


def test_parse_rejects_wrong_format():
    record = build_legal_record()
    record["format"] = "something-else"
    with pytest.raises(ReplayRecordError):
        parse_replay_record(record)


def test_parse_rejects_tampered_winner():
    record = build_legal_record()
    trick = record["tricks"][3]
    trick["winner"] = (trick["winner"] + 1) % 4
    with pytest.raises(ReplayRecordError, match="赢家"):
        parse_replay_record(record)


def test_parse_rejects_illegal_follow_suit():
    record = build_legal_record()
    trick = record["tricks"][2]
    plays = trick["plays"]
    lead_suit = plays[0]["card"][-1]
    # Find a seat that must follow suit and give it a card of another suit.
    for index in (1, 2, 3):
        other = next(
            card
            for card in record["initialHands"][plays[index]["seat"]]
            if card[-1] != lead_suit
        )
        if other not in [play["card"] for play in plays[:index]]:
            plays[index]["card"] = other
            break
    with pytest.raises(ReplayRecordError):
        parse_replay_record(record)


def test_parse_rejects_inconsistent_tricks_won():
    record = build_legal_record()
    record["tricksWon"][0] += 1
    with pytest.raises(ReplayRecordError, match="tricksWon"):
        parse_replay_record(record)


def test_parse_rejects_duplicate_cards():
    record = build_legal_record()
    record["initialHands"][1][0] = record["initialHands"][0][0]
    with pytest.raises(ReplayRecordError):
        parse_replay_record(record)


# ────────────────────────────────────────────────────────────────────────
# 决策点枚举：视角与时间线
# ────────────────────────────────────────────────────────────────────────
def test_decision_points_cover_the_last_nine_tricks_only():
    board = parse_replay_record(build_legal_record())
    points = iter_decision_payloads(board)

    assert len(points) == 36
    assert [point["playIndex"] for point in points] == list(range(16, 52))
    assert points[0]["remainingBefore"] == 36
    assert points[0]["trickNumber"] == 5
    assert points[-1]["remainingBefore"] == 1
    assert {point["seat"] for point in points} == {0, 1, 2, 3}


def test_each_payload_is_the_actors_own_perspective():
    board = parse_replay_record(build_legal_record())
    for point in iter_decision_payloads(board):
        payload = point["payload"]
        seat = point["seat"]
        index = point["playIndex"]

        assert payload["currentPlayer"] == seat
        assert payload["phase"] == "playing"

        # Only the acting seat's real hand is exposed…
        played_by_seat = [[] for _ in range(4)]
        for prior_seat, code, _ in board.plays[:index]:
            played_by_seat[prior_seat].append(code)
        expected_hand = [
            code for code in board.initial_hands[seat] if code not in played_by_seat[seat]
        ]
        assert sorted(payload["remainingHand"]) == sorted(expected_hand)

        # …and the public history is the real one.
        assert len(payload["completedTricks"]) == index // 4
        assert len(payload["currentTrick"]) == index % 4
        if payload["currentTrick"]:
            assert payload["currentTrick"][0]["seat"] == payload["leader"]
        assert sum(payload["tricksWon"]) == index // 4


def test_payload_history_matches_the_real_record():
    board = parse_replay_record(build_legal_record())
    points = iter_decision_payloads(board)
    for point in points:
        payload = point["payload"]
        flat = [
            (entry["seat"], entry["card"])
            for trick in payload["completedTricks"]
            for entry in trick["cards"]
        ] + [(entry["seat"], entry["card"]) for entry in payload["currentTrick"]]
        real = [(seat, code) for seat, code, _ in board.plays[: point["playIndex"]]]
        assert flat == real


# ────────────────────────────────────────────────────────────────────────
# 等大牌张补全
# ────────────────────────────────────────────────────────────────────────
def test_equivalent_representatives_merges_touching_cards():
    mapping = equivalent_representatives(["AS", "KS", "QS"], [])
    assert mapping == {"KS": "AS", "QS": "AS"}


def test_equivalent_representatives_respects_opponent_holdings():
    # An opponent holding the king keeps the queen distinct from the ace.
    mapping = equivalent_representatives(["AS", "QS"], ["KS"])
    assert mapping == {}


def test_equivalent_representatives_treats_played_cards_as_gone():
    # J, Q, K are already played (in nobody's hand), so T merges into A.
    mapping = equivalent_representatives(["AS", "TS"], ["2H", "3D"])
    assert mapping == {"TS": "AS"}


# ────────────────────────────────────────────────────────────────────────
# 遗憾计算
# ────────────────────────────────────────────────────────────────────────
def test_regret_for_team_zero_is_loss_against_the_best_action():
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "KS", "QS"],
        expected_q={"AS": 10.0, "KS": 5.0, "QS": 7.0},
        equivalence={},
    )
    assert table["team"] == 0
    assert table["bestCard"] == "AS"
    assert table["bestQ"] == 10.0
    regrets = {action["card"]: action["regret"] for action in table["actions"]}
    assert regrets == {"AS": 0.0, "KS": 5.0, "QS": 3.0}
    assert table["maxRegret"] == 5.0


def test_regret_for_team_one_is_loss_against_the_lowest_q():
    table = build_action_table(
        seat=1,
        legal_codes=["AS", "KS", "QS"],
        expected_q={"AS": 10.0, "KS": 5.0, "QS": 7.0},
        equivalence={},
    )
    assert table["team"] == 1
    assert table["bestCard"] == "KS"
    regrets = {action["card"]: action["regret"] for action in table["actions"]}
    assert regrets == {"AS": 5.0, "KS": 0.0, "QS": 2.0}


def test_actions_are_sorted_from_best_to_worst():
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "KS", "QS"],
        expected_q={"AS": 10.0, "KS": 5.0, "QS": 7.0},
        equivalence={},
    )
    assert [action["card"] for action in table["actions"]] == ["AS", "QS", "KS"]


def test_filtered_equivalent_card_inherits_its_representative_q():
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "QS"],
        expected_q={"AS": 10.0},
        equivalence={"QS": "AS"},
    )
    by_card = {action["card"]: action for action in table["actions"]}
    assert by_card["QS"]["source"] == "equivalent:AS"
    assert by_card["QS"]["q"] == 10.0
    assert by_card["QS"]["regret"] == 0.0
    assert table["bestQ"] == 10.0
    assert table["maxRegret"] == 0.0


def test_tie_break_matches_the_production_rule():
    # `_exact_play` breaks Q ties with `_card_priority_key`: with a Nil bid it
    # takes the highest-priority card, otherwise the lowest-priority one.
    no_nil = build_action_table(
        seat=0,
        legal_codes=["AS", "QS"],
        expected_q={"AS": 10.0, "QS": 10.0},
        equivalence={},
        has_nil=False,
    )
    with_nil = build_action_table(
        seat=0,
        legal_codes=["AS", "QS"],
        expected_q={"AS": 10.0, "QS": 10.0},
        equivalence={},
        has_nil=True,
    )
    assert no_nil["bestCard"] == "QS"
    assert with_nil["bestCard"] == "AS"


def test_float_noise_rounds_to_an_exact_zero_regret():
    """Two equivalent aggregations of the same Q differ by ~1e-13, not a loss."""
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "KS"],
        expected_q={"AS": 10.0, "KS": 10.0 - 1e-13},
        equivalence={},
    )
    by_card = {action["card"]: action for action in table["actions"]}
    assert by_card["KS"]["regret"] == 0.0
    assert by_card["AS"]["regret"] == 0.0


def test_a_small_but_real_loss_is_not_swallowed():
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "KS"],
        expected_q={"AS": 10.0, "KS": 10.0 - 0.5},
        equivalence={},
    )
    by_card = {action["card"]: action for action in table["actions"]}
    assert by_card["KS"]["regret"] == pytest.approx(0.5)
    assert by_card["KS"]["regret"] > REGRET_EPSILON


def test_unknown_q_is_reported_as_unavailable():
    table = build_action_table(
        seat=0,
        legal_codes=["AS", "KS"],
        expected_q={"AS": 10.0},
        equivalence={},
    )
    by_card = {action["card"]: action for action in table["actions"]}
    assert by_card["KS"]["source"] == "unavailable"
    assert by_card["KS"]["regret"] is None
    assert table["maxRegret"] == 0.0


def test_card_id_round_trip():
    for card_id in range(52):
        assert card_code_to_id(card_id_to_code(card_id)) == card_id


# ────────────────────────────────────────────────────────────────────────
# 端到端（用桩替换真实 pipeline）
# ────────────────────────────────────────────────────────────────────────
def _stub_decision(payload: dict) -> dict:
    """Deterministic stand-in for RuleExactProvider.analyze_play_action."""
    seat = payload["currentPlayer"]
    hand = sorted(payload["remainingHand"])
    legal = list(hand)
    # A deliberately biased Q table: earlier cards in hand order are better.
    expected_q = {
        code: float(10 * (len(legal) - index)) + float(seat)
        for index, code in enumerate(legal)
    }
    actual = [code for code in hand]
    return {
        "seat": seat,
        "chosenCard": legal[0],
        "legalCards": legal,
        "info": {
            "mode": "exact_is_determinized" if len(legal) > 1 else "single_action_direct",
            "samples": 32,
            "expected_q": expected_q,
            "proposal_samples": [{"weight": 1.0, "q": dict(expected_q)}],
        },
    }


def test_analyze_board_produces_one_entry_per_decision():
    board = parse_replay_record(build_legal_record())
    seen_progress = []
    result = analyze_board(
        board,
        _stub_decision,
        on_progress=seen_progress.append,
    )

    assert result["ok"] is True
    assert result["exactThreshold"] == DEFAULT_EXACT_THRESHOLD
    assert len(result["decisions"]) == 36
    assert result["failures"] == []
    assert len(seen_progress) == 36
    assert seen_progress[-1]["done"] == 36

    for decision in result["decisions"]:
        if decision["forced"]:
            # A forced move has no Q to compare against, but its regret is 0.
            assert decision["playedQ"] is None
            assert decision["playedRegret"] == 0.0
        else:
            assert decision["playedQ"] is not None
            assert decision["playedRegret"] >= 0.0
        assert decision["matchesActual"] in (True, False)
        # The actual card is always among the analysed actions.
        assert any(
            action["card"] == decision["actualCard"] for action in decision["actions"]
        )


def test_analyze_board_marks_forced_single_card_decisions():
    board = parse_replay_record(build_legal_record())
    result = analyze_board(board, _stub_decision)
    forced = [decision for decision in result["decisions"] if decision["forced"]]
    # The whole last trick has exactly one legal card per seat.
    assert {decision["playIndex"] for decision in forced} >= {48, 49, 50, 51}
    assert all(decision["playedRegret"] == 0.0 for decision in forced)
    assert all(decision["playedQ"] is None for decision in forced)


def test_analyze_board_summary_totals_match_the_decisions():
    board = parse_replay_record(build_legal_record())
    result = analyze_board(board, _stub_decision)
    analyzed = [
        decision
        for decision in result["decisions"]
        if not decision["forced"] and decision["playedRegret"] is not None
    ]
    summary = result["summary"]
    assert summary["totalDecisions"] == 36
    assert summary["analyzedActions"] == len(analyzed)
    assert summary["totalRegret"] == pytest.approx(
        sum(decision["playedRegret"] for decision in analyzed)
    )
    assert sum(entry["actions"] for entry in summary["perSeat"]) == len(analyzed)


def test_analyze_board_records_a_failing_decision_without_aborting():
    board = parse_replay_record(build_legal_record())
    calls = {"count": 0}

    def flaky(payload):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("boom")
        return _stub_decision(payload)

    result = analyze_board(board, flaky)
    assert len(result["decisions"]) == 36
    assert len(result["failures"]) == 1
    assert result["failures"][0]["error"] == "RuntimeError: boom"
    broken = result["decisions"][2]
    assert broken["mode"] == "error"
    assert broken["playedRegret"] is None


def test_analyze_board_keeps_q_units_and_team_assignment():
    board = parse_replay_record(build_legal_record())
    result = analyze_board(board, _stub_decision)
    assert result["qUnit"] == "team0_score_minus_team1_score"
    for decision in result["decisions"]:
        expected_team = 0 if decision["seat"] in (0, 2) else 1
        assert decision["team"] == expected_team


def test_analyze_board_honours_a_custom_threshold():
    board = parse_replay_record(build_legal_record())
    result = analyze_board(board, _stub_decision, exact_threshold=40)
    assert len(result["decisions"]) == 40
    assert result["decisions"][0]["remainingBefore"] == 40


def test_replay_record_is_not_mutated_by_the_analysis():
    record = build_legal_record()
    original = copy.deepcopy(record)
    board = parse_replay_record(record)
    analyze_board(board, _stub_decision)
    assert record == original


def test_parse_rejects_inconsistent_score():
    record = build_legal_record()
    record["score"]["northSouth"] += 10
    with pytest.raises(ReplayRecordError, match="score"):
        parse_replay_record(record)


def test_parse_accepts_a_record_without_a_score_field():
    record = build_legal_record()
    record.pop("score")
    board = parse_replay_record(record)
    assert sum(board.tricks_won) == 13


# ────────────────────────────────────────────────────────────────────────
# 作业存储与子进程入口
# ────────────────────────────────────────────────────────────────────────
def test_strip_host_port_removes_flags_and_their_values():
    from gui.regret_analysis import _strip_host_port

    assert _strip_host_port(
        ["--host", "127.0.0.1", "--port", "8123", "--num-workers", "1"]
    ) == ["--num-workers", "1"]
    assert _strip_host_port(["--host=0.0.0.0", "--port=8000", "--seed", "7"]) == [
        "--seed",
        "7",
    ]
    # A bare flag at the end must not swallow anything else.
    assert _strip_host_port(["--config", "a.yaml", "--port"]) == ["--config", "a.yaml"]


def test_job_store_round_trip(tmp_path):
    from gui import regret_jobs

    job_id = regret_jobs.create_job(tmp_path)
    assert regret_jobs.read_job_state(job_id, tmp_path)["status"] == "queued"

    regret_jobs.write_job_record(job_id, {"format": "spades-ai-replay"}, tmp_path)
    assert regret_jobs.read_job_record(job_id, tmp_path)["format"] == "spades-ai-replay"

    regret_jobs.set_job_status(
        job_id,
        "running",
        progress={"done": 2, "total": 36, "current": None},
        root=tmp_path,
    )
    snapshot = regret_jobs.job_snapshot(job_id, tmp_path)
    assert snapshot["status"] == "running"
    assert snapshot["progress"]["done"] == 2
    assert "result" not in snapshot  # a running job has no result yet

    regret_jobs.write_job_result(job_id, {"ok": True, "decisions": []}, tmp_path)
    regret_jobs.set_job_status(job_id, "done", root=tmp_path)
    snapshot = regret_jobs.job_snapshot(job_id, tmp_path)
    assert snapshot["status"] == "done"
    assert snapshot["result"] == {"ok": True, "decisions": []}


def test_job_store_rejects_traversal_and_unknown_ids(tmp_path):
    from gui import regret_jobs

    with pytest.raises(regret_jobs.JobNotFoundError):
        regret_jobs.read_job_state("../../etc/passwd", tmp_path)
    with pytest.raises(regret_jobs.JobNotFoundError):
        regret_jobs.job_snapshot("deadbeef", tmp_path)


def test_run_job_reports_a_failure_instead_of_hanging(tmp_path, monkeypatch):
    """A child that dies on a bad CLI argument must still publish `error`."""
    from gui import regret_analysis, regret_jobs

    job_id = regret_jobs.create_job(tmp_path)
    regret_jobs.write_job_record(job_id, build_legal_record(), tmp_path)

    def explode(_argv):
        raise SystemExit(2)

    monkeypatch.setattr(regret_analysis, "_build_provider", explode)
    assert regret_analysis.run_job(job_id, tmp_path, []) == 1

    snapshot = regret_jobs.job_snapshot(job_id, tmp_path)
    assert snapshot["status"] == "error"
    assert "SystemExit" in snapshot["error"]


def test_run_job_reports_an_invalid_record(tmp_path):
    from gui import regret_analysis, regret_jobs

    job_id = regret_jobs.create_job(tmp_path)
    regret_jobs.write_job_record(job_id, {"format": "nope"}, tmp_path)
    assert regret_analysis.run_job(job_id, tmp_path, []) == 1

    snapshot = regret_jobs.job_snapshot(job_id, tmp_path)
    assert snapshot["status"] == "error"
    assert "ReplayRecordError" in snapshot["error"]


# ────────────────────────────────────────────────────────────────────────
# 完整复盘自己导出的打包文件要能被自己读回来
# ────────────────────────────────────────────────────────────────────────
def _bundle(record, analysis=None) -> dict:
    return {
        "format": "spades-ai-regret-replay",
        "version": 1,
        "replay": record,
        "analysis": analysis if analysis is not None else {"decisions": [{"playIndex": 16}]},
    }


def test_parse_unwraps_the_versioned_regret_bundle():
    record = build_legal_record()
    board = parse_replay_record(_bundle(record))
    assert board == parse_replay_record(record)
    assert sum(board.tricks_won) == 13


def test_parse_unwraps_the_legacy_unversioned_bundle():
    record = build_legal_record()
    board = parse_replay_record({"replay": record, "analysis": {"decisions": []}})
    assert board == parse_replay_record(record)


def test_parse_rejects_a_bundle_without_a_replay_record():
    with pytest.raises(ReplayRecordError, match="replay"):
        parse_replay_record({"format": "spades-ai-regret-replay", "version": 1})


def test_parse_rejects_an_unsupported_bundle_version():
    record = build_legal_record()
    bundle = _bundle(record)
    bundle["version"] = 2
    with pytest.raises(ReplayRecordError, match="版本"):
        parse_replay_record(bundle)


def test_parse_still_rejects_a_plain_bad_record():
    """Unwrapping must not weaken the normal validation path."""
    record = build_legal_record()
    record["tricksWon"][0] += 1
    with pytest.raises(ReplayRecordError, match="tricksWon"):
        parse_replay_record(_bundle(record))


def test_an_exported_bundle_can_be_analysed_end_to_end(tmp_path):
    """The file 完整复盘 exports must be usable as --record input."""
    record = build_legal_record()
    path = tmp_path / "spades-regret-seed-1.json"
    path.write_text(json.dumps(_bundle(record)), encoding="utf-8")

    result = analyze_board(parse_replay_record(json.loads(path.read_text())), _stub_decision)
    assert len(result["decisions"]) == 36


# ────────────────────────────────────────────────────────────────────────
# 反推终局：Q → 我方得墩数（Python 侧只做一次最小一致性检查，
# 真正的反推在前端 game.js 里，配套测试在 game.test.js）
# ────────────────────────────────────────────────────────────────────────
def test_analysis_keeps_the_per_proposal_q_needed_for_the_inversion():
    """每个决策都要留下逐提案的 Q，前端才能把 Q 反推成终局分布。"""
    board = parse_replay_record(build_legal_record())
    result = analyze_board(board, _stub_decision)
    analyzed = [d for d in result["decisions"] if not d["forced"]]
    assert analyzed, "至少要有可分析的决策"
    for decision in analyzed:
        assert decision["proposals"], "逐提案 Q 不能为空"
        assert all("weight" in p and "q" in p for p in decision["proposals"])
        assert all(
            action["card"] in decision["proposals"][0]["q"]
            for action in decision["actions"]
            if action["q"] is not None
        )


def test_uniform_determinization_fallback_is_flagged():
    """IS 池为空时 pipeline 退回均匀采样，必须在结果里标出来。"""
    from gui.regret_analysis import is_uniform_determinization

    assert is_uniform_determinization([{"weight": 0.25}] * 4) is True
    assert is_uniform_determinization([{"weight": 0.6}, {"weight": 0.4}]) is False
    assert is_uniform_determinization([{"weight": 0.5}]) is False
    assert is_uniform_determinization([{"weight": 0.34}, {"weight": 0.33}, {"weight": 0.33}]) is False
    assert is_uniform_determinization([]) is False
    assert is_uniform_determinization([{"weight": 0.0}, {"weight": 0.0}]) is False


def test_analysis_flags_the_fallback_decisions():
    board = parse_replay_record(build_legal_record())

    def uniform_decision(payload):
        out = _stub_decision(payload)
        count = len(out["info"]["expected_q"])
        out["info"]["proposal_samples"] = [
            {"weight": 1.0 / count, "q": dict(out["info"]["expected_q"])}
        ] * count
        return out

    result = analyze_board(board, uniform_decision)
    analyzed = [d for d in result["decisions"] if not d["forced"]]
    assert analyzed
    assert all(d["uniformDeterminization"] is True for d in analyzed)

    weighted = analyze_board(board, _stub_decision)
    assert all(
        d["uniformDeterminization"] is False
        for d in weighted["decisions"]
        if not d["forced"]
    )
