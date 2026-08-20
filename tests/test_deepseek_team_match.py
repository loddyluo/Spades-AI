from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluate.deepseek_duplicate_report import build_duplicate_report
from evaluate.deepseek_team_match import (
    APP_ID_ENV,
    APP_KEY_ENV,
    CurrentAIComponents,
    DeepSeekHTTPError,
    DeepSeekPassThroughClient,
    DeepSeekPassThroughConfig,
    DeepSeekSpadesPlayer,
    LLMDecisionError,
    card_to_code,
    run_team_match,
    write_match_record,
)
from residual_bidder.actions import BidAction
from strategy.hyperparam_config import HyperparamConfig
from strategy.spades_match_runner import build_random_state
from trick_taking.card import Suit
from trick_taking.game_state import Phase
from trick_taking.games.spades import SpadesRules


def _completion(content: str, *, response_id: str = "resp-test") -> dict[str, Any]:
    return {
        "id": response_id,
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


class QueueTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": json.loads(json.dumps(body, ensure_ascii=False)),
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _config(**overrides: Any) -> DeepSeekPassThroughConfig:
    values = {
        "app_id": "test-app-id",
        "app_key": "test-app-key",
        "max_http_retries": 0,
    }
    values.update(overrides)
    return DeepSeekPassThroughConfig(**values)


def test_pass_through_protocol_matches_documented_header_and_body() -> None:
    transport = QueueTransport([_completion('{"action":"bid_1"}')])
    client = DeepSeekPassThroughClient(_config(), transport=transport)

    result = client.complete([{"role": "user", "content": "JSON only"}])

    assert result.content == '{"action":"bid_1"}'
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cached_tokens": 2,
    }
    call = transport.calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["headers"]["Authorization"] == (
        "Bearer test-app-id:test-app-key?"
        "provider=deepseek&model=deepseek-v4-flash&timeout=60"
    )
    assert call["headers"]["x-should-retry"] == "false"
    assert call["body"]["model"] == "deepseek-v4-flash"
    assert call["body"]["max_tokens"] == 328_000
    assert "thinking" not in call["body"]
    assert call["body"]["stream"] is False

    public = json.dumps(client.config.public_description())
    assert "test-app-id" not in public
    assert "test-app-key" not in public


def test_thinking_can_still_be_explicitly_disabled() -> None:
    transport = QueueTransport([_completion('{"action":"bid_1"}')])
    client = DeepSeekPassThroughClient(
        _config(thinking_enabled=False), transport=transport
    )

    client.complete([{"role": "user", "content": "JSON only"}])

    assert transport.calls[0]["body"]["thinking"] == {"type": "disabled"}


def test_api_errors_redact_both_credentials() -> None:
    transport = QueueTransport(
        [
            DeepSeekHTTPError(
                401, "bad test-app-id and test-app-key credentials"
            )
        ]
    )
    client = DeepSeekPassThroughClient(_config(), transport=transport)

    with pytest.raises(DeepSeekHTTPError) as exc_info:
        client.complete([{"role": "user", "content": "hello"}])

    message = str(exc_info.value)
    assert "test-app-id" not in message
    assert "test-app-key" not in message
    assert message.count("[REDACTED]") == 2


def test_llm_prompt_contains_no_opponent_hands() -> None:
    state = build_random_state(8811)
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.trump_suit = Suit.SPADES
    seat = 1
    state.turn = seat
    state.trick_leader = seat
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    legal_cards = rules.playable(state, state.hands[seat], seat)
    assert len(legal_cards) > 1

    selected = card_to_code(legal_cards[0])
    transport = QueueTransport([_completion(json.dumps({"action": selected}))])
    player = DeepSeekSpadesPlayer(
        DeepSeekPassThroughClient(_config(), transport=transport)
    )
    player.start_game(seat, list(state.hands[seat]), 4)
    player.set_teams(state.teams, ["bid_1"] * 4)
    view = state.get_player_view(seat)
    view["state"] = state

    assert player.play_card(legal_cards, view) == legal_cards[0]

    body = transport.calls[0]["body"]
    request = json.loads(body["messages"][1]["content"])
    snapshot = request["public_state"]
    assert snapshot["own_hand"] == sorted(
        card_to_code(card) for card in state.hands[seat]
    )
    assert "hands" not in snapshot
    assert "opponent_hands" not in snapshot
    opponent_codes = {
        card_to_code(card)
        for opponent in (0, 2, 3)
        for card in state.hands[opponent]
    }
    encoded_request = body["messages"][1]["content"]
    assert all(f'"{code}"' not in encoded_request for code in opponent_codes)


def test_invalid_llm_actions_abort_without_substitution() -> None:
    state = build_random_state(9922)
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.trump_suit = Suit.SPADES
    seat = 1
    state.turn = seat
    state.trick_leader = seat
    legal_cards = SpadesRules(enable_blind_nil=False).playable(
        state, state.hands[seat], seat
    )
    assert len(legal_cards) > 1

    transport = QueueTransport(
        [_completion("I choose a card"), _completion('{"action":"ZZ"}')]
    )
    player = DeepSeekSpadesPlayer(
        DeepSeekPassThroughClient(_config(), transport=transport),
        protocol_attempts=2,
    )
    player.start_game(seat, list(state.hands[seat]), 4)
    view = state.get_player_view(seat)
    view["state"] = state

    with pytest.raises(LLMDecisionError, match="牌局已中止"):
        player.play_card(legal_cards, view)

    assert len(transport.calls) == 2
    assert [item.valid for item in player.attempt_log] == [False, False]


class LegalActionTransport:
    """Mock model that obeys the schema without containing game logic."""

    def __init__(self) -> None:
        self.calls = 0

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del url, headers, timeout_seconds
        request = json.loads(body["messages"][1]["content"])
        legal = request["legal_actions"]
        action = "bid_1" if request["task"] == "choose_bid" else legal[0]
        assert action in legal
        self.calls += 1
        return _completion(
            json.dumps({"action": action}), response_id=f"resp-{self.calls}"
        )


class FakeActingBidder:
    def choose(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            action=BidAction.BID_1,
            fallback_reason=None,
            effective_policy_id="test-policy",
        )

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-test-bidder", "model_id": "test-only"}


def test_python_interface_runs_complete_mock_team_match() -> None:
    transport = LegalActionTransport()
    client = DeepSeekPassThroughClient(_config(), transport=transport)
    components = CurrentAIComponents(
        acting_bidder=FakeActingBidder(),
        # exact_threshold=0 keeps this interface test on the rule path.
        exact_solver=object(),
        hyperparam_config=HyperparamConfig.default(),
        exact_threshold=0,
        num_workers=1,
    )

    report = run_team_match(
        client=client,
        components=components,
        games=1,
        seed=1234,
        swap_sides=False,
    )

    assert report["format"] == "spades-ai-deepseek-team-match"
    assert report["version"] == 1
    assert report["configuration"]["played_games"] == 1
    game = report["games"][0]
    assert game["seat_assignment"] == {
        "0": "current_spades_ai",
        "1": "deepseek-v4-flash",
        "2": "current_spades_ai",
        "3": "deepseek-v4-flash",
    }
    assert len(game["plays"]) == 52
    assert len(game["tricks"]) == 13
    assert len(game["bids"]) == 4
    assert transport.calls > 0

    serialized = json.dumps(report)
    assert "test-app-id" not in serialized
    assert "test-app-key" not in serialized


def test_python_interface_can_run_only_the_swapped_partnership() -> None:
    transport = LegalActionTransport()
    client = DeepSeekPassThroughClient(_config(), transport=transport)
    components = CurrentAIComponents(
        acting_bidder=FakeActingBidder(),
        exact_solver=object(),
        hyperparam_config=HyperparamConfig.default(),
        exact_threshold=0,
        num_workers=1,
    )

    report = run_team_match(
        client=client,
        components=components,
        games=1,
        seed=1234,
        current_ai_seats=(1, 3),
    )

    assert report["configuration"]["played_games"] == 1
    assert report["configuration"]["first_game_current_ai_seats"] == [1, 3]
    assert report["games"][0]["seat_assignment"] == {
        "0": "deepseek-v4-flash",
        "1": "current_spades_ai",
        "2": "deepseek-v4-flash",
        "3": "current_spades_ai",
    }


def test_duplicate_report_pairs_same_deal_and_normalizes_both_tables(
    tmp_path: Path,
) -> None:
    transport = LegalActionTransport()
    client = DeepSeekPassThroughClient(_config(), transport=transport)
    components = CurrentAIComponents(
        acting_bidder=FakeActingBidder(),
        exact_solver=object(),
        hyperparam_config=HyperparamConfig.default(),
        exact_threshold=0,
        num_workers=1,
    )
    paired = run_team_match(
        client=client,
        components=components,
        games=1,
        seed=1234,
        swap_sides=True,
    )
    table_a_dir = tmp_path / "table_a"
    table_b_dir = tmp_path / "table_b"
    for game, output in (
        (paired["games"][0], table_a_dir / "seed_1234.json"),
        (paired["games"][1], table_b_dir / "table_b_seed_1234.json"),
    ):
        single = {
            **paired,
            "configuration": {
                **paired["configuration"],
                "swap_sides": False,
                "played_games": 1,
            },
            "games": [game],
        }
        write_match_record(single, output)

    summary, bundle = build_duplicate_report(
        table_a_dir=table_a_dir,
        table_b_dir=table_b_dir,
        seeds=[1234],
    )

    board = summary["boards"][0]
    assert summary["configuration"]["played_tables"] == 2
    assert board["current_ai_duplicate_payoff"] == sum(
        game["current_ai_payoff"] for game in paired["games"]
    )
    assert paired["games"][0]["initial_hands"] == paired["games"][1][
        "initial_hands"
    ]
    assert len(summary["replay_records"]) == 2
    assert len(bundle["records"]) == 2
    assert summary["deepseek_totals"]["api_attempts"] == len(
        paired["games"][0]["deepseek_attempts"]
    ) + len(paired["games"][1]["deepseek_attempts"])
    assert summary["deepseek_totals"]["invalid_responses"] == 0


def test_credentials_are_required_by_name_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(APP_ID_ENV, raising=False)
    monkeypatch.delenv(APP_KEY_ENV, raising=False)

    with pytest.raises(ValueError) as exc_info:
        DeepSeekPassThroughConfig.from_env()

    assert APP_ID_ENV in str(exc_info.value)
    assert APP_KEY_ENV in str(exc_info.value)


def test_match_record_refuses_accidental_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "match.json"
    write_match_record({"ok": True}, output)

    with pytest.raises(FileExistsError):
        write_match_record({"ok": False}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
