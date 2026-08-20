"""Run partnership Spades matches between the deployed AI and DeepSeek V4 Flash.

The DeepSeek side uses the pass-through protocol documented in
``deepseek+deepseek-v4-flash+_chat_completions.pdf``.  Credentials are read
from environment variables and are never written to match records.

This module exposes :func:`run_team_match` as the Python test interface and a
CLI via ``python -m evaluate.deepseek_team_match``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from residual_bidder.actions import to_local_bid  # noqa: E402
from residual_bidder.deployment import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH as DEFAULT_ACTING_BID_CHECKPOINT,
    DEFAULT_CONFIG_PATH as DEFAULT_RESIDUAL_BIDDER_CONFIG,
    load_deployed_acting_bidder,
)
from strategy.hyperparam_config import HyperparamConfig  # noqa: E402
from strategy.rule_exact_first4_nil_player import (  # noqa: E402
    RuleExactFirst4NilPlayer,
)
from strategy.spades_match_runner import SpadesMatchRunner  # noqa: E402
from trick_taking.card import Card  # noqa: E402
from trick_taking.games.spades import SpadesRules  # noqa: E402
from trick_taking.player import AIPlayer  # noqa: E402
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (  # noqa: E402
    ExactDoubleDummyCppFastestSolver,
)


DEFAULT_ENDPOINT = (
    "http://trpc-gpt-eval.production.polaris:8080/v1/chat/completions"
)
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 328_000
APP_ID_ENV = "DEEPSEEK_APP_ID"
APP_KEY_ENV = "DEEPSEEK_APP_KEY"


class TeamMatchError(RuntimeError):
    """Base error for an aborted team match."""


class DeepSeekAPIError(TeamMatchError):
    """The pass-through endpoint failed or returned a malformed response."""


class DeepSeekHTTPError(DeepSeekAPIError):
    """HTTP failure with a status code safe for retry classification."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


class LLMDecisionError(TeamMatchError):
    """DeepSeek did not return one of the explicitly legal actions."""


class CurrentAIFallbackError(TeamMatchError):
    """The deployed AI attempted to enter a fallback path."""


def card_to_code(card: Card) -> str:
    """Return the portable UI spelling, for example ``AS`` or ``TC``."""

    return f"{card.rank.short}{card.suit.short}"


@dataclass(frozen=True)
class DeepSeekPassThroughConfig:
    """Configuration for the documented DeepSeek pass-through protocol."""

    app_id: str = field(repr=False)
    app_key: str = field(repr=False)
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    gateway_timeout_seconds: int = 60
    request_timeout_seconds: float = 90.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_enabled: bool = True
    max_http_retries: int = 2

    def __post_init__(self) -> None:
        if not self.app_id:
            raise ValueError(f"{APP_ID_ENV} 不能为空")
        if not self.app_key:
            raise ValueError(f"{APP_KEY_ENV} 不能为空")
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("DeepSeek endpoint 必须是有效的 HTTP(S) URL")
        if not self.model:
            raise ValueError("DeepSeek model 不能为空")
        if self.gateway_timeout_seconds <= 0:
            raise ValueError("gateway_timeout_seconds 必须大于 0")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds 必须大于 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.max_http_retries < 0:
            raise ValueError("max_http_retries 不能为负数")

    @classmethod
    def from_env(
        cls,
        *,
        endpoint: str | None = None,
        model: str = DEFAULT_MODEL,
        gateway_timeout_seconds: int = 60,
        request_timeout_seconds: float = 90.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking_enabled: bool = True,
        max_http_retries: int = 2,
    ) -> "DeepSeekPassThroughConfig":
        missing = [
            name
            for name in (APP_ID_ENV, APP_KEY_ENV)
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(
                "缺少 DeepSeek 凭据环境变量: " + ", ".join(missing)
            )
        return cls(
            app_id=os.environ[APP_ID_ENV],
            app_key=os.environ[APP_KEY_ENV],
            endpoint=(
                endpoint
                or os.environ.get("DEEPSEEK_API_URL")
                or DEFAULT_ENDPOINT
            ),
            model=model,
            gateway_timeout_seconds=gateway_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            max_http_retries=max_http_retries,
        )

    def public_description(self) -> dict[str, Any]:
        """Return only non-secret fields suitable for logs and JSON output."""

        return {
            "protocol": "deepseek-pass-through-openai-chat-completions",
            "endpoint": self.endpoint,
            "model": self.model,
            "gateway_timeout_seconds": self.gateway_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_tokens": self.max_tokens,
            "thinking_enabled": self.thinking_enabled,
            "max_http_retries": self.max_http_retries,
            "credential_env": {
                "app_id": APP_ID_ENV,
                "app_key": APP_KEY_ENV,
            },
        }


class JsonTransport(Protocol):
    """Injectable HTTP boundary used by unit tests and the live client."""

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        ...


class UrllibJsonTransport:
    """Minimal standard-library JSON transport with bounded error bodies."""

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds
            ) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw_error = exc.read(4096).decode("utf-8", errors="replace")
            raise DeepSeekHTTPError(
                int(exc.code),
                f"DeepSeek HTTP {exc.code}: {raw_error}",
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeepSeekAPIError(
                f"DeepSeek 网络请求失败: {type(exc).__name__}"
            ) from None

        if not 200 <= status < 300:
            raise DeepSeekHTTPError(status, f"DeepSeek HTTP {status}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekAPIError("DeepSeek 响应不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise DeepSeekAPIError("DeepSeek 响应 JSON 顶层必须是对象")
        return payload


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    finish_reason: str | None
    response_id: str | None
    model: str | None
    usage: dict[str, int]


class DeepSeekPassThroughClient:
    """Client for the PDF's APP_ID/APP_KEY pass-through protocol."""

    _RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        config: DeepSeekPassThroughConfig,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibJsonTransport()

    def _authorization(self) -> str:
        query = urllib.parse.urlencode(
            [
                ("provider", "deepseek"),
                ("model", self.config.model),
                ("timeout", str(self.config.gateway_timeout_seconds)),
            ]
        )
        return (
            f"Bearer {self.config.app_id}:{self.config.app_key}?{query}"
        )

    def _redact(self, text: str) -> str:
        redacted = str(text)
        for secret in (self.config.app_id, self.config.app_key):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    @staticmethod
    def _safe_usage(raw_usage: Any) -> dict[str, int]:
        if not isinstance(raw_usage, dict):
            return {}
        result: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and value >= 0:
                result[key] = value
        prompt_details = raw_usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = prompt_details.get("cached_tokens")
            if isinstance(cached, int) and cached >= 0:
                result["cached_tokens"] = cached
        completion_details = raw_usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens")
            if isinstance(reasoning, int) and reasoning >= 0:
                result["reasoning_tokens"] = reasoning
        return result

    def complete(self, messages: Sequence[dict[str, str]]) -> ChatCompletion:
        if not messages:
            raise ValueError("messages 不能为空")
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if not self.config.thinking_enabled:
            body["thinking"] = {"type": "disabled"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": self._authorization(),
            # Keep retry timing under this harness's control.
            "x-should-retry": "false",
        }

        payload: dict[str, Any] | None = None
        for attempt in range(self.config.max_http_retries + 1):
            try:
                payload = self._transport.post_json(
                    url=self.config.endpoint,
                    headers=headers,
                    body=body,
                    timeout_seconds=self.config.request_timeout_seconds,
                )
                break
            except DeepSeekHTTPError as exc:
                retryable = exc.status in self._RETRYABLE_STATUS
                if not retryable or attempt >= self.config.max_http_retries:
                    raise DeepSeekHTTPError(
                        exc.status, self._redact(str(exc))
                    ) from None
            except DeepSeekAPIError as exc:
                if attempt >= self.config.max_http_retries:
                    raise DeepSeekAPIError(self._redact(str(exc))) from None
            time.sleep(min(4.0, 0.5 * (2**attempt)))

        if payload is None:  # defensive; loop either breaks or raises
            raise DeepSeekAPIError("DeepSeek 请求未返回结果")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekAPIError("DeepSeek 响应缺少 choices[0]")
        first = choices[0]
        if not isinstance(first, dict):
            raise DeepSeekAPIError("DeepSeek choices[0] 格式错误")
        message = first.get("message")
        if not isinstance(message, dict):
            raise DeepSeekAPIError("DeepSeek 响应缺少 message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekAPIError("DeepSeek message.content 为空")

        return ChatCompletion(
            content=content,
            finish_reason=(
                first.get("finish_reason")
                if isinstance(first.get("finish_reason"), str)
                else None
            ),
            response_id=(
                payload.get("id") if isinstance(payload.get("id"), str) else None
            ),
            model=(
                payload.get("model")
                if isinstance(payload.get("model"), str)
                else None
            ),
            usage=self._safe_usage(payload.get("usage")),
        )


@dataclass
class DecisionAttempt:
    phase: str
    seat: int
    attempt: int
    valid: bool
    action: str | None
    error: str | None
    finish_reason: str | None
    response_id: str | None
    usage: dict[str, int]


class DeepSeekSpadesPlayer(AIPlayer):
    """Imperfect-information Spades player backed by DeepSeek chat completions."""

    _SYSTEM_PROMPT = """你是四人固定搭档制 Spades（黑桃王）牌手。
座位 0/2 是一队，1/3 是一队；黑桃是将牌，未破黑桃前不能首攻黑桃，除非手中只剩黑桃；跟牌时必须跟首攻花色。
计分：团队完成数字定约得 bid*10，超墩每墩净扣 9；未完成定约扣 bid*10；Nil 成功 +50、失败 -50。
你只能依据自己的手牌和公开叫牌/出牌历史决策，不能假设知道其他人的手牌。请与对家合作。
只返回一个 JSON 对象，格式严格为 {\"action\":\"合法动作之一\"}，不要 Markdown，不要解释。"""

    def __init__(
        self,
        client: DeepSeekPassThroughClient,
        *,
        protocol_attempts: int = 2,
    ) -> None:
        if protocol_attempts <= 0:
            raise ValueError("protocol_attempts 必须大于 0")
        self.client = client
        self.protocol_attempts = protocol_attempts
        self.position = -1
        self.hand: list[Card] = []
        self.teams = [0, 1, 0, 1]
        self.bid_values: list[Any] = [None] * 4
        self.attempt_log: list[DecisionAttempt] = []

    def start_game(
        self, position: int, hand: list[Card], num_players: int
    ) -> None:
        if num_players != 4:
            raise ValueError("DeepSeekSpadesPlayer 只支持四人 Spades")
        self.position = position
        self.hand = list(hand)
        self.teams = [0, 1, 0, 1]
        self.bid_values = [None] * 4
        self.attempt_log = []

    def bid_placed(self, player_id: int, bid_value: Any) -> None:
        if bid_value != "pass" and 0 <= player_id < 4:
            self.bid_values[player_id] = bid_value

    def set_teams(self, teams: list[int], bids: list[Any]) -> None:
        self.teams = list(teams)
        self.bid_values = list(bids)

    def card_played(self, player_id: int, card: Card) -> None:
        if player_id == self.position and card in self.hand:
            self.hand.remove(card)

    @staticmethod
    def _parse_json_action(content: str) -> str:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start < 0:
                raise ValueError("回答中没有 JSON 对象") from None
            try:
                payload, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                raise ValueError("回答中的 JSON 无法解析") from None
        if not isinstance(payload, dict):
            raise ValueError("JSON 顶层不是对象")
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("JSON 缺少非空字符串 action")
        return action.strip()

    def _public_snapshot(self, state_view: dict[str, Any]) -> dict[str, Any]:
        """Serialize only own-hand and public information, never opponent hands."""

        state = state_view.get("state")
        completed_tricks: list[dict[str, Any]] = []
        if state is not None:
            for index, trick in enumerate(getattr(state, "trick_history", [])):
                completed_tricks.append(
                    {
                        "index": index,
                        "leader": int(trick.leader),
                        "winner": int(trick.winner),
                        "cards": [
                            {"seat": int(seat), "card": card_to_code(card)}
                            for seat, card in trick.cards
                        ],
                    }
                )

        bids = []
        for bidder, value, is_pass in state_view.get("bids", []):
            bids.append(
                {
                    "seat": int(bidder),
                    "action": str(value),
                    "pass": bool(is_pass),
                }
            )

        table_cards = [
            {"seat": int(seat), "card": card_to_code(card)}
            for seat, card in state_view.get("table_cards", [])
        ]
        own_hand = sorted(
            (card_to_code(card) for card in state_view.get("hand", [])),
            key=str,
        )
        team_id = (
            self.teams[self.position]
            if 0 <= self.position < len(self.teams)
            else self.position % 2
        )
        team_seats = [
            seat for seat, value in enumerate(self.teams) if value == team_id
        ]

        return {
            "seat": self.position,
            "partner_seat": (self.position + 2) % 4,
            "team_seats": team_seats,
            "dealer_seat": int(state_view.get("dealer_seat", -1)),
            "own_hand": own_hand,
            "hand_sizes": [int(x) for x in state_view.get("hand_size", [])],
            "bids": bids,
            "tricks_won_by_seat": [
                int(x) for x in state_view.get("tricks_won", [])
            ],
            "completed_tricks": completed_tricks,
            "current_trick": table_cards,
            "current_trick_leader": int(
                state_view.get("trick_leader", self.position)
            ),
            "spades_broken": bool(state_view.get("trump_broken", False)),
        }

    def _choose_action(
        self,
        *,
        phase: str,
        legal_actions: list[str],
        state_view: dict[str, Any],
    ) -> str:
        if not legal_actions:
            raise LLMDecisionError(f"座位 {self.position} 没有合法 {phase} 动作")

        request = {
            "task": "choose_bid" if phase == "bid" else "choose_card",
            "public_state": self._public_snapshot(state_view),
            "legal_actions": legal_actions,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    request, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]

        last_error = "未知协议错误"
        for attempt in range(1, self.protocol_attempts + 1):
            completion = self.client.complete(messages)
            action: str | None = None
            try:
                action = self._parse_json_action(completion.content)
                normalized = action.lower() if phase == "bid" else action.upper()
                legal_map = {
                    (item.lower() if phase == "bid" else item.upper()): item
                    for item in legal_actions
                }
                if normalized not in legal_map:
                    raise ValueError(
                        f"action 不在 legal_actions 中: {action!r}"
                    )
                selected = legal_map[normalized]
            except ValueError as exc:
                last_error = str(exc)
                self.attempt_log.append(
                    DecisionAttempt(
                        phase=phase,
                        seat=self.position,
                        attempt=attempt,
                        valid=False,
                        action=action,
                        error=last_error,
                        finish_reason=completion.finish_reason,
                        response_id=completion.response_id,
                        usage=completion.usage,
                    )
                )
                if attempt < self.protocol_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "上次回答不符合接口约束。只返回 "
                                '{"action":"..."}，action 必须逐字取自: '
                                + json.dumps(legal_actions, ensure_ascii=False)
                            ),
                        }
                    )
                continue

            self.attempt_log.append(
                DecisionAttempt(
                    phase=phase,
                    seat=self.position,
                    attempt=attempt,
                    valid=True,
                    action=selected,
                    error=None,
                    finish_reason=completion.finish_reason,
                    response_id=completion.response_id,
                    usage=completion.usage,
                )
            )
            return selected

        raise LLMDecisionError(
            f"DeepSeek 座位 {self.position} 连续 {self.protocol_attempts} 次"
            f"未返回合法 {phase} 动作: {last_error}；牌局已中止"
        )

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        legal = [str(value) for value in legal_bids]
        return self._choose_action(
            phase="bid", legal_actions=legal, state_view=state_view
        )

    def play_card(
        self, legal_cards: list[Card], state_view: dict
    ) -> Card:
        if not legal_cards:
            raise LLMDecisionError(f"DeepSeek 座位 {self.position} 无合法牌")
        if len(legal_cards) == 1:
            card = legal_cards[0]
            self.attempt_log.append(
                DecisionAttempt(
                    phase="play_forced",
                    seat=self.position,
                    attempt=0,
                    valid=True,
                    action=card_to_code(card),
                    error=None,
                    finish_reason=None,
                    response_id=None,
                    usage={},
                )
            )
            return card

        codes = [card_to_code(card) for card in legal_cards]
        selected = self._choose_action(
            phase="play", legal_actions=codes, state_view=state_view
        )
        by_code = {card_to_code(card): card for card in legal_cards}
        return by_code[selected]


@dataclass
class CurrentAIDecision:
    phase: str
    seat: int
    action: str
    detail: str


class CurrentSpadesAIPlayer(RuleExactFirst4NilPlayer):
    """Current production card player plus the deployed residual-Q bidder."""

    def __init__(
        self,
        *,
        acting_bidder: Any,
        exact_solver: Any,
        exact_threshold: int,
        hyperparam_config: HyperparamConfig,
        num_workers: int,
        deal_id: str,
        room_id: str,
    ) -> None:
        super().__init__(
            exact_solver=exact_solver,
            exact_threshold=exact_threshold,
            bid_model=None,
            bid_device="cpu",
            hyperparam_config=hyperparam_config,
            num_workers=num_workers,
        )
        self._acting_bidder = acting_bidder
        self._deal_id = deal_id
        self._room_id = room_id
        self.decision_log: list[CurrentAIDecision] = []

    def start_game(
        self, position: int, hand: list[Card], num_players: int
    ) -> None:
        super().start_game(position, hand, num_players)
        self.decision_log = []

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        # Matches gui/game_server.py: blind-nil is not part of the production
        # flat bidding flow.  Keep this branch explicit for defensive reuse.
        if legal_bids == ["blind_nil", "pass"]:
            self.decision_log.append(
                CurrentAIDecision(
                    phase="bid", seat=self.position, action="pass",
                    detail="blind_nil_auto_pass",
                )
            )
            return "pass"

        state = state_view.get("state")
        if state is None:
            raise CurrentAIFallbackError(
                f"当前 AI 座位 {self.position} 叫牌缺少 GameState"
            )
        decision = self._acting_bidder.choose(
            state,
            legal_bids,
            logical_seat=self.position,
            deal_id=self._deal_id,
            room_id=self._room_id,
        )
        action = to_local_bid(decision.action)
        fallback_reason = getattr(decision, "fallback_reason", None)
        if fallback_reason is not None:
            raise CurrentAIFallbackError(
                f"当前 AI 座位 {self.position} 叫牌触发 fallback: "
                f"{fallback_reason}；牌局已中止"
            )
        if action not in legal_bids:
            raise CurrentAIFallbackError(
                f"当前 AI 座位 {self.position} 返回非法叫牌 {action!r}；"
                "牌局已中止"
            )
        self.last_bid_info = {
            "chosen_bid": action,
            "policy_id": getattr(decision, "effective_policy_id", None),
            "fallback_reason": None,
            "legal_bids": list(legal_bids),
        }
        self.decision_log.append(
            CurrentAIDecision(
                phase="bid",
                seat=self.position,
                action=action,
                detail="residual_q_100k",
            )
        )
        return action

    def play_card(
        self, legal_cards: list[Card], state_view: dict
    ) -> Card:
        card = super().play_card(legal_cards, state_view)
        info = self.last_play_info if isinstance(self.last_play_info, dict) else {}
        mode = str(info.get("mode", ""))
        fallback_reason = info.get("fallback_reason")
        if fallback_reason is not None or "fallback" in mode.lower():
            reason = fallback_reason or mode or "unknown"
            raise CurrentAIFallbackError(
                f"当前 AI 座位 {self.position} 出牌触发 fallback: {reason}；"
                "牌局已中止"
            )
        if card not in legal_cards:
            raise CurrentAIFallbackError(
                f"当前 AI 座位 {self.position} 返回非法牌 {card_to_code(card)}；"
                "牌局已中止"
            )
        self.decision_log.append(
            CurrentAIDecision(
                phase="play",
                seat=self.position,
                action=card_to_code(card),
                detail=mode,
            )
        )
        return card


@dataclass(frozen=True)
class CurrentAIComponents:
    acting_bidder: Any
    exact_solver: Any
    hyperparam_config: HyperparamConfig
    exact_threshold: int = 36
    num_workers: int = 0


def load_current_ai_components(
    *,
    device: str = "cpu",
    exact_threshold: int = 36,
    config_path: Path = REPO_ROOT / "configs" / "8.yaml",
    acting_bid_checkpoint: Path = DEFAULT_ACTING_BID_CHECKPOINT,
    residual_bidder_config: Path = DEFAULT_RESIDUAL_BIDDER_CONFIG,
    bid_policy_seed: int | None = None,
    num_workers: int = 0,
) -> CurrentAIComponents:
    """Load exactly the acting-bid and card-play components used by the GUI."""

    hyperparams = HyperparamConfig.from_yaml(str(config_path))
    acting_bidder = load_deployed_acting_bidder(
        checkpoint_path=Path(acting_bid_checkpoint),
        config_path=Path(residual_bidder_config),
        repo_root=REPO_ROOT,
        device=device,
        policy_seed=bid_policy_seed,
    )
    exact_solver = ExactDoubleDummyCppFastestSolver()
    return CurrentAIComponents(
        acting_bidder=acting_bidder,
        exact_solver=exact_solver,
        hyperparam_config=hyperparams,
        exact_threshold=int(exact_threshold),
        num_workers=int(num_workers),
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _sum_usage(players: list[AIPlayer]) -> dict[str, int]:
    total: dict[str, int] = {}
    for player in players:
        if not isinstance(player, DeepSeekSpadesPlayer):
            continue
        for attempt in player.attempt_log:
            for key, value in attempt.usage.items():
                total[key] = total.get(key, 0) + int(value)
    return total


def _run_one_game(
    *,
    client: DeepSeekPassThroughClient,
    components: CurrentAIComponents,
    seed: int,
    current_ai_seats: tuple[int, int],
    protocol_attempts: int,
) -> dict[str, Any]:
    if set(current_ai_seats) not in ({0, 2}, {1, 3}):
        raise ValueError("current_ai_seats 必须是搭档座位 (0,2) 或 (1,3)")
    _set_random_seed(seed)

    current_set = set(current_ai_seats)
    deal_id = f"deepseek-team-match:{seed}"
    room_id = (
        f"deepseek-team-match:{seed}:"
        + "-".join(str(seat) for seat in sorted(current_set))
    )
    players: list[AIPlayer] = []
    seat_assignment: dict[str, str] = {}
    for seat in range(4):
        if seat in current_set:
            players.append(
                CurrentSpadesAIPlayer(
                    acting_bidder=components.acting_bidder,
                    exact_solver=components.exact_solver,
                    exact_threshold=components.exact_threshold,
                    hyperparam_config=components.hyperparam_config,
                    num_workers=components.num_workers,
                    deal_id=deal_id,
                    room_id=room_id,
                )
            )
            seat_assignment[str(seat)] = "current_spades_ai"
        else:
            players.append(
                DeepSeekSpadesPlayer(
                    client, protocol_attempts=protocol_attempts
                )
            )
            seat_assignment[str(seat)] = "deepseek-v4-flash"

    # Blind nil is disabled to match the GUI's one-shot production bidding
    # flow. Nil and bids 1..13 remain available to both teams.
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
    )
    initial_hands = {
        str(seat): [card_to_code(card) for card in hand]
        for seat, hand in enumerate(runner.state.hands)
    }
    result = runner.play_game()

    current_reference = current_ai_seats[0]
    deepseek_reference = next(seat for seat in range(4) if seat not in current_set)
    current_payoff = float(result.scores[current_reference])
    deepseek_payoff = float(result.scores[deepseek_reference])
    winner = (
        "current_spades_ai"
        if current_payoff > deepseek_payoff
        else "deepseek-v4-flash"
        if deepseek_payoff > current_payoff
        else "tie"
    )

    llm_attempts: list[dict[str, Any]] = []
    current_ai_decisions: list[dict[str, Any]] = []
    for player in players:
        if isinstance(player, DeepSeekSpadesPlayer):
            llm_attempts.extend(asdict(item) for item in player.attempt_log)
        elif isinstance(player, CurrentSpadesAIPlayer):
            current_ai_decisions.extend(
                asdict(item) for item in player.decision_log
            )

    return {
        "seed": seed,
        "seat_assignment": seat_assignment,
        "initial_hands": initial_hands,
        "bids": list(result.bids),
        "tricks_won": list(result.tricks_won),
        "scores": [float(value) for value in result.scores],
        "current_ai_payoff": current_payoff,
        "deepseek_payoff": deepseek_payoff,
        "winner": winner,
        "tricks": [
            {
                "index": index,
                "leader": int(trick.leader),
                "winner": int(trick.winner),
                "cards": [
                    {"seat": int(seat), "card": card_to_code(card)}
                    for seat, card in trick.cards
                ],
            }
            for index, trick in enumerate(runner.state.trick_history)
        ],
        "plays": [
            {
                "index": index,
                "seat": int(record.player_id),
                "card": card_to_code(record.card),
                "legal_cards": [
                    card_to_code(card) for card in record.legal_cards
                ],
            }
            for index, record in enumerate(runner.records)
        ],
        "deepseek_attempts": llm_attempts,
        "current_ai_decisions": current_ai_decisions,
        "deepseek_usage": _sum_usage(players),
    }


def _describe_acting_bidder(acting_bidder: Any) -> dict[str, Any]:
    describe = getattr(acting_bidder, "describe", None)
    if callable(describe):
        value = describe()
        if isinstance(value, dict):
            return value
    return {"name": type(acting_bidder).__name__}


def run_team_match(
    *,
    client: DeepSeekPassThroughClient,
    components: CurrentAIComponents,
    games: int = 1,
    seed: int = 0,
    current_ai_seats: tuple[int, int] = (0, 2),
    swap_sides: bool = False,
    protocol_attempts: int = 2,
) -> dict[str, Any]:
    """Run the requested games and return a credential-free JSON record.

    ``games`` counts unique deals. ``current_ai_seats`` selects the first
    partnership. With ``swap_sides=True``, each deal is then played again with
    the two AI teams on the opposite partnerships.
    """

    if games <= 0:
        raise ValueError("games 必须大于 0")
    if seed < 0:
        raise ValueError("seed 不能为负数")
    if protocol_attempts <= 0:
        raise ValueError("protocol_attempts 必须大于 0")
    if set(current_ai_seats) not in ({0, 2}, {1, 3}):
        raise ValueError("current_ai_seats 必须是搭档座位 (0,2) 或 (1,3)")

    game_records: list[dict[str, Any]] = []
    for deal_index in range(games):
        deal_seed = seed + deal_index
        opposite_seats = (1, 3) if current_ai_seats == (0, 2) else (0, 2)
        seat_sets = (
            [current_ai_seats, opposite_seats]
            if swap_sides
            else [current_ai_seats]
        )
        for current_ai_seats in seat_sets:
            game_records.append(
                _run_one_game(
                    client=client,
                    components=components,
                    seed=deal_seed,
                    current_ai_seats=current_ai_seats,
                    protocol_attempts=protocol_attempts,
                )
            )

    current_wins = sum(
        game["winner"] == "current_spades_ai" for game in game_records
    )
    deepseek_wins = sum(
        game["winner"] == "deepseek-v4-flash" for game in game_records
    )
    ties = len(game_records) - current_wins - deepseek_wins
    total_usage: dict[str, int] = {}
    for game in game_records:
        for key, value in game["deepseek_usage"].items():
            total_usage[key] = total_usage.get(key, 0) + int(value)
    api_attempts = sum(
        1
        for game in game_records
        for item in game["deepseek_attempts"]
        if item["attempt"] > 0
    )

    return {
        "format": "spades-ai-deepseek-team-match",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "deals": games,
            "seed": seed,
            "swap_sides": swap_sides,
            "played_games": len(game_records),
            "default_current_ai_seats": [0, 2],
            "first_game_current_ai_seats": list(current_ai_seats),
            "deepseek": client.config.public_description(),
            "current_ai": {
                "card_player": "RuleExactFirst4NilPlayer",
                "acting_bidder": _describe_acting_bidder(
                    components.acting_bidder
                ),
                "exact_threshold": components.exact_threshold,
                "num_workers": components.num_workers,
            },
            "protocol_attempts": protocol_attempts,
            "blind_nil_enabled": False,
        },
        "summary": {
            "current_spades_ai_wins": current_wins,
            "deepseek_v4_flash_wins": deepseek_wins,
            "ties": ties,
            "current_ai_mean_payoff": sum(
                game["current_ai_payoff"] for game in game_records
            )
            / len(game_records),
            "deepseek_mean_payoff": sum(
                game["deepseek_payoff"] for game in game_records
            )
            / len(game_records),
            "deepseek_api_attempts": api_attempts,
            "deepseek_usage": total_usage,
        },
        "games": game_records,
    }


def write_match_record(
    report: dict[str, Any], output_path: Path, *, overwrite: bool = False
) -> Path:
    """Atomically write a match record, refusing accidental overwrite."""

    target = Path(output_path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "当前生产 Spades AI（座位 0/2）对 DeepSeek V4 Flash（1/3）"
            "的队式赛测试接口"
        )
    )
    parser.add_argument("--games", type=int, default=1, help="独立牌副数")
    parser.add_argument("--seed", type=int, default=0, help="首副牌随机种子")
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="每副牌同牌换边再赛一次",
    )
    parser.add_argument(
        "--current-ai-seats",
        choices=("0,2", "1,3"),
        default="0,2",
        help="首桌当前 Spades AI 的搭档座位（默认 0,2）",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--gateway-timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 DeepSeek 思考模式（默认开启；用 --no-thinking 关闭）",
    )
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--protocol-attempts", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exact-threshold", type=int, default=36)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "8.yaml"
    )
    parser.add_argument(
        "--acting-bid-checkpoint",
        type=Path,
        default=DEFAULT_ACTING_BID_CHECKPOINT,
    )
    parser.add_argument(
        "--residual-bidder-config",
        type=Path,
        default=DEFAULT_RESIDUAL_BIDDER_CONFIG,
    )
    parser.add_argument("--bid-policy-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="允许覆盖显式输出文件"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    default_name = (
        "deepseek_team_match_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_seed{args.seed}.json"
    )
    output = args.output or (REPO_ROOT / "output" / default_name)

    try:
        config = DeepSeekPassThroughConfig.from_env(
            endpoint=args.endpoint,
            model=args.model,
            gateway_timeout_seconds=args.gateway_timeout,
            request_timeout_seconds=args.request_timeout,
            max_tokens=args.max_tokens,
            thinking_enabled=args.thinking,
            max_http_retries=args.api_retries,
        )
        client = DeepSeekPassThroughClient(config)
        components = load_current_ai_components(
            device=args.device,
            exact_threshold=args.exact_threshold,
            config_path=args.config,
            acting_bid_checkpoint=args.acting_bid_checkpoint,
            residual_bidder_config=args.residual_bidder_config,
            bid_policy_seed=args.bid_policy_seed,
            num_workers=args.num_workers,
        )
        report = run_team_match(
            client=client,
            components=components,
            games=args.games,
            seed=args.seed,
            current_ai_seats=tuple(
                int(seat) for seat in args.current_ai_seats.split(",")
            ),
            swap_sides=args.swap_sides,
            protocol_attempts=args.protocol_attempts,
        )
        written = write_match_record(
            report, output, overwrite=bool(args.overwrite)
        )
    except (TeamMatchError, ValueError, FileExistsError) as exc:
        print(f"队式赛失败: {exc}", file=sys.stderr)
        return 1

    summary = report["summary"]
    print(f"队式赛完成，赛果已写入: {written}")
    print(
        "当前 AI / DeepSeek / 平局: "
        f"{summary['current_spades_ai_wins']} / "
        f"{summary['deepseek_v4_flash_wins']} / {summary['ties']}"
    )
    print(
        "平均 payoff（当前 AI / DeepSeek）: "
        f"{summary['current_ai_mean_payoff']:.2f} / "
        f"{summary['deepseek_mean_payoff']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
