"""Production play: solver-leaf MLP first four, exact solver thereafter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rl.first4_observation import (
    FirstFourFeatureEncoderV2,
    build_first_four_observation,
)
from rl.nil_first4_observation import (
    NilFirstFourFeatureEncoderV1,
    build_nil_first_four_observation,
)
from rl.nil_solver_leaf_env import (
    NIL_LOWER,
    NIL_PARTNER,
    NIL_ROLES,
    NIL_SELF,
    NIL_UPPER,
)
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import legal_mask_from_ids, mask_policy_logits
from strategy.rule_based_first4_player import _trick_current_winner
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from trick_taking.card import Card, Suit, cards_to_bitset
from trick_taking.game_state import Bid, GameState, Phase, TrickRecord
from trick_taking.player import AIPlayer


@dataclass(frozen=True, slots=True)
class _MLPReplayContext:
    nil_seats: tuple[int, ...]


class _NoRuleFirstFourPlayer(AIPlayer):
    """No-op callbacks plus a fail-closed guard against rule play."""

    def start_game(
        self,
        position: int,
        hand: list[Card],
        num_players: int,
    ) -> None:
        del position, hand, num_players

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        del legal_cards, state_view
        raise RuntimeError("rule-based first-four play is disabled in production")


def _validated_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must contain 64 hex digits")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must contain 64 hex digits") from error
    return value.lower()


def role_for_nil_configuration(nil_seats: Sequence[int], seat: int) -> str:
    """Select the deployed four-role policy for any one-to-four-Nil deal.

    Every Nil bidder uses ``nil_self``. A non-Nil seat whose partner bid Nil
    uses ``nil_partner``. With two same-team Nil bidders, the two opponents
    use ``nil_lower`` and ``nil_upper`` relative to the lower-numbered Nil
    seat, which makes the otherwise symmetric A/B assignment deterministic.
    The same rules naturally cover three and four Nil bidders.
    """

    normalized = tuple(sorted(set(int(value) for value in nil_seats)))
    if not normalized or any(value not in range(4) for value in normalized):
        raise ValueError("nil_seats must contain one or more seats in [0, 3]")
    if seat not in range(4):
        raise ValueError("seat must be in [0, 3]")
    nil_set = frozenset(normalized)
    if seat in nil_set:
        return NIL_SELF
    if (seat + 2) % 4 in nil_set:
        return NIL_PARTNER

    anchor = normalized[0]
    delta = (seat - anchor) % 4
    if delta == 1:
        return NIL_LOWER
    if delta == 3:
        return NIL_UPPER
    raise ValueError(
        f"cannot assign a Nil role for nil_seats={normalized!r}, seat={seat}"
    )


class SolverLeafMLPExactPlayer(RuleExactFirst4Player):
    """Use deployed MLPs for every first-four decision, then exact play.

    No-Nil deals use the selected non-Nil solver-leaf actor. Deals containing
    Nil use the four role-specific actors, including the deterministic
    multi-Nil mapping implemented by :func:`role_for_nil_configuration`.
    Posterior replay in the inherited exact stage uses the same MLP mapping,
    so the production card-play path contains no rule-policy substitution.
    """

    def __init__(
        self,
        *,
        nonnil_actor: PolicyMLP,
        nonnil_model_id: str,
        nonnil_actor_sha256: str,
        nil_actors: Mapping[str, PolicyMLP],
        nil_model_id: str,
        nil_bundle_sha256: str,
        exact_solver: Any | None = None,
        exact_threshold: int = 36,
        bid_model=None,
        bid_device: str = "cpu",
        hyperparam_config: Any | None = None,
        num_workers: int = 0,
    ) -> None:
        if not isinstance(nonnil_actor, PolicyMLP):
            raise TypeError("nonnil_actor must be a PolicyMLP")
        if set(nil_actors) != set(NIL_ROLES):
            raise ValueError("nil_actors must contain exactly the four Nil roles")
        if any(not isinstance(actor, PolicyMLP) for actor in nil_actors.values()):
            raise TypeError("every Nil actor must be a PolicyMLP")
        if exact_threshold < 36:
            raise ValueError(
                "exact_threshold must be at least 36; otherwise cards after the "
                "MLP boundary would require an unapproved fallback"
            )
        if not isinstance(nonnil_model_id, str) or not nonnil_model_id:
            raise ValueError("nonnil_model_id must be nonempty")
        if not isinstance(nil_model_id, str) or not nil_model_id:
            raise ValueError("nil_model_id must be nonempty")

        super().__init__(
            exact_solver=exact_solver,
            exact_threshold=exact_threshold,
            bid_model=bid_model,
            bid_device=bid_device,
            hyperparam_config=hyperparam_config,
            num_workers=num_workers,
            first4_player=_NoRuleFirstFourPlayer(),
        )
        self._nonnil_actor = nonnil_actor.eval()
        self._nonnil_encoder = FirstFourFeatureEncoderV2()
        self._nonnil_model_id = nonnil_model_id
        self._nonnil_actor_sha256 = _validated_sha256(
            nonnil_actor_sha256,
            label="nonnil_actor_sha256",
        )
        self._nil_actors = {role: actor.eval() for role, actor in nil_actors.items()}
        self._nil_encoder = NilFirstFourFeatureEncoderV1()
        self._nil_model_id = nil_model_id
        self._nil_bundle_sha256 = _validated_sha256(
            nil_bundle_sha256,
            label="nil_bundle_sha256",
        )

    @staticmethod
    def _nil_bid_seats(bid_values: Sequence[Any]) -> tuple[int, ...]:
        return tuple(
            seat
            for seat, value in enumerate(bid_values)
            if value in ("nil", "blind_nil")
        )

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            raise RuntimeError("solver-leaf MLP card play requires the GameState")

        if len(legal_cards) == 1:
            my_remaining = len(state.hands[self.position])
            mode = "last_card_direct" if my_remaining <= 1 else "single_action_direct"
            self.last_play_info = {"mode": mode}
            return legal_cards[0]

        remaining = sum(len(hand) for hand in state.hands)
        if remaining <= self.exact_threshold:
            return super().play_card(legal_cards, state_view)
        return self._choose_mlp_card(
            state,
            legal_cards,
            player_id=self.position,
            record_diagnostics=True,
        )

    def _choose_mlp_card(
        self,
        state: GameState,
        legal_cards: list[Card],
        *,
        player_id: int,
        record_diagnostics: bool,
    ) -> Card:
        if state.turn != player_id:
            raise RuntimeError(
                f"MLP player position {player_id} does not match turn {state.turn}"
            )
        nil_seats = self._nil_bid_seats(state.max_bid)
        if nil_seats:
            observation = build_nil_first_four_observation(
                state,
                player_id,
                legal_cards,
            )
            encoder = self._nil_encoder
            role = role_for_nil_configuration(nil_seats, player_id)
            actor = self._nil_actors[role]
            model_id = self._nil_model_id
            model_sha256 = self._nil_bundle_sha256
            mode = "nil_solver_leaf_mlp_first4"
        else:
            observation = build_first_four_observation(
                state,
                player_id,
                legal_cards,
            )
            encoder = self._nonnil_encoder
            role = "nonnil"
            actor = self._nonnil_actor
            model_id = self._nonnil_model_id
            model_sha256 = self._nonnil_actor_sha256
            mode = "solver_leaf_mlp_first4"

        feature = encoder.encode(observation)
        legal_mask = legal_mask_from_ids(observation.legal_card_ids)
        encoded_legal_mask = feature[
            encoder.LEGAL_START : encoder.LEGAL_START + 52
        ].astype(np.bool_)
        if not np.array_equal(legal_mask, encoded_legal_mask):
            raise RuntimeError("MLP observation and action legal masks disagree")

        try:
            actor_device = next(actor.parameters()).device
        except StopIteration as error:
            raise RuntimeError(f"MLP actor {role!r} has no parameters") from error
        feature_tensor = torch.from_numpy(feature).to(
            device=actor_device,
            dtype=torch.float32,
        )
        mask_tensor = torch.from_numpy(legal_mask).to(device=actor_device)
        with torch.inference_mode():
            logits = actor(feature_tensor)
            if logits.shape != (52,) or not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError(f"MLP actor {role!r} returned invalid logits")
            masked_logits = mask_policy_logits(logits, mask_tensor)
            probabilities = torch.softmax(masked_logits, dim=-1)
            action = int(torch.argmax(masked_logits).item())
            chosen_probability = float(probabilities[action].item())
        if not legal_mask[action] or not np.isfinite(chosen_probability):
            raise RuntimeError(f"MLP actor {role!r} selected an invalid action")
        card = {candidate.card_id: candidate for candidate in legal_cards}.get(action)
        if card is None:
            raise RuntimeError(f"MLP actor {role!r} selected an illegal card")

        if record_diagnostics:
            self.last_play_info = {
                "mode": mode,
                "role": role,
                "nil_seats": list(nil_seats),
                "model_id": model_id,
                "model_sha256": model_sha256,
                "action_card_id": action,
                "chosen_probability": chosen_probability,
            }
        return card

    def _create_first4_replay_player(
        self,
        player_id: int,
        initial_hand: list[Card],
        max_bid: list[str] | None,
    ) -> Any:
        del player_id, initial_hand
        if max_bid is None or len(max_bid) != 4:
            raise RuntimeError("MLP replay requires all four public bids")
        return _MLPReplayContext(nil_seats=self._nil_bid_seats(max_bid))

    def _first4_replay_expected_card(
        self,
        replay_player: Any,
        legal_cards: list[Card],
        state_view: dict[str, Any],
        *,
        player_id: int,
        current_hand: list[Card],
        prior_plays: list[tuple[int, Card]],
        max_bid: list[str] | None,
    ) -> Card:
        del state_view
        if not isinstance(replay_player, _MLPReplayContext):
            raise RuntimeError("production replay received a non-MLP policy")
        if max_bid is None:
            raise RuntimeError("MLP replay requires all four public bids")
        replay_state = self._build_replay_state(
            player_id=player_id,
            current_hand=current_hand,
            prior_plays=prior_plays,
            max_bid=max_bid,
        )
        if replay_player.nil_seats != self._nil_bid_seats(max_bid):
            raise RuntimeError("MLP replay Nil configuration changed unexpectedly")
        return self._choose_mlp_card(
            replay_state,
            legal_cards,
            player_id=player_id,
            record_diagnostics=False,
        )

    def _first4_replay_expected_cards_batch(
        self,
        replay_players: Sequence[Any],
        legal_batch: Sequence[list[Card]],
        state_views: Sequence[dict[str, Any]],
        *,
        player_id: int,
        hands: Sequence[list[Card]],
        prior_plays: Sequence[tuple[int, Card]],
        max_bid: list[str] | None,
    ) -> list[Card | None]:
        """Judge every proposal of one replay step with a single MLP forward.

        Why one forward is enough: both observation builders only touch
        concealed cards through ``hand_card_ids`` / ``legal_card_ids`` (plus
        the 12 hand-derived encoder columns); every other column comes from
        public state that all proposals of a step share (same ``prior_plays``
        and ``max_bid``).  So one template observation supplies every public
        column, and only those three blocks are rewritten per row.  The actor
        is deterministic (argmax over masked logits) and stateless, so batching
        cannot change any verdict.

        Cost: the previous code ran this decision ``len(hands)`` times, once
        per proposal, paying a full Python/torch call overhead per row; this
        runs it once.
        """

        del state_views
        if max_bid is None or len(max_bid) != 4:
            raise RuntimeError("MLP replay requires all four public bids")
        if not hands:
            return []
        if not (len(replay_players) == len(legal_batch) == len(hands)):
            raise RuntimeError("batch replay received mismatched proposal lists")
        for replay_player in replay_players:
            if not isinstance(replay_player, _MLPReplayContext):
                raise RuntimeError("production replay received a non-MLP policy")
        nil_seats = self._nil_bid_seats(max_bid)

        # ── 1. 用第 0 份提案建一个模板观测（只取公开列）──
        template_state = self._build_replay_state(
            player_id=player_id,
            current_hand=list(hands[0]),
            prior_plays=prior_plays,
            max_bid=max_bid,
        )
        if nil_seats:
            template_observation = build_nil_first_four_observation(
                template_state,
                player_id,
                legal_batch[0],
            )
            encoder = self._nil_encoder
            actor = self._nil_actors[
                role_for_nil_configuration(nil_seats, player_id)
            ]
        else:
            template_observation = build_first_four_observation(
                template_state,
                player_id,
                legal_batch[0],
            )
            encoder = self._nonnil_encoder
            actor = self._nonnil_actor

        base = np.array(encoder.encode(template_observation), dtype=np.float32)
        # 抹掉三块与提案相关的列：hand、legal、hand_derived
        base[encoder.HAND_START : encoder.BIDS_START] = 0.0
        base[encoder.HAND_DERIVED_START : encoder.TOTAL_DIM] = 0.0

        # ── 2. 逐提案只写那三块（整批向量化，不做逐行 numpy 调用）──
        rows = len(hands)
        features = np.repeat(base[None, :], rows, axis=0)
        masks = np.zeros((rows, 52), dtype=np.bool_)

        hand_id_rows = [
            [card.card_id for card in hand]
            for hand in hands
        ]
        legal_id_rows = [
            [card.card_id for card in legal]
            for legal in legal_batch
        ]
        hand_widths = np.fromiter(
            (len(row_ids) for row_ids in hand_id_rows),
            dtype=np.int64,
            count=rows,
        )
        legal_widths = np.fromiter(
            (len(row_ids) for row_ids in legal_id_rows),
            dtype=np.int64,
            count=rows,
        )
        # Every proposal of one replay step gives the acting seat the same
        # number of cards (they all started from a 13-card deal and share the
        # observed play sequence), so the hand block reshapes to a rectangle.
        if rows and int(hand_widths.min()) != int(hand_widths.max()):
            raise RuntimeError("replay hands must have a uniform width per step")
        hand_ids = np.fromiter(
            (cid for row_ids in hand_id_rows for cid in row_ids),
            dtype=np.int64,
            count=int(hand_widths.sum()),
        ).reshape(rows, -1)
        legal_ids = np.fromiter(
            (cid for row_ids in legal_id_rows for cid in row_ids),
            dtype=np.int64,
            count=int(legal_widths.sum()),
        )
        legal_row_index = np.repeat(np.arange(rows), legal_widths)

        features[np.repeat(np.arange(rows), hand_widths), hand_ids.ravel()] = 1.0
        features[legal_row_index, encoder.LEGAL_START + legal_ids] = 1.0
        masks[legal_row_index, legal_ids] = True

        suits = hand_ids // 13
        rank_index = hand_ids % 13
        derived = encoder.HAND_DERIVED_START
        suit_slots = [suits == suit for suit in range(4)]
        features[:, derived : derived + 4] = np.stack(
            [slot.sum(axis=1) for slot in suit_slots], axis=1
        ) / 13.0
        features[:, derived + 4 : derived + 8] = np.stack(
            [(slot & (rank_index >= 10)).sum(axis=1) for slot in suit_slots],
            axis=1,
        ) / 3.0
        features[:, derived + 8 : derived + 12] = np.stack(
            [(slot & (rank_index == 12)).any(axis=1) for slot in suit_slots],
            axis=1,
        ).astype(np.float32)

        encoded_masks = features[
            :, encoder.LEGAL_START : encoder.LEGAL_START + 52
        ].astype(np.bool_)
        if not np.array_equal(masks, encoded_masks):
            raise RuntimeError("MLP observation and action legal masks disagree")

        # ── 3. 一次前向，整批出结果 ──
        try:
            actor_device = next(actor.parameters()).device
        except StopIteration as error:
            raise RuntimeError("MLP actor has no parameters") from error
        with torch.inference_mode():
            logits = actor(
                torch.from_numpy(features).to(
                    device=actor_device,
                    dtype=torch.float32,
                )
            )
            if logits.shape != (rows, 52) or not bool(
                torch.isfinite(logits).all().item()
            ):
                raise RuntimeError("MLP actor returned invalid logits")
            masked_logits = mask_policy_logits(
                logits,
                torch.from_numpy(masks).to(device=actor_device),
            )
            actions = torch.argmax(masked_logits, dim=-1).tolist()

        # ── 4. 每行映射回合法牌 ──
        verdicts: list[Card | None] = []
        for row, legal in enumerate(legal_batch):
            action = int(actions[row])
            card = {candidate.card_id: candidate for candidate in legal}.get(action)
            if card is None:
                raise RuntimeError("MLP actor selected an illegal card")
            verdicts.append(card)
        return verdicts

    def _first4_replay_card_played(
        self,
        replay_player: Any,
        player_id: int,
        card: Card,
    ) -> None:
        del player_id, card
        if not isinstance(replay_player, _MLPReplayContext):
            raise RuntimeError("production replay received a non-MLP policy")

    def _handle_first4_replay_error(self, error: Exception) -> None:
        raise RuntimeError("deployed MLP posterior replay failed") from error

    @staticmethod
    def _build_replay_state(
        *,
        player_id: int,
        current_hand: list[Card],
        prior_plays: list[tuple[int, Card]],
        max_bid: list[str],
    ) -> GameState:
        if len(prior_plays) >= 16:
            raise ValueError("MLP replay is only defined for the first four tricks")
        hands: list[list[Card]] = [[] for _ in range(4)]
        hands[player_id] = list(current_hand)
        state = GameState()
        state.init_for_deal(4, hands, [], [])
        state.phase = Phase.PLAYING
        state.max_bid = list(max_bid)
        state.bids = [
            Bid(player_id=seat, value=value, is_pass=False)
            for seat, value in enumerate(max_bid)
        ]
        state.teams = [0, 1, 0, 1]
        state.turn = player_id
        state.current_bidder = player_id
        state.trump_suit = Suit.SPADES

        complete_cards = len(prior_plays) // 4 * 4
        for offset in range(0, complete_cards, 4):
            trick_cards = list(prior_plays[offset : offset + 4])
            winner, _ = _trick_current_winner(trick_cards, Suit.SPADES)
            state.trick_history.append(
                TrickRecord(
                    cards=trick_cards,
                    winner=winner,
                    leader=trick_cards[0][0],
                )
            )
            state.tricks_won[winner] += 1
        state.tricks_played = len(state.trick_history)
        state.table_cards = list(prior_plays[complete_cards:])
        state.trick_leader = (
            state.table_cards[0][0] if state.table_cards else player_id
        )
        state.played_bitset = 0
        for _, card in prior_plays:
            state.played_bitset |= card.bit
        state.trump_broken = any(card.suit == Suit.SPADES for _, card in prior_plays)
        state.spades_broken = state.trump_broken
        state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
        return state


__all__ = ["SolverLeafMLPExactPlayer", "role_for_nil_configuration"]
