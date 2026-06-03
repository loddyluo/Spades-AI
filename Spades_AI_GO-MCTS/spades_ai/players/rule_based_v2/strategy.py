"""Strategic priority assessment for rule-based player decisions."""
from __future__ import annotations

from spades_ai.game.scoring import BidType
from spades_ai.game.state import GameState


def assess_strategy(state: GameState, player: int) -> list[str]:
    """Return an ordered list of strategic priorities for the given player.

    Possible priorities (in order of consideration):
    - PROTECT_PARTNER_NIL   — partner bid nil and hasn't won a trick yet
    - ATTACK_OPP_NIL        — an opponent bid nil and hasn't been busted
    - TEAM_SET              — team cannot reach its bid in remaining tricks
    - NEED_TRICKS           — team still needs tricks to make its bid
    - AVOID_OVERTRICKS      — team has met its bid, avoid the immediate
                               -9 penalty for each overtrick (no bags)
    - CAN_SET_OPP           — opponent team is close to being set
    - OPP_ALREADY_SET       — opponent team is already going to be set
    """
    priorities: list[str] = []

    partner = (player + 2) % 4
    opp1, opp2 = (player + 1) % 4, (player + 3) % 4

    # --- Nil protection / attack ----------------------------------------
    if (
        state.bids[partner].bid_type in (BidType.NIL, BidType.BLIND_NIL)
        and state.tricks_won[partner] == 0
    ):
        priorities.append("PROTECT_PARTNER_NIL")

    for opp in [opp1, opp2]:
        if (
            state.bids[opp].bid_type in (BidType.NIL, BidType.BLIND_NIL)
            and state.tricks_won[opp] == 0
        ):
            priorities.append("ATTACK_OPP_NIL")
            break

    # --- Team trick needs -----------------------------------------------
    my_bid = _team_non_nil_bid(state, player)
    my_tricks = state.tricks_won[player] + state.tricks_won[partner]
    needed = my_bid - my_tricks
    remaining = 13 - state.trick_number + 1

    if needed > remaining:
        priorities.append("TEAM_SET")
    elif needed > 0:
        priorities.append("NEED_TRICKS")
    else:
        priorities.append("AVOID_OVERTRICKS")

    # --- Opponent dynamics ----------------------------------------------
    opp_bid = _team_non_nil_bid(state, opp1)
    opp_tricks = state.tricks_won[opp1] + state.tricks_won[opp2]
    opp_needed = opp_bid - opp_tricks

    if opp_needed > remaining:
        priorities.append("OPP_ALREADY_SET")
    elif opp_needed > 0 and opp_needed >= remaining - 1:
        priorities.append("CAN_SET_OPP")

    return priorities


def _team_non_nil_bid(state: GameState, player: int) -> int:
    """Sum of NORMAL bids for the team of *player*."""
    partner = (player + 2) % 4
    return sum(
        state.bids[p].value
        for p in [player, partner]
        if state.bids[p].bid_type == BidType.NORMAL
    )
