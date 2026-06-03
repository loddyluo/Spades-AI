from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class BidType(Enum):
    NORMAL = auto()
    NIL = auto()
    BLIND_NIL = auto()


@dataclass(frozen=True)
class PlayerResult:
    bid: int
    bid_type: BidType
    tricks_won: int


def compute_team_score(team_results: list[PlayerResult]) -> int:
    """Compute one team's hand score under the project Spades rules.

    Important: this project does NOT use cumulative bag scoring.  Each
    overtrick is penalized immediately by -9 points; there is no "10 bags
    = -100" state carried across hands.
    """
    score = 0
    for pr in team_results:
        if pr.bid_type == BidType.NIL:
            score += 50 if pr.tricks_won == 0 else -50
        elif pr.bid_type == BidType.BLIND_NIL:
            score += 100 if pr.tricks_won == 0 else -100

    non_nil_bid = sum(pr.bid for pr in team_results if pr.bid_type == BidType.NORMAL)
    team_tricks = sum(pr.tricks_won for pr in team_results)

    if non_nil_bid == 0:
        return score

    if team_tricks >= non_nil_bid:
        score += non_nil_bid * 10
        score -= (team_tricks - non_nil_bid) * 9
    else:
        score -= non_nil_bid * 10

    return score


def compute_hand_scores(all_results: list[PlayerResult]) -> tuple[int, int]:
    team_a = [all_results[0], all_results[2]]
    team_b = [all_results[1], all_results[3]]
    return compute_team_score(team_a), compute_team_score(team_b)
