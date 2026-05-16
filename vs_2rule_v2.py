#!/usr/bin/env python3
"""vs_2rule_v2.py: two humans vs two RuleBasedPlayer V2 AI.

The two RuleBasedPlayer V2 AIs sit on the same team (either seats 0&2 or
seats 1&3).  They cannot see each other's hand and decide independently.
The program only knows the AI players' hands; human hands are unknown.

Interaction format:
    Line 1:   "02" (seats 0&2 = AI) or "13" (seats 1&3 = AI)
    Line 2:   13 cards for AI seat A (space-separated)
    Line 3:   13 cards for AI seat B (space-separated)

    Bidding phase (seat order 0->1->2->3):
      - AI seats: program computes and outputs the bid
      - Human seats: you type the bid

    Playing phase (each trick):
      - AI seats: program computes and outputs the card
      - Human seats: you type the card

Card format (2 chars):
    Suit: C/D/H/S, Rank: 2-9/T=10/J/Q/K/A,  e.g. DA=Ace of Diamonds, CT=10 of Clubs

Bid format:
    number (0=nil, 1-13) or "nil"/"blind_nil"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
COLLAB_ROOT = REPO_ROOT / "Spades_AI_GO-MCTS"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLAB_ROOT))

from trick_taking.card import Card, Suit, Rank, _STANDARD_CARDS as STANDARD_52
from trick_taking.game_state import GameState, Phase, Bid
from trick_taking.games.spades import SpadesRules

from spades_ai.game.card import Card as GoCard, Rank as GoRank, Suit as GoSuit
from spades_ai.game.scoring import BidType as GoBidType
from spades_ai.game.state import Bid as GoBid, GameState as GoGameState, Phase as GoPhase
from spades_ai.game.trick import Trick as GoTrick, TrickCard as GoTrickCard
from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as RuleBasedPlayerV2


# ── Card conversion helpers ──────────────────────────────────────────────

def to_go_card(card: Card) -> GoCard:
    return GoCard(GoRank(card.rank.value), GoSuit[card.suit.name])


def to_local_card(card: GoCard) -> Card:
    return Card(Suit[card.suit.name], Rank(card.rank.value))


def card_to_str(card: Card) -> str:
    return f"{card.suit.short}{card.rank.short}"


# ── State conversion ─────────────────────────────────────────────────────

def _convert_completed_trick(record_cards: list[tuple[int, Card]]) -> GoTrick:
    go_cards = tuple(GoTrickCard(player=pid, card=to_go_card(c)) for pid, c in record_cards)
    led_suit = go_cards[0].card.suit
    return GoTrick(cards=go_cards, led_suit=led_suit)


def _to_go_bid(raw_bid) -> GoBid | None:
    if raw_bid is None:
        return None
    if isinstance(raw_bid, str):
        if raw_bid == "nil":
            return GoBid(value=0, bid_type=GoBidType.NIL)
        if raw_bid == "blind_nil":
            return GoBid(value=0, bid_type=GoBidType.BLIND_NIL)
        if raw_bid.startswith("bid_"):
            return GoBid(value=int(raw_bid.split("_", 1)[1]), bid_type=GoBidType.NORMAL)
    if isinstance(raw_bid, int):
        return GoBid(value=raw_bid, bid_type=GoBidType.NORMAL)
    return None


def _infer_void_shown(state: GameState):
    voids: list[set[GoSuit]] = [set(), set(), set(), set()]
    def _consume(cards):
        if not cards:
            return
        led_suit = cards[0][1].suit
        for pid, card in cards[1:]:
            if card.suit != led_suit:
                voids[pid].add(GoSuit[led_suit.name])
    for rec in state.trick_history:
        _consume(list(rec.cards))
    if state.table_cards:
        _consume(list(state.table_cards))
    return tuple(frozenset(s) for s in voids)


def to_go_state(state: GameState) -> GoGameState:
    go_hands = tuple(frozenset(to_go_card(c) for c in h) for h in state.hands)
    go_bids = tuple(_to_go_bid(b) for b in state.max_bid)
    completed = tuple(_convert_completed_trick(list(r.cards)) for r in state.trick_history)
    current_trick = tuple(
        GoTrickCard(player=pid, card=to_go_card(c)) for pid, c in state.table_cards
    )
    go_phase = {
        Phase.BIDDING: GoPhase.BIDDING,
        Phase.PLAYING: GoPhase.PLAYING,
        Phase.SCORING: GoPhase.FINISHED,
        Phase.DEALING: GoPhase.BIDDING,
    }[state.phase]
    trick_number = 0 if state.phase == Phase.BIDDING else state.tricks_played + 1
    current_player = state.current_bidder if state.phase == Phase.BIDDING else state.turn
    return GoGameState(
        hands=go_hands,
        bids=go_bids,
        completed_tricks=completed,
        current_trick_cards=current_trick,
        current_player=current_player,
        leader=state.trick_leader,
        trick_number=trick_number,
        tricks_won=tuple(state.tricks_won),
        spades_broken=bool(state.spades_broken or state.trump_broken),
        phase=go_phase,
        void_shown=_infer_void_shown(state),
    )


# ── Bid normalization ────────────────────────────────────────────────────

def normalize_bid_input(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("nil", "blind_nil"):
        return raw
    try:
        n = int(raw)
        return "nil" if n == 0 else f"bid_{n}"
    except ValueError:
        return raw


def go_bid_to_local(go_bid: GoBid) -> str:
    if go_bid.bid_type == GoBidType.NIL:
        return "nil"
    if go_bid.bid_type == GoBidType.BLIND_NIL:
        return "blind_nil"
    return f"bid_{go_bid.value}"


def bid_display(bid_str: str) -> str:
    if bid_str == "nil":
        return "Nil"
    if bid_str == "blind_nil":
        return "Blind Nil"
    if bid_str.startswith("bid_"):
        return bid_str.split("_")[1]
    return bid_str


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    def prompt(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # ── Step 1: AI seat config ──
    prompt("=== Humans vs RuleBasedPlayer V2 ===")
    prompt("AI seats: enter 02 (seats 0&2 = AI) or 13 (seats 1&3 = AI)")
    line = sys.stdin.readline().strip()
    if line == "13":
        ai_seats = [1, 3]
        human_seats = [0, 2]
    else:
        ai_seats = [0, 2]
        human_seats = [1, 3]
    prompt(f"AI (RuleBasedV2): seats {ai_seats[0]} & {ai_seats[1]}")
    prompt(f"Humans:           seats {human_seats[0]} & {human_seats[1]}")

    # ── Step 2: Read AI hands ──
    prompt(f"Enter hand for AI seat {ai_seats[0]} (13 cards, space-separated):")
    hand_a = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]
    prompt(f"Enter hand for AI seat {ai_seats[1]} (13 cards, space-separated):")
    hand_b = [Card.from_str(s) for s in sys.stdin.readline().strip().split()]

    # ── Step 3: Build GameState ──
    # AI hands are known to the program; human hands are filled with
    # remaining cards as placeholders (the AI never peeks at them).
    all_cards = list(STANDARD_52)
    state = GameState()
    state.init_for_deal(4, [[] for _ in range(4)], [], all_cards)
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.BIDDING
    state.current_bidder = 0
    state.turn = 0
    state.trick_leader = 0

    state.hands[ai_seats[0]] = list(hand_a)
    state.hands[ai_seats[1]] = list(hand_b)

    # Fill human seats with remaining cards (placeholder — AI cannot see these)
    known_ids = {c.card_id for c in hand_a} | {c.card_id for c in hand_b}
    remaining = [c for c in all_cards if c.card_id not in known_ids]
    import random
    random.Random().shuffle(remaining)
    state.hands[human_seats[0]] = remaining[:13]
    state.hands[human_seats[1]] = remaining[13:]
    state.hand_bitsets = [
        sum(1 << c.card_id for c in state.hands[p]) for p in range(4)
    ]

    # ── Step 4: AI player instances ──
    ai_players = {seat: RuleBasedPlayerV2() for seat in ai_seats}
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)

    # ── Step 5: Bidding phase ──
    prompt("=== Bidding Phase ===")
    all_bids: dict[int, str] = {}

    for p in range(4):
        if p in ai_seats:
            # AI bids via collaborator engine
            go_state = to_go_state(state)
            go_bid = ai_players[p].choose_bid(go_state)
            bid_str = go_bid_to_local(go_bid)
            all_bids[p] = bid_str
            display = bid_display(bid_str)
            print(display, flush=True)
            prompt(f"AI seat {p} bids {display}")
        else:
            prompt(f"Human seat {p} bid:")
            raw = sys.stdin.readline().strip()
            all_bids[p] = normalize_bid_input(raw)
            prompt(f"-> seat {p} bids {raw}")

        # Update state bids incrementally so next bidder sees previous bids
        state.bids.append(Bid(player_id=p, value=all_bids[p]))
        state.max_bid[p] = all_bids[p]

    # ── Transition to playing phase ──
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    state.turn = 0
    state.trick_leader = 0

    prompt("")
    prompt("Bids summary:")
    for p in range(4):
        tag = "AI" if p in ai_seats else "Human"
        prompt(f"  seat {p} ({tag}): {bid_display(all_bids[p])}")
    prompt("")

    # ── Helper functions ──────────────────────────────────────────────

    def ensure_in_hand(card: Card, pid: int) -> None:
        """Make card present in state.hands[pid] (move from another hand if needed).

        When a human plays a card, it may currently be in a placeholder hand
        of another human seat.  Move it to the correct player.
        """
        if card in state.hands[pid]:
            return
        for q in range(4):
            if q != pid and card in state.hands[q]:
                state.hands[q].remove(card)
                break
        state.hands[pid].append(card)

    def apply_play(card: Card, pid: int) -> None:
        """Play card as pid and update state in-place."""
        ensure_in_hand(card, pid)
        state.play_card_to_table(pid, card)
        if card.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        state.turn = (pid + 1) % 4
        if state.trick_complete:
            winner = rules.winner_trick(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner
            prompt(f"  Trick winner: seat {winner}")

    # ── Step 6: Playing phase ──
    prompt("=== Playing Phase ===")

    for trick_num in range(1, 14):
        if rules.end_trickgame(state):
            break

        leader = state.trick_leader
        prompt(f"--- Trick {trick_num}, leader = seat {leader} ---")

        for pos in range(4):
            cur = (leader + pos) % 4

            if cur in ai_seats:
                go_state = to_go_state(state)
                go_card = ai_players[cur].choose_card(go_state)
                action = to_local_card(go_card)

                # Verify legality against local engine
                legal = rules.playable(state, state.hands[cur], cur)
                if action not in legal:
                    prompt(f"  [WARN] AI seat {cur} returned {card_to_str(action)} not in legal, using fallback")
                    action = legal[0]

                card_str = card_to_str(action)
                print(card_str, flush=True)
                prompt(f"-> AI seat {cur} plays {card_str}")
            else:
                prompt(f"Human seat {cur} play:")
                line = sys.stdin.readline()
                if not line:
                    return
                action = Card.from_str(line.strip())
                prompt(f"-> seat {cur} plays {card_to_str(action)}")

            apply_play(action, cur)

    # ── Step 7: Scoring ──
    prompt("")
    prompt("=== Final Scores ===")
    scores = rules.score(state)
    for p in range(4):
        tag = "AI" if p in ai_seats else "Human"
        prompt(f"  seat {p} ({tag}): tricks={state.tricks_won[p]}, bid={bid_display(all_bids[p])}, score={scores[p]:+.0f}")

    team0 = scores[0]
    team1 = scores[1]
    prompt(f"  Team 0 (seats 0&2): {team0:+.0f}")
    prompt(f"  Team 1 (seats 1&3): {team1:+.0f}")
    if team0 > team1:
        prompt("  Team 0 wins!")
    elif team1 > team0:
        prompt("  Team 1 wins!")
    else:
        prompt("  Draw!")


if __name__ == "__main__":
    main()
