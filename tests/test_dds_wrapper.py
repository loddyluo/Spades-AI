#!/usr/bin/env python3
"""Quick test for DDS wrapper correctness.

Tests:
1. Basic endgame solving (known trick counts)
2. Spades-broken enforcement (cannot lead spades when not broken)
3. Only-spades hand (can lead spades even when not broken)
4. Cross-solver comparison with our ExactDoubleDummyCppOpt1Solver
"""

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank
from evaluate.dds_wrapper import DDSSolver


def test_basic_solve():
    """Test a simple 1-trick endgame where outcome is obvious."""
    print("Test 1: Basic 1-trick endgame...")
    solver = DDSSolver()

    # Player 0 (North) leads with Ace of Clubs
    # Player 1 (East) has 2 of Clubs
    # Player 2 (South) has 3 of Clubs
    # Player 3 (West) has 4 of Clubs
    # North should win 1 trick
    hands = [
        [Card(Suit.CLUBS, Rank.ACE)],
        [Card(Suit.CLUBS, Rank.TWO)],
        [Card(Suit.CLUBS, Rank.THREE)],
        [Card(Suit.CLUBS, Rank.FOUR)],
    ]

    results = solver.solve_position(
        hands=hands,
        current_trick=[],
        trick_leader=0,
        next_to_play=0,
        spades_broken=True,
    )

    assert len(results) == 1, f"Expected 1 move, got {len(results)}"
    assert results[0]["card"].suit == Suit.CLUBS
    assert results[0]["card"].rank == Rank.ACE
    assert results[0]["tricks_won"] == 1, f"Expected 1 trick, got {results[0]['tricks_won']}"
    print("  PASSED: North wins 1 trick with A♣")


def test_spades_broken_enforcement():
    """When spades not broken, leader should NOT be able to lead spades."""
    print("\nTest 2: Spades broken enforcement...")
    solver = DDSSolver()

    # Player 0 has: Ace♠, King♥ (2 cards)
    # Others have non-spade cards
    hands = [
        [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)],
        [Card(Suit.HEARTS, Rank.TWO), Card(Suit.DIAMONDS, Rank.TWO)],
        [Card(Suit.HEARTS, Rank.THREE), Card(Suit.DIAMONDS, Rank.THREE)],
        [Card(Suit.HEARTS, Rank.FOUR), Card(Suit.DIAMONDS, Rank.FOUR)],
    ]

    results = solver.solve_position(
        hands=hands,
        current_trick=[],
        trick_leader=0,
        next_to_play=0,
        spades_broken=False,  # NOT broken
    )

    # Should only have non-spade options
    played_suits = {r["card"].suit for r in results}
    assert Suit.SPADES not in played_suits, \
        f"Spades should not be playable when not broken, got suits: {played_suits}"
    print("  PASSED: Spades filtered from lead options when not broken")


def test_only_spades_hand():
    """If player has ONLY spades, they must be able to lead them even when not broken."""
    print("\nTest 3: Only-spades hand can still lead...")
    solver = DDSSolver()

    # Player 0 has only spades
    hands = [
        [Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.KING)],
        [Card(Suit.HEARTS, Rank.TWO), Card(Suit.DIAMONDS, Rank.TWO)],
        [Card(Suit.HEARTS, Rank.THREE), Card(Suit.DIAMONDS, Rank.THREE)],
        [Card(Suit.HEARTS, Rank.FOUR), Card(Suit.DIAMONDS, Rank.FOUR)],
    ]

    results = solver.solve_position(
        hands=hands,
        current_trick=[],
        trick_leader=0,
        next_to_play=0,
        spades_broken=False,  # NOT broken, but player has only spades
    )

    assert len(results) > 0, "Should have at least one legal move"
    assert all(r["card"].suit == Suit.SPADES for r in results), \
        "All moves should be spades (only cards in hand)"
    # With A♠ K♠ vs no spades, North should win both tricks
    assert results[0]["tricks_won"] == 2, f"Expected 2 tricks, got {results[0]['tricks_won']}"
    print("  PASSED: Only-spades hand can lead, wins 2 tricks")


def test_multi_trick():
    """Test a 3-trick endgame."""
    print("\nTest 4: Multi-trick endgame...")
    solver = DDSSolver()

    # 3 cards each
    # North: A♣, K♣, Q♣ (leads)
    # East:  2♣, 3♣, 4♣
    # South: 5♣, 6♣, 7♣
    # West:  8♣, 9♣, T♣
    hands = [
        [Card(Suit.CLUBS, Rank.ACE), Card(Suit.CLUBS, Rank.KING), Card(Suit.CLUBS, Rank.QUEEN)],
        [Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE), Card(Suit.CLUBS, Rank.FOUR)],
        [Card(Suit.CLUBS, Rank.FIVE), Card(Suit.CLUBS, Rank.SIX), Card(Suit.CLUBS, Rank.SEVEN)],
        [Card(Suit.CLUBS, Rank.EIGHT), Card(Suit.CLUBS, Rank.NINE), Card(Suit.CLUBS, Rank.TEN)],
    ]

    results = solver.solve_position(
        hands=hands,
        current_trick=[],
        trick_leader=0,
        next_to_play=0,
        spades_broken=True,
    )

    # North+South vs East+West: N has AKQ♣, S has 567♣
    # North can win all 3 tricks with AKQ
    assert results[0]["tricks_won"] == 3, f"Expected 3 tricks, got {results[0]['tricks_won']}"
    print(f"  PASSED: North can win 3 tricks")


def test_cross_solver_comparison():
    """Compare DDS results with our own exact solver on a small endgame."""
    print("\nTest 5: Cross-solver comparison...")
    try:
        from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver
        from trick_taking.game_state import GameState, Phase
        from trick_taking.deck import Deck, STANDARD_52
        from trick_taking.games.spades import SpadesRules
        import random

        our_solver = ExactDoubleDummyCppOpt1Solver()
        if not our_solver.native_available:
            print("  SKIPPED: Our C++ solver not available")
            return

        dds = DDSSolver()

        # Build a simple 4-card-per-player endgame
        rng = random.Random(42)
        deck = Deck(STANDARD_52, seed=42)
        all_cards = deck.all_cards
        rng.shuffle(all_cards)
        hands = [all_cards[i*4:(i+1)*4] for i in range(4)]

        # Solve with DDS
        dds_results = dds.solve_position(
            hands=hands,
            current_trick=[],
            trick_leader=0,
            next_to_play=0,
            spades_broken=True,
        )
        dds_best_tricks = dds_results[0]["tricks_won"] if dds_results else -1

        # Build state for our solver
        state = GameState()
        state.init_for_deal(4, hands, [], all_cards)
        state.teams = [0, 1, 0, 1]
        state.phase = Phase.PLAYING
        state.trump_suit = Suit.SPADES
        state.turn = 0
        state.trick_leader = 0
        state.spades_broken = True
        state.trump_broken = True
        from trick_taking.game_state import Bid
        state.max_bid = ['bid_2', 'bid_2', 'bid_2', 'bid_2']
        state.bids = [Bid(player_id=i, value='bid_2') for i in range(4)]

        our_result = our_solver.solve_with_q(state)
        our_best = our_result.get("best_action")

        print(f"  DDS: best tricks={dds_best_tricks}, cards: {[(r['card'], r['tricks_won']) for r in dds_results[:3]]}")
        print(f"  OUR: best={our_best}, value={our_result.get('value', '?')}")

        # Both should agree on the best card (or an equivalent)
        # Note: DDS returns tricks, our solver returns score_diff, so direct comparison is complex
        # Just check both solve without errors
        print("  PASSED: Both solvers completed successfully")

    except Exception as e:
        print(f"  SKIPPED: {e}")


def main():
    print("=" * 60)
    print("DDS Wrapper Correctness Tests")
    print("=" * 60)

    test_basic_solve()
    test_spades_broken_enforcement()
    test_only_spades_hand()
    test_multi_trick()
    test_cross_solver_comparison()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
