"""Integration tests — full games with random players."""

from trick_taking.driver import GeneralCardGame
from trick_taking.games.spades import SpadesRules
from trick_taking.games.hearts import HeartsRules
from trick_taking.players.random_player import RandomPlayer


class TestSpadesIntegration:
    """Full Spades game with random players."""

    def test_full_game_completes(self) -> None:
        rules = SpadesRules(enable_nil=False, enable_blind_nil=False)
        players = [RandomPlayer(seed=i) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=42)

        result = game.play_game()

        assert result.game_name == "Spades"
        assert len(result.scores) == 4
        assert sum(result.tricks_won) == 13
        # Teams: players 0&2 get same score, 1&3 get same score
        assert result.scores[0] == result.scores[2]
        assert result.scores[1] == result.scores[3]
        # Scores are zero-sum between teams
        assert result.scores[0] + result.scores[1] == 0

    def test_multiple_games(self) -> None:
        rules = SpadesRules(enable_nil=False, enable_blind_nil=False)
        players = [RandomPlayer(seed=i) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=100)

        match = game.play_match(10)

        assert match.num_games == 10
        assert len(match.game_results) == 10
        for gr in match.game_results:
            assert sum(gr.tricks_won) == 13

    def test_nil_bidding(self) -> None:
        """Test that nil bidding works without errors."""
        rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
        players = [RandomPlayer(seed=i + 10) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=42)

        result = game.play_game()
        assert sum(result.tricks_won) == 13

    def test_deterministic(self) -> None:
        """Same seeds should produce same results."""
        rules = SpadesRules(enable_nil=False, enable_blind_nil=False)

        players1 = [RandomPlayer(seed=i) for i in range(4)]
        game1 = GeneralCardGame(rules, players1, seed=42)
        result1 = game1.play_game()

        players2 = [RandomPlayer(seed=i) for i in range(4)]
        game2 = GeneralCardGame(rules, players2, seed=42)
        result2 = game2.play_game()

        assert result1.scores == result2.scores
        assert result1.tricks_won == result2.tricks_won


class TestHeartsIntegration:
    """Full Hearts game with random players."""

    def test_full_game_completes(self) -> None:
        rules = HeartsRules()
        players = [RandomPlayer(seed=i + 20) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=42)

        result = game.play_game()

        assert result.game_name == "Hearts"
        assert len(result.scores) == 4
        assert sum(result.tricks_won) == 13
        # All scores should be <= 0 (penalties are negative)
        for s in result.scores:
            assert s <= 0

    def test_total_points_26(self) -> None:
        """Total penalty points should be 26 (or 78 if shoot the moon)."""
        rules = HeartsRules()
        players = [RandomPlayer(seed=i + 30) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=42)

        result = game.play_game()

        # Negative scores, so total should be -26 or -78 (shoot the moon)
        total = abs(sum(result.scores))
        assert total in (26, 78)

    def test_multiple_games(self) -> None:
        rules = HeartsRules()
        players = [RandomPlayer(seed=i + 40) for i in range(4)]
        game = GeneralCardGame(rules, players, seed=100)

        match = game.play_match(10)

        assert match.num_games == 10
        for gr in match.game_results:
            assert sum(gr.tricks_won) == 13

    def test_deterministic(self) -> None:
        rules = HeartsRules()

        players1 = [RandomPlayer(seed=i) for i in range(4)]
        game1 = GeneralCardGame(rules, players1, seed=42)
        result1 = game1.play_game()

        players2 = [RandomPlayer(seed=i) for i in range(4)]
        game2 = GeneralCardGame(rules, players2, seed=42)
        result2 = game2.play_game()

        assert result1.scores == result2.scores
