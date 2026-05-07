"""
General card game driver loop — the paper's Fig. 3.

Paper reference: Section 4 "General Card Game Play", Fig. 3
"The general driver loop is implemented in Fig. 3 and is aimed to serve
all games, so that games may or may not have extra dog cards to be taken
and discarded (as in Tarot and Skat)."

This module is completely game-agnostic. It only uses the GameRules and
AIPlayer interfaces to run any trick-taking card game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from trick_taking.card import Card, cards_to_bitset
from trick_taking.deck import Deck
from trick_taking.game_state import GameState, Phase, Bid
from trick_taking.game_rules import GameRules
from trick_taking.player import AIPlayer


@dataclass
class GameResult:
    """Result of a single game (deal)."""
    scores: list[float]
    tricks_won: list[int]
    winner: int  # player with best score
    game_name: str = ""
    round_number: int = 0


@dataclass
class MatchResult:
    """Result of a multi-game match."""
    game_results: list[GameResult] = field(default_factory=list)
    cumulative_scores: list[float] = field(default_factory=list)
    num_games: int = 0

    @property
    def winner(self) -> int:
        return max(range(len(self.cumulative_scores)),
                   key=lambda i: self.cumulative_scores[i])


class GeneralCardGame:
    """
    Paper's Fig. 3 general card game driver loop.

    Pseudocode from paper:
        void GeneralCardGame(AI ai[N_PLAYERS], Game game)
            list<card> hands = deal(deck, distrib);
            for (auto h : hands) { ai[i].startGame(i, h); i++; }
            int bid[N_PLAYERS], turn = 0;
            while (!end_bidding(game))
                int b = ai[game.turn].placeBid(max_bid);
                for (int i = 0; i < N_PLAYERS; i++) ai[i].bidPlaced(b);
                update_bid(game, b);
            card dog = allbits & ~allhands;
            for (int i = 0; i < N_PLAYERS; i++) ai[i].setTeams(game.team, bid);
            if (dog) ... declareGame / setGame ...
            while (!end_trickgame(game))
                card c = ai[game.turn].playCard();
                ai[game.turn].hand &= ~c;
                for (int p = 0; p < N_PLAYERS; p++) ai[p].cardPlayed(game.turn, c);
                updateCard(game, c);

    This implementation faithfully follows the paper's control flow.
    """

    def __init__(self, rules: GameRules, players: list[AIPlayer],
                 seed: Optional[int] = None) -> None:
        if len(players) != rules.num_players:
            raise ValueError(
                f"Expected {rules.num_players} players, got {len(players)}"
            )
        self.rules = rules
        self.players = players
        self.seed = seed
        self.state = GameState()
        self._deal_count = 0

    def play_game(self) -> GameResult:
        """
        Execute one complete game: deal → bid → declare → play tricks → score.
        Returns the game result with per-player scores.
        """
        self._deal()
        self._start_game()
        self._bidding_phase()
        self._set_teams()
        self._declaration_phase()
        self._exchange_phase()
        self._trick_phase()
        return self._scoring()

    def play_match(self, num_games: int) -> MatchResult:
        """Play multiple games and accumulate scores."""
        result = MatchResult(
            cumulative_scores=[0.0] * self.rules.num_players,
            num_games=num_games,
        )
        for i in range(num_games):
            self._deal_count = i
            game_result = self.play_game()
            game_result.round_number = i
            result.game_results.append(game_result)
            for pid in range(self.rules.num_players):
                result.cumulative_scores[pid] += game_result.scores[pid]
        return result

    # ─── Phase implementations (Paper's Fig. 3) ─────────────────────

    def _deal(self) -> None:
        """Paper: "list<card> hands = deal(deck, distrib);" """
        rules = self.rules
        effective_seed = (
            (self.seed + self._deal_count) if self.seed is not None else None
        )
        deck = Deck(rules.deck_config, seed=effective_seed)

        num_p = rules.num_players
        hands = [deck.deal(rules.cards_per_hand) for _ in range(num_p)]
        dog = deck.deal(rules.dog_size) if rules.dog_size > 0 else []

        self.state = GameState()
        self.state.init_for_deal(num_p, hands, dog, deck.all_cards)
        self.state.phase = Phase.DEALING
        self.state.dealer_seat = self._deal_count % num_p
        self.state.turn = (self.state.dealer_seat + 1) % num_p
        self.state.current_bidder = self.state.turn
        self.state.trick_leader = self.state.turn

    def _start_game(self) -> None:
        """Paper: "for (auto h : hands) { ai[i].startGame(i, h); }" """
        for i, player in enumerate(self.players):
            player.start_game(
                position=i,
                hand=list(self.state.hands[i]),  # copy — player can't cheat
                num_players=self.rules.num_players,
            )

    def _bidding_phase(self) -> None:
        """
        Paper: "while (!end_bidding(game))
            int b = ai[game.turn].placeBid(max_bid);
            for (int i = 0; i < N_PLAYERS; i++) ai[i].bidPlaced(b);
            update_bid(game, b);"
        """
        if not self.rules.has_bidding:
            return

        self.state.phase = Phase.BIDDING

        while not self.rules.end_bidding(self.state):
            bidder = self.state.current_bidder
            legal = self.rules.legal_bids(self.state, bidder)

            if not legal:
                # No legal bids — auto-pass
                bid_value = "pass"
            else:
                view = self.state.get_player_view(bidder)
                bid_value = self.players[bidder].place_bid(legal, view)

            # Validate bid
            if legal and bid_value not in legal:
                raise ValueError(
                    f"Player {bidder} bid {bid_value!r}, "
                    f"legal bids: {legal}"
                )

            # Record bid
            is_pass = (bid_value == "pass")
            bid_record = Bid(player_id=bidder, value=bid_value, is_pass=is_pass)
            self.state.bids.append(bid_record)
            if not is_pass:
                self.state.max_bid[bidder] = bid_value

            # Notify all players
            for player in self.players:
                player.bid_placed(bidder, bid_value)

            # Next bidder
            self.state.current_bidder = self.rules.next_bid_turn(self.state)

    def _set_teams(self) -> None:
        """Paper: "for (int i = 0; i < N_PLAYERS; i++) ai[i].setTeams(game.team, bid);" """
        self.state.teams = self.rules.set_team(self.state)
        self.state.points = self.rules.initial_points(self.state)

        bid_values = [b.value for b in self.state.bids]
        for player in self.players:
            player.set_teams(list(self.state.teams), bid_values)

    def _declaration_phase(self) -> None:
        """Paper: "if (!ai[game.declarer].isHandGame(game)) ... declareGame ..." """
        if not self.rules.has_declaration:
            return

        self.state.phase = Phase.DECLARING
        declarer = self.state.declarer or 0
        legal = self.rules.legal_declarations(self.state, declarer)

        if legal:
            view = self.state.get_player_view(declarer)
            declaration = self.players[declarer].declare_game(legal, view)
            self.state.declaration = declaration

            # Notify all players
            for player in self.players:
                player.set_game(declaration)

    def _exchange_phase(self) -> None:
        """Paper: Dog/Skat/Talon exchange if applicable."""
        if not self.rules.has_exchange:
            return

        self.state.phase = Phase.EXCHANGING
        declarer = self.state.declarer or 0

        view = self.state.get_player_view(declarer)
        if self.players[declarer].is_hand_game(
            list(self.state.dog), view
        ):
            return  # Hand game — no exchange

        # Give dog to declarer, let them exchange
        legal = self.rules.legal_exchanges(self.state, declarer)
        if legal:
            view = self.state.get_player_view(declarer)
            # Exchange is game-specific; simplified here
            pass

    def _trick_phase(self) -> None:
        """
        Paper's main playing loop:
        "while (!end_trickgame(game))
            card c = ai[game.turn].playCard();
            ai[game.turn].hand &= ~c;
            for (int p = 0; p < N_PLAYERS; p++) ai[p].cardPlayed(game.turn, c);
            updateCard(game, c);"
        """
        self.state.phase = Phase.PLAYING

        # Determine initial trump
        trump_suits = self.rules.trump_mask(self.state)
        if trump_suits and len(trump_suits) == 1:
            self.state.trump_suit = next(iter(trump_suits))

        while not self.rules.end_trickgame(self.state):
            # Start a new trick
            self.state.table_cards = []

            for _ in range(self.rules.num_players):
                current = self.state.turn
                hand = self.state.hands[current]

                # Get legal cards (paper's "playable")
                legal_cards = self.rules.playable(self.state, hand, current)
                if not legal_cards:
                    raise RuntimeError(
                        f"Player {current} has no legal cards! "
                        f"Hand: {hand}, Table: {self.state.table_cards}"
                    )

                # AI chooses a card (paper's "playCard")
                view = self.state.get_player_view(current)
                card = self.players[current].play_card(legal_cards, view)

                # Validate
                if card not in legal_cards:
                    raise ValueError(
                        f"Player {current} played {card}, "
                        f"legal: {legal_cards}"
                    )

                # Update state (paper: "ai[game.turn].hand &= ~c;")
                self.state.play_card_to_table(current, card)

                # Check trump broken
                if (self.state.trump_suit is not None
                        and card.suit == self.state.trump_suit):
                    self.state.trump_broken = True
                    self.state.spades_broken = True

                # Notify all players (paper: "for all p: ai[p].cardPlayed(turn, c)")
                for player in self.players:
                    player.card_played(current, card)

                # Next player in trick
                self.state.turn = (current + 1) % self.rules.num_players

            # Trick complete — determine winner (paper's "winner_trick")
            winner = self.rules.winner_trick(self.state)
            self.state.complete_trick(winner)

            # Winner leads next trick
            self.state.turn = winner
            self.state.trick_leader = winner

        self.state.phase = Phase.SCORING

    def _scoring(self) -> GameResult:
        """Paper: "score — determine the current score of a given player" """
        scores = self.rules.score(self.state)
        self.state.points = scores

        best = max(range(len(scores)), key=lambda i: scores[i])
        return GameResult(
            scores=scores,
            tricks_won=list(self.state.tricks_won),
            winner=best,
            game_name=self.rules.game_name,
        )
