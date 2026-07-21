from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.state import GameState, Phase
from spades_ai.game.trick import TrickCard


def _playing_state(card: Card, *, spades_broken: bool = False) -> GameState:
    return GameState(
        hands=(
            frozenset({card}),
            frozenset(),
            frozenset(),
            frozenset(),
        ),
        bids=(),
        completed_tricks=(),
        current_trick_cards=(),
        current_player=0,
        leader=0,
        trick_number=1,
        tricks_won=(0, 0, 0, 0),
        spades_broken=spades_broken,
        phase=Phase.PLAYING,
        void_shown=(frozenset(), frozenset(), frozenset(), frozenset()),
    )


def test_leading_only_spade_breaks_spades_immutably() -> None:
    ace_of_spades = Card(rank=Rank.ACE, suit=Suit.SPADES)
    state = _playing_state(ace_of_spades)

    next_state = state.play_card(ace_of_spades)

    assert next_state.spades_broken is True
    assert state.spades_broken is False


def test_non_spade_cannot_reset_broken_state() -> None:
    ace_of_hearts = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    state = _playing_state(ace_of_hearts, spades_broken=True)

    next_state = state.play_card(ace_of_hearts)

    assert next_state.spades_broken is True


def test_fourth_card_spade_stays_broken_after_trick_resolution() -> None:
    ace_of_hearts = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    king_of_hearts = Card(rank=Rank.KING, suit=Suit.HEARTS)
    queen_of_hearts = Card(rank=Rank.QUEEN, suit=Suit.HEARTS)
    two_of_spades = Card(rank=Rank.TWO, suit=Suit.SPADES)
    state = GameState(
        hands=(
            frozenset(),
            frozenset(),
            frozenset(),
            frozenset({two_of_spades}),
        ),
        bids=(),
        completed_tricks=(),
        current_trick_cards=(
            TrickCard(player=0, card=ace_of_hearts),
            TrickCard(player=1, card=king_of_hearts),
            TrickCard(player=2, card=queen_of_hearts),
        ),
        current_player=3,
        leader=0,
        trick_number=1,
        tricks_won=(0, 0, 0, 0),
        spades_broken=False,
        phase=Phase.PLAYING,
        void_shown=(frozenset(), frozenset(), frozenset(), frozenset()),
    )

    next_state = state.play_card(two_of_spades)

    assert next_state.current_trick_cards == ()
    assert next_state.completed_tricks[-1].winner() == 3
    assert next_state.spades_broken is True
