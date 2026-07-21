from strategy.rule_based_first4_player import RuleBasedFirst4Player
from trick_taking.card import Card


def _card(code: str) -> Card:
    return Card.from_str(code)


def test_fourth_hand_overtakes_led_spade_with_lowest_winner() -> None:
    player = RuleBasedFirst4Player()
    legal_cards = [_card("S3"), _card("SQ"), _card("SA")]
    player.start_game(position=3, hand=legal_cards, num_players=4)

    played = player.play_card(
        legal_cards,
        {
            "table_cards": [
                (0, _card("S9")),
                (1, _card("S2")),
                (2, _card("SJ")),
            ],
            "tricks_played": 0,
        },
    )

    assert played == _card("SQ")


def test_fourth_hand_follows_low_when_off_suit_spade_already_wins() -> None:
    player = RuleBasedFirst4Player()
    legal_cards = [_card("H5"), _card("HA")]
    player.start_game(position=3, hand=legal_cards, num_players=4)

    played = player.play_card(
        legal_cards,
        {
            "table_cards": [
                (0, _card("H2")),
                (1, _card("H3")),
                (2, _card("S4")),
            ],
            "tricks_played": 0,
        },
    )

    assert played == _card("H5")
