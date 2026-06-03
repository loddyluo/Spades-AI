from spades_ai.encoding.vocabulary import Vocabulary


class LegalityChecker:
    def __init__(self):
        self._vocab = Vocabulary()
        self._card_ids = {tid for tok, tid in self._vocab.token_to_id.items() if tok.startswith('C_')}

    def is_card_token(self, token_id):
        return token_id in self._card_ids

    def is_card_playable(self, card_token_id, played_card_tokens):
        if not self.is_card_token(card_token_id):
            return True
        return card_token_id not in played_card_tokens

    def has_duplicate_cards(self, token_sequence):
        seen = set()
        for tid in token_sequence:
            if tid in self._card_ids:
                if tid in seen:
                    return True
                seen.add(tid)
        return False

    def count_illegal_cards(self, token_sequence):
        seen, count = set(), 0
        for tid in token_sequence:
            if tid in self._card_ids:
                if tid in seen:
                    count += 1
                seen.add(tid)
        return count
