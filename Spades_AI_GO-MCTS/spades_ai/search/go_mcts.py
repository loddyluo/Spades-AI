from dataclasses import dataclass
import torch
from spades_ai.game.card import Card
from spades_ai.search.mcts_node import MCTSNode
from spades_ai.search.legality_checker import LegalityChecker
from spades_ai.encoding.encoder import _RANK_TO_CHAR, _SUIT_TO_CHAR
from spades_ai.encoding.vocabulary import Vocabulary


@dataclass
class GOMCTSConfig:
    n_runs: int = 100
    n_steps: int = 5
    C: float = 0.3
    mu: float = 0.01
    threshold: float = 0.05


class GOMCTSSearch:
    def __init__(self, model, config=None, device=None):
        self._model = model
        self._cfg = config if config is not None else GOMCTSConfig()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        self._vocab = Vocabulary()
        self._checker = LegalityChecker()
        # Build card token id -> Card mapping
        self._card_map = {}
        for card in Card.all_cards():
            tok = f"C_{_RANK_TO_CHAR[card.rank.value]}{_SUIT_TO_CHAR[card.suit]}"
            tid = self._vocab.token_to_id.get(tok)
            if tid is not None:
                self._card_map[tid] = card

    def _card_to_token(self, card):
        tok = f"C_{_RANK_TO_CHAR[card.rank.value]}{_SUIT_TO_CHAR[card.suit]}"
        return self._vocab.token_to_id[tok]

    # ------------------------------------------------------------------
    # Helper: count plays in the current (in-progress) trick
    # ------------------------------------------------------------------
    def _count_plays_in_current_trick(self, tokens):
        """Count [POS_Px][C_card] plays in the current trick.

        Scans backwards for the last TRICK_SEP(5) and counts POS tokens
        (58-61) after it.  If 4 plays follow the separator the trick is
        already complete — the encoder only emits the current-trick
        section when current_trick_cards is non-empty, so a full quartet
        means no new trick has started in the sequence yet.
        """
        last_sep = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == 5:  # TRICK_SEP
                last_sep = i
                break

        if last_sep == -1:
            # No TRICK_SEP at all — before the first trick's plays
            return 0

        # Count POS tokens (POS_P0=58 .. POS_P3=61) after the separator
        count = 0
        for i in range(last_sep + 1, len(tokens)):
            if 58 <= tokens[i] <= 61:
                count += 1

        # 4 plays ⇒ the last trick is complete; current trick has 0 plays
        if count >= 4:
            return 0
        return count

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    def run(self, observation_tokens, legal_cards, perspective_player):
        """Returns (best_card, info_dict)."""
        legal_tids = {self._card_to_token(c) for c in legal_cards}
        root = MCTSNode()
        max_pos = self._model.config.n_positions

        # --- FIX 1: strip trailing [STEPS, EOS] before search sequences ---
        # The encoder appends [STEPS_<n>, EOS] for the value head, but the
        # policy model was trained to predict cards after POS_Px tokens;
        # appending cards after the value suffix is out-of-distribution.
        h_init = list(observation_tokens)
        if h_init and h_init[-1] == 1:  # EOS
            h_init.pop()
        if h_init and self._vocab.id_to_token[h_init[-1]].startswith("STEPS_"):
            h_init.pop()

        # --- FIX 2: POS prefix and opponent-sampling counts ---
        # pos_token: the POS_Px marker for the perspective player
        pos_token = 58 + perspective_player  # POS_P0=58 .. POS_P3=61
        # How many opponents still need to play after us in this trick?
        plays_in_trick = self._count_plays_in_current_trick(h_init)
        remaining_in_trick = max(0, 3 - plays_in_trick)

        for _ in range(self._cfg.n_runs):
            h = list(h_init)
            node = root
            path = [root]

            # ----------------------------------------------------------
            # SELECT: traverse existing tree edges using UCT.
            # At root level the action is a card token-id; the sequence
            # must contain [POS_Px, C_card] (2 tokens), not just the
            # bare card id.  Deeper levels (if any) append raw tokens.
            # ----------------------------------------------------------
            at_root = True
            while node.children and len(h) < max_pos:
                action = node.best_child_uct(self._cfg.C)
                node = node.children[action]
                if at_root:
                    h.extend([pos_token, action])  # [POS_Px, C_card]
                    at_root = False
                else:
                    h.append(action)
                path.append(node)

            # ----------------------------------------------------------
            # EXPAND: create children when node already visited
            # ----------------------------------------------------------
            if len(h) < max_pos and node.visit_count > 0:
                if node is root:
                    # Root: expand ALL legal cards for full exploration
                    for tid in legal_tids:
                        node.get_or_create_child(tid)
                else:
                    # Non-root: use threshold to prune low-probability branches
                    probs = self._get_policy(h, max_pos)
                    for tid in legal_tids:
                        if probs.get(tid, 0) >= self._cfg.threshold or not node.children:
                            node.get_or_create_child(tid)
                    if not node.children:
                        for tid in legal_tids:
                            node.get_or_create_child(tid)
                action = node.best_child_uct(self._cfg.C)
                node = node.children[action]
                if at_root:
                    h.extend([pos_token, action])  # [POS_Px, C_card]
                    at_root = False
                else:
                    h.append(action)
                path.append(node)

            # ----------------------------------------------------------
            # ROLLOUT (P0 + P1):
            #   Phase 1 — complete the current trick by sampling the
            #     remaining opponents' plays (each = 2 tokens:
            #     POS_Px + C_card) autoregressively.  These are NOT
            #     added to the search tree.
            #   Phase 2 — simulate n_steps additional complete tricks
            #     via autoregressive sampling.  Each trick is ~30
            #     tokens (state block + separator + 4 plays), so we
            #     sample up to n_steps * 30 tokens.
            # ----------------------------------------------------------
            rh = list(h)

            # Phase 1 (P0): sample remaining opponent plays in this trick
            for _ in range(remaining_in_trick):
                if len(rh) >= max_pos:
                    break
                rh.append(self._sample_next(rh, max_pos))  # POS token
                if len(rh) >= max_pos:
                    break
                rh.append(self._sample_next(rh, max_pos))  # Card token

            # Phase 2 (P1): simulate n_steps more tricks autoregressively
            tokens_per_trick = 30  # ~20 state + 1 sep + 8 play + ~1 buf
            max_rollout_tokens = self._cfg.n_steps * tokens_per_trick
            tokens_sampled = 0
            while tokens_sampled < max_rollout_tokens and len(rh) < max_pos:
                rh.append(self._sample_next(rh, max_pos))
                tokens_sampled += 1

            # EVALUATE: get value from value head
            value = self._get_value(rh, max_pos)

            # CHECK: count illegal (duplicate) cards, penalize
            illegal = self._checker.count_illegal_cards(rh)
            if illegal > 0:
                value -= self._cfg.mu * illegal

            # BACKUP: update all nodes on path
            for n in path:
                n.update(value)

        # Select best action by visit count
        visit_counts = {
            self._card_map[a]: c.visit_count
            for a, c in root.children.items()
            if a in self._card_map
        }
        best = (
            max(visit_counts, key=visit_counts.get)
            if visit_counts
            else next(iter(legal_cards))
        )
        return best, {'visit_counts': visit_counts, 'root_value': root.mean_value}

    def _get_policy(self, h, max_pos):
        ids = torch.tensor([h[-max_pos:]], dtype=torch.long, device=self._device)
        with torch.no_grad():
            lm, _ = self._model(ids)
        probs = torch.softmax(lm[0, -1, :], dim=0)
        return {i: probs[i].item() for i in range(probs.size(0))}

    def _sample_next(self, h, max_pos):
        ids = torch.tensor([h[-max_pos:]], dtype=torch.long, device=self._device)
        with torch.no_grad():
            lm, _ = self._model(ids)
        return torch.multinomial(torch.softmax(lm[0, -1, :], dim=0), 1).item()

    def _get_value(self, h, max_pos):
        # The value head is anchored on the LAST EOS position (see
        # gpt2_policy_value.GPT2PolicyValueModel.forward).  Training samples
        # always end with the pair ``[STEPS_<n>, EOS]`` before any padding,
        # where ``n`` = remaining card-play actions.  At inference time we
        # must mirror that suffix exactly so the model reads from the same
        # distribution it was trained on.
        h_eval = list(h)
        # Strip any trailing value-head suffix the search might have left in
        # place — we'll re-append the canonical [STEPS_<n>, EOS] suffix below.
        while h_eval and h_eval[-1] == 1:  # EOS = 1
            h_eval.pop()
        while h_eval and self._vocab.id_to_token[h_eval[-1]].startswith("STEPS_"):
            h_eval.pop()

        # Compute remaining_steps from the token stream.  Card tokens are
        # ``C_<rank><suit>`` which occupy a contiguous id range right after
        # the structural tokens (BOS=0, EOS=1, PAD=2, STATE=3, /STATE=4,
        # TRICK_SEP=5).  There are 52 card tokens, so card ids are 6..57.
        cards_played = sum(1 for t in h_eval if 6 <= t <= 57)
        remaining = max(0, min(52, 52 - cards_played))
        steps_token_id = self._vocab.get_id(f"STEPS_{remaining}")
        h_eval.append(steps_token_id)
        h_eval.append(1)  # EOS

        # If the appended suffix pushes us over the position budget, drop
        # tokens from the *head* so the [STEPS, EOS] anchor stays inside
        # the window.
        if len(h_eval) > max_pos:
            h_eval = h_eval[-max_pos:]

        ids = torch.tensor([h_eval], dtype=torch.long, device=self._device)
        with torch.no_grad():
            _, vl = self._model(ids)
        probs = torch.softmax(vl[0], dim=0)
        from spades_ai.training.outcome_labels import OutcomeLabeler
        lab = OutcomeLabeler()
        bins = (
            torch.arange(lab.num_labels, dtype=torch.float32, device=self._device)
            + lab.min_score
        )
        # Normalize the expected score-diff to [-1, 1] using the labeler's
        # max magnitude (decoupled from the previous hardcoded 260).
        denom = max(abs(lab.min_score), abs(lab.max_score))
        return (probs * bins).sum().item() / denom
