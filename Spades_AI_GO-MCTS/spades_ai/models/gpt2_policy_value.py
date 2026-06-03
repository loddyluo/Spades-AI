"""GPT-2 based policy + value model for Spades AI."""
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


# EOS token id — matches vocabulary.py ordering ("BOS"=0, "EOS"=1, "PAD"=2, ...)
EOS_TOKEN_ID = 1


@dataclass
class GPT2PolicyValueConfig:
    """Configuration for the dual-head GPT-2 model."""
    # vocab_size 438 = 385 baseline tokens + 53 STEPS_<n> tokens (n in [0,52]).
    # The STEPS token is appended just before EOS by the encoder so the value
    # head can condition on "how many card-play actions remain".
    vocab_size: int = 438
    # n_positions 512 (was 256) — a full 13-trick hand encodes to ~415 tokens
    # (BOS + bids + 13 * trick-block + STEPS_<n> + EOS).  At 256 every single
    # training sample was being truncated to its tail, dropping the first
    # ~5-6 tricks and crippling the model.  512 fits everything with margin.
    n_positions: int = 512
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 8
    num_labels: int = 721
    embd_pdrop: float = 0.05
    resid_pdrop: float = 0.05
    attn_pdrop: float = 0.05
    activation_function: str = "gelu_new"


class GPT2PolicyValueModel(nn.Module):
    """GPT-2 backbone with a language-model head and a value head.

    Forward returns:
        logits  : (B, S, vocab_size)  — policy logits over every token position
        value   : (B, num_labels)     — value distribution read from the
                                        *last EOS-anchored position* in each
                                        sequence (falls back to the last
                                        position if no EOS is present).

    The value head is intentionally renamed to ``value_head_v3`` (was
    ``value_head_v2`` in the previous fix, ``value_head`` originally) so
    that *any* pre-fix checkpoint fails loudly with a missing-key error
    if loaded against this model.  This protects against silent
    regressions where the head would be queried on hidden states drawn
    from a distribution it was never trained on (e.g. truncated
    sequences, terminal-only training data, or PAD-position hiddens).
    """

    def __init__(self, config: GPT2PolicyValueConfig) -> None:
        super().__init__()
        self.config = config

        gpt2_cfg = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            embd_pdrop=config.embd_pdrop,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            activation_function=config.activation_function,
        )
        self.transformer = GPT2Model(gpt2_cfg)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Renamed again (v2 -> v3) — see class docstring.
        self.value_head_v3 = nn.Linear(config.n_embd, config.num_labels)

    def forward(self, input_ids: torch.Tensor):
        """Run the model.

        Args:
            input_ids: Long tensor of shape (B, S).

        Returns:
            logits: (B, S, vocab_size)
            value:  (B, num_labels)
        """
        hidden = self.transformer(input_ids).last_hidden_state  # (B, S, n_embd)
        logits = self.lm_head(hidden)                           # (B, S, vocab_size)

        # Read value from the LAST EOS position in each sequence so the
        # train/inference distributions match (training samples always
        # contain a trailing EOS before any PAD; inference is responsible
        # for appending an EOS too — see go_mcts._get_value).
        B, S = input_ids.shape
        eos_mask = input_ids == EOS_TOKEN_ID                    # (B, S) bool
        # ``argmax`` on a boolean tensor returns the first True index.
        # Flipping reverses the sequence so we actually get the *last* True
        # index in the original.  When a row has no EOS at all, fall back
        # to position S-1 (defensive — the inference path guarantees EOS).
        any_eos = eos_mask.any(dim=1)
        last_eos_idx = (S - 1) - eos_mask.flip(dims=[1]).int().argmax(dim=1)
        idx = torch.where(
            any_eos,
            last_eos_idx,
            torch.full((B,), S - 1, dtype=torch.long, device=input_ids.device),
        )                                                       # (B,)
        value = self.value_head_v3(hidden[torch.arange(B, device=input_ids.device), idx])
        return logits, value
