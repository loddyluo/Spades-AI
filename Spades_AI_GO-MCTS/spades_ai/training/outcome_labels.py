"""Outcome labeling for Spades game results."""


class OutcomeLabeler:
    """Maps score differentials to integer class labels.

    By default covers the range [-360, 360] with bin_width=1, yielding 721
    distinct labels.  The previous default of [-260, 260] silently clamped
    extreme outcomes (e.g. nil + opponent failed contract = ±360 spread),
    starving the value head of signal at the tails.
    """

    def __init__(self, min_score: int = -360, max_score: int = 360) -> None:
        self.min_score = min_score
        self.max_score = max_score
        # Backwards-compatible aliases (used by older code that read _min/_max
        # directly).  Prefer min_score/max_score in new code.
        self._min = min_score
        self._max = max_score

    @property
    def num_labels(self) -> int:
        """Total number of distinct labels (max - min + 1)."""
        return self.max_score - self.min_score + 1

    @property
    def bin_width(self) -> int:
        """Width of each label bin (always 1 for exact mapping)."""
        return 1

    def score_diff_to_label(self, score_diff: int) -> int:
        """Convert a score differential to a label index in [0, num_labels)."""
        clamped = max(self.min_score, min(self.max_score, score_diff))
        return clamped - self.min_score

    def label_to_score_diff(self, label: int) -> int:
        """Convert a label index back to its corresponding score differential."""
        return label + self.min_score
