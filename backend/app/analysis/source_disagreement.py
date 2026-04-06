"""Cross-source disagreement: spread of daily net sentiment across ingest channels."""

from __future__ import annotations

# Channels compared when measuring disagreement (must match DB data_source values).
STANDARD_CHANNELS: tuple[str, ...] = ("reddit", "news", "twitter")

# Minimum labelled rows per channel per day to include that channel in disagreement
# (reduces noise from single-post days).
MIN_ROWS_PER_CHANNEL_PER_DAY = 3


def disagreement_metrics(
    nets: dict[str, float],
) -> tuple[float | None, float | None]:
    """
    From per-channel net sentiment in [-1, 1]-ish, return (range, population std).

    Range = max(net) - min(net). Std = spread around the mean net for that day.
    Returns (None, None) if fewer than two channels qualify.
    """
    if len(nets) < 2:
        return None, None
    vals = list(nets.values())
    rng = max(vals) - min(vals)
    mean_v = sum(vals) / len(vals)
    var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    std = var**0.5
    return round(rng, 4), round(std, 4)
