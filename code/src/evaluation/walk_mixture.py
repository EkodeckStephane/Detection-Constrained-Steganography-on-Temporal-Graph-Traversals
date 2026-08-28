from __future__ import annotations

import hashlib
from collections.abc import Sequence


def deterministic_walk_intensity(
    sequence_id: str,
    *,
    intensities: Sequence[float] = (0.05, 0.20, 0.50, 1.00),
    seed: int = 20260827,
) -> float:
    """Assign one design-Eve embedding intensity to a complete walk unit.

    The assignment depends only on the public seed and ``sequence_id``. It is
    therefore invariant to file order, sharding, multiprocessing and a variable
    number of arithmetic-coder or cover-sampling calls inside other sequences.
    Each trajectory/session is represented exactly once in Eve training.
    """

    values = tuple(float(value) for value in intensities)
    if not values:
        raise ValueError("intensities must be non-empty")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("intensities must lie in [0, 1]")
    digest = hashlib.sha256(
        f"intensity:{int(seed)}:{sequence_id}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:8], "big", signed=False) % len(values)
    return values[index]
