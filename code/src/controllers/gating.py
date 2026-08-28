from __future__ import annotations

import hashlib


class PublicEmbedGate:
    """Secret-independent deterministic thinning of EMBED-eligible events.

    The gate maps a public event index and a public seed to a uniform variate
    using SHA-256. Alice, Bob and Eve can therefore reproduce the decision
    without sharing secret state or consuming the payload RNG. Increasing the
    throttle produces a nested superset of accepted public events.
    """

    def __init__(self, throttle: float, *, seed: int = 20260827) -> None:
        if not 0.0 <= throttle <= 1.0:
            raise ValueError("throttle must lie in [0, 1]")
        self.throttle = float(throttle)
        self.seed = int(seed)

    def uniform(self, event_index: int) -> float:
        if event_index < 0:
            raise ValueError("event_index must be non-negative")
        payload = f"{self.seed}:{int(event_index)}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        value = int.from_bytes(digest[:8], "big", signed=False)
        return value / float(1 << 64)

    def allow_embed(self, event_index: int) -> bool:
        return self.uniform(event_index) < self.throttle
