from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from models.temporal import TemporalBackoffModel
from stego.coding import Candidate


class CausalBackoffCoverModel:
    """ASOC V2 backoff cover with an explicit causal-history convention.

    ``actor_history`` conditions an actor on its previous emitted action.
    ``walk_source`` treats the current node as the complete first-order state.
    Keeping this choice explicit prevents mobility experiments from silently
    inheriting the actor-action history semantics used by interaction streams.
    """

    def __init__(
        self,
        *,
        prior_strength: float = 8.0,
        top_k: int = 32,
        history_mode: str = "actor_history",
    ) -> None:
        self._model = TemporalBackoffModel(
            prior_strength=prior_strength,
            top_k=top_k,
            history_mode=history_mode,
        )
        self.history_mode = history_mode

    def fit(self, train: pd.DataFrame) -> CausalBackoffCoverModel:
        self._model.fit(train)
        return self

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        return self._model.candidate_distribution(source, previous_destination)

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        return self._model.has_context(source, previous_destination)

    @property
    def destinations(self) -> frozenset[Hashable]:
        return self._model.destinations
