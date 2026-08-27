from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import pandas as pd

from models.cover_model import CoverModel
from stego.coding import Candidate


@dataclass(frozen=True)
class AdmissibilityStats:
    sources: int
    destinations: int
    observed_edges: int


class ObservedOutgoingAdmissibility:
    """Domain-support oracle learned only from declared valid transitions.

    ``allow_global_catalog`` is appropriate only when domain semantics make
    every catalog destination valid for an unseen actor (for example, an
    actor may choose any public item). It must be disabled for mobility or
    other carriers where unseen source-to-destination transitions would imply
    physically or topologically invalid moves.
    """

    def __init__(self, *, allow_global_catalog: bool = False) -> None:
        self.allow_global_catalog = bool(allow_global_catalog)
        self._outgoing: dict[Hashable, frozenset[Hashable]] = {}
        self._catalog: frozenset[Hashable] = frozenset()

    def fit(self, frame: pd.DataFrame) -> ObservedOutgoingAdmissibility:
        required = {"source", "destination"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing admissibility columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit admissibility on an empty frame")

        outgoing: defaultdict[Hashable, set[Hashable]] = defaultdict(set)
        catalog: set[Hashable] = set()
        for source, destination in frame[["source", "destination"]].itertuples(index=False):
            outgoing[source].add(destination)
            catalog.add(destination)
        self._outgoing = {
            source: frozenset(destinations)
            for source, destinations in outgoing.items()
        }
        self._catalog = frozenset(catalog)
        return self

    def actions(
        self,
        source: Hashable,
        previous_destination: Hashable | None = None,
    ) -> frozenset[Hashable]:
        del previous_destination
        if source in self._outgoing:
            return self._outgoing[source]
        if self.allow_global_catalog:
            return self._catalog
        return frozenset()

    def contains(
        self,
        source: Hashable,
        action: Hashable,
        previous_destination: Hashable | None = None,
    ) -> bool:
        return action in self.actions(source, previous_destination)

    def stats(self) -> AdmissibilityStats:
        if not self._catalog:
            raise ValueError("Admissibility oracle has not been fitted")
        return AdmissibilityStats(
            sources=len(self._outgoing),
            destinations=len(self._catalog),
            observed_edges=sum(len(value) for value in self._outgoing.values()),
        )


class AdmissibilityConstrainedCoverModel:
    """Restrict and renormalize a probabilistic cover model on valid actions."""

    def __init__(
        self,
        base_model: CoverModel,
        oracle: ObservedOutgoingAdmissibility,
    ) -> None:
        self.base_model = base_model
        self.oracle = oracle

    def fit(self, train: pd.DataFrame) -> AdmissibilityConstrainedCoverModel:
        self.base_model.fit(train)
        self.oracle.fit(train)
        return self

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        admissible = self.oracle.actions(source, previous_destination)
        if not admissible:
            return []
        candidates = [
            candidate
            for candidate in self.base_model.candidate_distribution(source, previous_destination)
            if candidate.action in admissible
        ]
        if not candidates:
            return []
        mass = sum(candidate.probability for candidate in candidates)
        if mass <= 0:
            return []
        return [
            Candidate(candidate.action, candidate.probability / mass)
            for candidate in candidates
        ]

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        return bool(self.oracle.actions(source, previous_destination)) and self.base_model.has_context(
            source, previous_destination
        )

    @property
    def destinations(self) -> frozenset[Hashable]:
        return self.base_model.destinations


def all_actions_are_admissible(
    oracle: ObservedOutgoingAdmissibility,
    rows: Iterable[tuple[Hashable, Hashable, Hashable | None]],
) -> bool:
    return all(
        oracle.contains(source, action, previous)
        for source, action, previous in rows
    )
