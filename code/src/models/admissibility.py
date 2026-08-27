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
    every catalog destination valid for an unseen actor. It remains disabled
    for mobility or other carriers where unseen source-to-destination moves
    would imply physically or topologically invalid transitions.
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


class PublicCatalogAdmissibility:
    """Public action-space oracle for actor-to-item interaction domains.

    The catalog defines *which actions are valid*, not their probabilities.
    Frequencies and temporal conditionals remain learned only from the declared
    cover-training region. A fixed catalog may therefore be supplied from a
    public domain schema without leaking validation/test outcome frequencies.

    When no catalog is supplied, the class conservatively falls back to the
    destinations observed in the training frame.
    """

    def __init__(self, catalog: Iterable[Hashable] | None = None) -> None:
        self._fixed_catalog = None if catalog is None else frozenset(catalog)
        self._catalog: frozenset[Hashable] = frozenset()

    def fit(self, frame: pd.DataFrame) -> PublicCatalogAdmissibility:
        required = {"source", "destination"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing admissibility columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit admissibility on an empty frame")
        observed = frozenset(frame["destination"].tolist())
        catalog = self._fixed_catalog if self._fixed_catalog is not None else observed
        if not observed.issubset(catalog):
            raise ValueError("Public catalog must contain every cover-training destination")
        if not catalog:
            raise ValueError("Public catalog must be non-empty")
        self._catalog = frozenset(catalog)
        return self

    def actions(
        self,
        source: Hashable,
        previous_destination: Hashable | None = None,
    ) -> frozenset[Hashable]:
        del source, previous_destination
        return self._catalog

    def contains(
        self,
        source: Hashable,
        action: Hashable,
        previous_destination: Hashable | None = None,
    ) -> bool:
        del source, previous_destination
        return action in self._catalog

    def stats(self) -> AdmissibilityStats:
        if not self._catalog:
            raise ValueError("Admissibility oracle has not been fitted")
        return AdmissibilityStats(
            sources=0,
            destinations=len(self._catalog),
            observed_edges=0,
        )


class AdmissibilityConstrainedCoverModel:
    """Restrict and renormalize a probabilistic cover model on valid actions."""

    def __init__(
        self,
        base_model: CoverModel,
        oracle: ObservedOutgoingAdmissibility | PublicCatalogAdmissibility,
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
    oracle: ObservedOutgoingAdmissibility | PublicCatalogAdmissibility,
    rows: Iterable[tuple[Hashable, Hashable, Hashable | None]],
) -> bool:
    return all(
        oracle.contains(source, action, previous)
        for source, action, previous in rows
    )
