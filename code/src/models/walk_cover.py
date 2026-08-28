from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Iterable

import pandas as pd

from stego.coding import Candidate


class WalkCoverModel:
    """First-order walk cover model with separate likelihood and support roles.

    ``likelihood_distribution`` implements the public cover model Q. It backs
    off to the global cover distribution when a source was unseen in
    ``cover_train`` so that a natural validation transition remains scoreable.

    ``admissible_distribution`` implements the observed outgoing support A(H_t).
    ``viable_admissible_distribution`` further restricts this support to the
    greatest cover-train-only outgoing viability kernel. Once a steganographic
    walk diverges, using the viable support guarantees that every emitted next
    state has at least one future continuation inside the same public kernel.
    """

    def __init__(self, *, prior_strength: float = 8.0, top_k: int = 32) -> None:
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if top_k < 2:
            raise ValueError("top_k must be at least two")
        self.prior_strength = float(prior_strength)
        self.top_k = int(top_k)
        self._source_counts: dict[Hashable, Counter[Hashable]] = {}
        self._global_counts: Counter[Hashable] = Counter()
        self._viable_nodes: frozenset[Hashable] = frozenset()
        self._entropy_scale_cache: dict[float, float] = {}

    @property
    def destinations(self) -> frozenset[Hashable]:
        return frozenset(self._global_counts)

    @property
    def viable_nodes(self) -> frozenset[Hashable]:
        self._ensure_fitted()
        return self._viable_nodes

    def fit(self, frame: pd.DataFrame) -> WalkCoverModel:
        required = {"source", "destination"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing walk columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit walk cover model on an empty frame")
        return self.fit_pairs(frame[["source", "destination"]].itertuples(index=False, name=None))

    def fit_pairs(
        self,
        pairs: Iterable[tuple[Hashable, Hashable]],
    ) -> WalkCoverModel:
        source_counts: defaultdict[Hashable, Counter[Hashable]] = defaultdict(Counter)
        global_counts: Counter[Hashable] = Counter()
        rows = 0
        for source, destination in pairs:
            source_counts[source][destination] += 1
            global_counts[destination] += 1
            rows += 1
        if rows == 0:
            raise ValueError("Cannot fit walk cover model on an empty stream")
        self._source_counts = dict(source_counts)
        self._global_counts = global_counts
        self._viable_nodes = _greatest_outgoing_viability_kernel(self._source_counts)
        self._entropy_scale_cache.clear()
        return self

    def has_context(
        self,
        source: Hashable,
        previous_destination: Hashable | None = None,
    ) -> bool:
        del previous_destination
        return source in self._source_counts

    def source_observation_count(self, source: Hashable) -> int:
        counts = self._source_counts.get(source)
        return 0 if counts is None else int(sum(counts.values()))

    def context_observation_count(
        self,
        source: Hashable,
        previous_destination: Hashable | None = None,
    ) -> int:
        del previous_destination
        return self.source_observation_count(source)

    def admissible_actions(self, source: Hashable) -> frozenset[Hashable]:
        counts = self._source_counts.get(source)
        return frozenset() if counts is None else frozenset(counts)

    def admissible_count(self, source: Hashable) -> int:
        return len(self.admissible_actions(source))

    def viable_admissible_actions(self, source: Hashable) -> frozenset[Hashable]:
        return frozenset(
            action
            for action in self.admissible_actions(source)
            if action in self._viable_nodes
        )

    def viable_admissible_count(self, source: Hashable) -> int:
        return len(self.viable_admissible_actions(source))

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None = None,
    ) -> list[Candidate]:
        """Cover-likelihood distribution Q, including global cold-start backoff."""

        del previous_destination
        self._ensure_fitted()
        counts = self._source_counts.get(source)
        if counts is None:
            return _ranked_distribution(self._global_counts, self.top_k)

        source_top = {item for item, _ in counts.most_common(self.top_k)}
        global_top = {item for item, _ in self._global_counts.most_common(self.top_k)}
        support = source_top | global_top
        source_total = sum(counts.values())
        global_total = sum(self._global_counts.values())
        denominator = source_total + self.prior_strength
        probabilities = {
            item: (
                counts[item]
                + self.prior_strength * self._global_counts[item] / global_total
            )
            / denominator
            for item in support
        }
        return _ranked_probabilities(probabilities, self.top_k)

    def likelihood_distribution(self, source: Hashable) -> list[Candidate]:
        return self.candidate_distribution(source)

    def admissible_distribution(self, source: Hashable) -> list[Candidate]:
        """Return Q restricted and renormalized on observed outgoing support A."""

        return self._restricted_distribution(source, self.admissible_actions(source))

    def viable_admissible_distribution(self, source: Hashable) -> list[Candidate]:
        """Return Q restricted to the public sink-free viability kernel A+."""

        return self._restricted_distribution(source, self.viable_admissible_actions(source))

    def _restricted_distribution(
        self,
        source: Hashable,
        actions: frozenset[Hashable],
    ) -> list[Candidate]:
        if not actions:
            return []
        candidates = [
            item
            for item in self.candidate_distribution(source)
            if item.action in actions
        ]
        if not candidates:
            return []
        mass = sum(float(item.probability) for item in candidates)
        if mass <= 0:
            return []
        return [
            Candidate(item.action, float(item.probability) / mass)
            for item in candidates
        ]

    def nonviable_probability_mass(self, source: Hashable) -> float:
        """Probability mass in A(H_t) removed by the public viability kernel."""

        base = self.admissible_distribution(source)
        if not base:
            return 1.0
        return float(
            sum(item.probability for item in base if item.action not in self._viable_nodes)
        )

    def future_admissible_count(self, action: Hashable) -> int:
        return self.admissible_count(action)

    def future_viable_count(self, action: Hashable) -> int:
        return self.viable_admissible_count(action)

    def robust_entropy_scale_bits(self, *, quantile: float = 0.95) -> float:
        """Return a cover-train-only robust entropy scale for fuzzy inputs.

        The scale is computed on the actual controllable support A+ and weighted
        by cover-training source frequency. No Eve, validation, development or
        holdout outcome enters the scale.
        """

        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        self._ensure_fitted()
        key = float(quantile)
        cached = self._entropy_scale_cache.get(key)
        if cached is not None:
            return cached

        weighted: list[tuple[float, int]] = []
        total_weight = 0
        for source, counts in self._source_counts.items():
            candidates = self.viable_admissible_distribution(source)
            if not candidates:
                continue
            probabilities = [float(item.probability) for item in candidates]
            entropy = -sum(
                probability * math.log2(probability)
                for probability in probabilities
                if probability > 0
            )
            weight = int(sum(counts.values()))
            if weight <= 0:
                continue
            weighted.append((float(entropy), weight))
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("cover-training entropy scale has no positive weight")

        target = quantile * total_weight
        cumulative = 0
        selected = 0.0
        for entropy, weight in sorted(weighted, key=lambda item: item[0]):
            cumulative += weight
            selected = entropy
            if cumulative >= target:
                break
        scale = max(float(selected), 1e-6)
        self._entropy_scale_cache[key] = scale
        return scale

    def _ensure_fitted(self) -> None:
        if not self._global_counts:
            raise ValueError("Walk cover model must be fitted before use")


def _greatest_outgoing_viability_kernel(
    source_counts: dict[Hashable, Counter[Hashable]],
) -> frozenset[Hashable]:
    """Return the greatest set in which every node has an outgoing successor.

    The computation is a reverse-pruning fixed point over unique observed edges
    and therefore depends only on cover-training topology. Any action emitted
    inside this kernel has at least one subsequent continuation inside it.
    """

    alive = set(source_counts)
    reverse: defaultdict[Hashable, list[Hashable]] = defaultdict(list)
    out_degree: dict[Hashable, int] = {}
    for source, counts in source_counts.items():
        targets = set(counts).intersection(alive)
        out_degree[source] = len(targets)
        for destination in targets:
            reverse[destination].append(source)

    queue = deque(source for source, degree in out_degree.items() if degree == 0)
    while queue:
        removed = queue.popleft()
        if removed not in alive:
            continue
        alive.remove(removed)
        for predecessor in reverse.get(removed, ()):
            if predecessor not in alive:
                continue
            out_degree[predecessor] -= 1
            if out_degree[predecessor] == 0:
                queue.append(predecessor)
    return frozenset(alive)


def _ranked_distribution(counts: Counter[Hashable], top_k: int) -> list[Candidate]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain positive mass")
    probabilities = {item: value / total for item, value in counts.items()}
    return _ranked_probabilities(probabilities, top_k)


def _ranked_probabilities(
    probabilities: dict[Hashable, float],
    top_k: int,
) -> list[Candidate]:
    ranked = sorted(
        probabilities.items(),
        key=lambda item: (-float(item[1]), repr(item[0])),
    )[:top_k]
    mass = sum(float(probability) for _, probability in ranked)
    if mass <= 0:
        raise ValueError("probabilities must contain positive mass")
    return [
        Candidate(action, float(probability) / mass)
        for action, probability in ranked
    ]
