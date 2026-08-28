from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.walk_cover import WalkCoverModel
from stego.coding import Candidate
from stego.walk_semantics import WalkDeadEndError, cover_walk_action


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "A", "A", "B", "X"],
            "destination": ["B", "B", "C", "C", "D"],
        }
    )


def test_unseen_walk_source_is_scoreable_but_not_admissible() -> None:
    model = WalkCoverModel(prior_strength=1.0, top_k=4).fit(_train())

    q = model.likelihood_distribution("UNSEEN")
    assert q
    assert sum(item.probability for item in q) == pytest.approx(1.0)
    assert model.admissible_actions("UNSEEN") == frozenset()
    assert model.admissible_distribution("UNSEEN") == []
    assert model.context_observation_count("UNSEEN") == 0


def test_known_walk_embedding_support_contains_only_observed_outgoing_edges() -> None:
    model = WalkCoverModel(prior_strength=1.0, top_k=4).fit(_train())

    admissible = model.admissible_distribution("A")
    assert {item.action for item in admissible}.issubset({"B", "C"})
    assert {item.action for item in admissible} == {"B", "C"}
    assert sum(item.probability for item in admissible) == pytest.approx(1.0)
    assert model.future_admissible_count("B") == 1
    assert model.future_admissible_count("D") == 0


def test_walk_entropy_scale_is_cover_train_only_positive_and_cached() -> None:
    model = WalkCoverModel(prior_strength=1.0, top_k=4).fit(_train())
    first = model.robust_entropy_scale_bits(quantile=0.95)
    second = model.robust_entropy_scale_bits(quantile=0.95)
    assert first == second
    assert 0.0 < first <= math.log2(4)
    with pytest.raises(ValueError):
        model.robust_entropy_scale_bits(quantile=0.0)


def test_zero_payload_cover_preserves_natural_action_even_for_unseen_source() -> None:
    action = cover_walk_action(
        observed_source="UNSEEN",
        natural_action="natural-next",
        emitted_source="UNSEEN",
        admissible_candidates=[],
        cover_rng=np.random.default_rng(17),
    )
    assert action == "natural-next"


def test_diverged_cover_samples_only_current_admissible_support() -> None:
    candidates = [Candidate("C", 0.8), Candidate("D", 0.2)]
    rng = np.random.default_rng(23)
    actions = {
        cover_walk_action(
            observed_source="A",
            natural_action="B",
            emitted_source="X",
            admissible_candidates=candidates,
            cover_rng=rng,
        )
        for _ in range(50)
    }
    assert actions.issubset({"C", "D"})
    assert actions


def test_diverged_walk_with_empty_support_is_explicit_dead_end() -> None:
    with pytest.raises(WalkDeadEndError):
        cover_walk_action(
            observed_source="A",
            natural_action="B",
            emitted_source="X",
            admissible_candidates=[],
            cover_rng=np.random.default_rng(29),
        )
