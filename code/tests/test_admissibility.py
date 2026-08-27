from __future__ import annotations

import pandas as pd

from models.admissibility import (
    AdmissibilityConstrainedCoverModel,
    ObservedOutgoingAdmissibility,
    PublicCatalogAdmissibility,
)
from models.temporal import TemporalBackoffModel


def test_mobility_oracle_rejects_unobserved_and_unseen_source_moves() -> None:
    frame = pd.DataFrame(
        {
            "source": ["cell:A", "cell:A", "cell:B"],
            "destination": ["cell:B", "cell:C", "cell:C"],
            "timestamp": [1, 2, 3],
        }
    )
    oracle = ObservedOutgoingAdmissibility(allow_global_catalog=False).fit(frame)

    assert oracle.contains("cell:A", "cell:B")
    assert oracle.contains("cell:A", "cell:C")
    assert not oracle.contains("cell:A", "cell:Z")
    assert oracle.actions("cell:UNSEEN") == frozenset()


def test_actor_catalog_mode_can_allow_global_destination_catalog_for_unseen_actor() -> None:
    frame = pd.DataFrame(
        {
            "source": ["user-1", "user-2"],
            "destination": ["item-A", "item-B"],
            "timestamp": [1, 2],
        }
    )
    oracle = ObservedOutgoingAdmissibility(allow_global_catalog=True).fit(frame)

    assert oracle.actions("unseen-user") == frozenset({"item-A", "item-B"})


def test_public_catalog_is_source_independent_and_can_be_fixed_ex_ante() -> None:
    frame = pd.DataFrame(
        {
            "source": ["user-1", "user-2"],
            "destination": ["item-A", "item-B"],
            "timestamp": [1, 2],
        }
    )
    oracle = PublicCatalogAdmissibility({"item-A", "item-B", "item-C"}).fit(frame)

    expected = frozenset({"item-A", "item-B", "item-C"})
    assert oracle.actions("user-1") == expected
    assert oracle.actions("unseen-user") == expected
    assert oracle.contains("user-1", "item-C")
    assert not oracle.contains("user-1", "item-Z")


def test_public_catalog_does_not_accept_training_destination_outside_schema() -> None:
    frame = pd.DataFrame(
        {
            "source": ["user-1"],
            "destination": ["item-B"],
            "timestamp": [1],
        }
    )
    try:
        PublicCatalogAdmissibility({"item-A"}).fit(frame)
    except ValueError as error:
        assert "must contain every cover-training destination" in str(error)
    else:
        raise AssertionError("Expected fixed public catalog validation to fail")


def test_constrained_cover_never_returns_action_outside_observed_support() -> None:
    frame = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B"],
            "destination": ["B", "C", "C", "D"],
            "timestamp": [1, 2, 3, 4],
        }
    )
    model = AdmissibilityConstrainedCoverModel(
        TemporalBackoffModel(prior_strength=1.0, top_k=4),
        ObservedOutgoingAdmissibility(allow_global_catalog=False),
    ).fit(frame)

    candidates = model.candidate_distribution("A", None)

    assert candidates
    assert {candidate.action for candidate in candidates} <= {"B", "C"}
    assert abs(sum(candidate.probability for candidate in candidates) - 1.0) < 1e-12
    assert model.candidate_distribution("UNSEEN", None) == []
