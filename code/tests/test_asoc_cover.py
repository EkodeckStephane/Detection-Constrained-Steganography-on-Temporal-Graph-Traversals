from __future__ import annotations

import pandas as pd

from models.asoc_cover import CausalBackoffCoverModel


def test_asoc_cover_exposes_declared_history_mode() -> None:
    actor = CausalBackoffCoverModel(history_mode="actor_history")
    walk = CausalBackoffCoverModel(history_mode="walk_source")

    assert actor.history_mode == "actor_history"
    assert walk.history_mode == "walk_source"


def test_walk_source_ignores_redundant_previous_destination_argument() -> None:
    frame = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B"],
            "destination": ["B", "C", "A", "C"],
            "timestamp": [1, 2, 3, 4],
            "sequence_id": ["t1", "t2", "t1", "t2"],
        }
    )
    model = CausalBackoffCoverModel(
        prior_strength=1.0,
        top_k=4,
        history_mode="walk_source",
    ).fit(frame)

    without_previous = model.candidate_distribution("A", None)
    with_previous = model.candidate_distribution("A", "some-prior-node")

    assert without_previous == with_previous
