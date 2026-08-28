from __future__ import annotations

import pandas as pd

from models.temporal import TemporalBackoffModel, evaluate_temporal_splits


def temporal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u1", "u2", "u2", "u1", "u1"],
            "destination": ["a", "b", "a", "b", "c", "d", "a", "b"],
            "timestamp": list(range(8)),
            "split": ["train", "train", "train", "train", "train", "validation", "test", "test"],
        }
    )


def test_temporal_backoff_conditions_on_previous_destination() -> None:
    model = TemporalBackoffModel(prior_strength=1.0, top_k=4).fit(
        temporal_frame().loc[lambda frame: frame["split"] == "train"]
    )

    after_a = model.candidate_distribution("u1", "a")
    after_b = model.candidate_distribution("u1", "b")

    assert after_a[0].action == "b"
    assert after_b[0].action == "a"


def test_effective_observation_count_follows_actual_backoff_level() -> None:
    train = temporal_frame().loc[lambda frame: frame["split"] == "train"]
    model = TemporalBackoffModel(prior_strength=1.0, top_k=4).fit(train)

    # Exact context exists after "a" and should be used as the finest evidence.
    assert model.exact_context_observation_count("u1", "a") > 0
    assert model.context_observation_count("u1", "a") == model.exact_context_observation_count("u1", "a")

    # An unseen exact history for an observed source backs off to source-level
    # evidence rather than being treated as a complete cold start.
    assert model.exact_context_observation_count("u1", "never-seen") == 0
    assert model.source_observation_count("u1") > 0
    assert model.context_observation_count("u1", "never-seen") == model.source_observation_count("u1")

    # A genuinely unseen source still has zero evidence.
    assert model.context_observation_count("new-user", "anything") == 0


def test_sequence_id_isolates_history_for_shared_mobility_nodes() -> None:
    frame = pd.DataFrame(
        {
            "source": ["cell:A", "cell:B", "cell:A", "cell:C"],
            "destination": ["cell:B", "cell:D", "cell:C", "cell:E"],
            "timestamp": [1, 2, 3, 4],
            "sequence_id": ["trip-1", "trip-1", "trip-2", "trip-2"],
        }
    )
    model = TemporalBackoffModel(prior_strength=1.0, top_k=5).fit(frame)
    predictions = list(model.iter_predictions(frame))

    # trip-2 begins at cell:A independently; it must not inherit trip-1's
    # previous destination merely because both trajectories visit cell:A.
    assert predictions[0].previous_destination is None
    assert predictions[2].previous_destination is None
    assert predictions[1].previous_destination == "cell:B"
    assert predictions[3].previous_destination == "cell:C"


def test_temporal_split_evaluation_reports_metrics() -> None:
    metrics = evaluate_temporal_splits(temporal_frame(), prior_strength=1.0, top_k=4)

    assert set(metrics) == {"train", "validation", "test"}
    assert metrics["test"].rows == 2
    assert metrics["test"].mean_entropy_bits >= 0
