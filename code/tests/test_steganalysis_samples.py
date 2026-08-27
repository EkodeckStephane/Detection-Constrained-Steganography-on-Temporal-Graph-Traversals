from __future__ import annotations

import pandas as pd

from models.temporal import TemporalBackoffModel
from steganalysis.samples import FEATURE_COLUMNS, SampleConfig, feature_matrix, make_steganalysis_records
from stego.coding import Candidate


def test_make_steganalysis_records_pairs_natural_and_stego() -> None:
    frame = pd.DataFrame(
        {
            "source": ["u1", "u1", "u1", "u1"],
            "destination": ["a", "b", "a", "b"],
            "timestamp": [1, 2, 3, 4],
            "split": ["train", "train", "validation", "validation"],
        }
    )
    model = TemporalBackoffModel(prior_strength=1.0, top_k=4).fit(
        frame.loc[frame["split"] == "train"]
    )

    records = make_steganalysis_records(
        model,
        frame.loc[frame["split"] == "validation"],
        split="validation",
        config=SampleConfig(max_bits_per_transition=1, seed=11),
    )
    x, y = feature_matrix(records)

    assert len(records) == 4
    assert set(records["label"]) == {0, 1}
    assert x.shape[1] == len(FEATURE_COLUMNS)
    assert y.tolist().count(1) == 2
    assert set(records["source"]) == {"u1"}


class _HistoryAwareStub:
    destinations = frozenset({"a", "b"})

    def candidate_distribution(self, source: str, previous: str | None) -> list[Candidate]:
        del source, previous
        return [Candidate("a", 0.5), Candidate("b", 0.5)]

    def has_context(self, source: str, previous: str | None) -> bool:
        del source, previous
        return True


def test_stego_history_uses_emitted_action_not_natural_counterfactual() -> None:
    frame = pd.DataFrame(
        {
            "source": ["u1", "u1", "u1"],
            "destination": ["a", "a", "a"],
            "timestamp": [1, 2, 3],
        }
    )

    records = make_steganalysis_records(
        _HistoryAwareStub(),
        frame,
        split="validation",
        config=SampleConfig(
            max_bits_per_transition=1,
            seed=11,
            max_local_total_variation=1.0,
            max_local_kl_bits=1.0,
            min_entropy_bits=0.0,
        ),
    )

    natural = records.loc[records["label"] == 0].reset_index(drop=True)
    stego = records.loc[records["label"] == 1].reset_index(drop=True)

    # With this deterministic seed, the first one-bit symbol selects "b".
    assert stego.loc[0, "action"] == "b"
    assert natural.loc[1, "previous_action"] == "a"
    assert stego.loc[1, "previous_action"] == "b"


def test_sequence_ids_prevent_cross_trajectory_state_leakage() -> None:
    frame = pd.DataFrame(
        {
            "source": ["cell:A", "cell:A", "cell:A", "cell:A"],
            "destination": ["a", "a", "a", "a"],
            "timestamp": [1, 2, 3, 4],
            "sequence_id": ["trip-1", "trip-2", "trip-1", "trip-2"],
        }
    )

    records = make_steganalysis_records(
        _HistoryAwareStub(),
        frame,
        split="validation",
        config=SampleConfig(
            max_bits_per_transition=1,
            seed=11,
            max_local_total_variation=1.0,
            max_local_kl_bits=1.0,
            min_entropy_bits=0.0,
        ),
    )
    stego = records.loc[records["label"] == 1].reset_index(drop=True)

    assert stego.loc[0, "sequence_id"] == "trip-1"
    assert stego.loc[1, "sequence_id"] == "trip-2"
    assert stego.loc[0, "previous_action"] == ""
    assert stego.loc[1, "previous_action"] == ""
    assert stego.loc[2, "previous_action"] == stego.loc[0, "action"]
    assert stego.loc[3, "previous_action"] == stego.loc[1, "action"]


def test_walk_semantics_propagates_emitted_destination_as_next_source() -> None:
    frame = pd.DataFrame(
        {
            "source": ["cell:A", "cell:A", "cell:B", "cell:B"],
            "destination": ["a", "a", "a", "a"],
            "timestamp": [1, 2, 3, 4],
            "sequence_id": ["trip-1", "trip-2", "trip-1", "trip-2"],
        }
    )

    records = make_steganalysis_records(
        _HistoryAwareStub(),
        frame,
        split="validation",
        config=SampleConfig(
            max_bits_per_transition=1,
            seed=11,
            max_local_total_variation=1.0,
            max_local_kl_bits=1.0,
            min_entropy_bits=0.0,
            transition_semantics="walk",
        ),
    )
    stego = records.loc[records["label"] == 1].reset_index(drop=True)

    assert stego.loc[0, "source"] == "cell:A"
    assert stego.loc[1, "source"] == "cell:A"
    assert stego.loc[2, "source"] == stego.loc[0, "action"]
    assert stego.loc[3, "source"] == stego.loc[1, "action"]


class _WalkIdentityStub:
    destinations = frozenset({"B", "C", "D", "X"})

    def candidate_distribution(self, source: str, previous: str | None) -> list[Candidate]:
        del previous
        candidates = {
            "A": [Candidate("B", 0.8), Candidate("X", 0.2)],
            "B": [Candidate("C", 0.8), Candidate("X", 0.2)],
            "C": [Candidate("D", 0.8), Candidate("X", 0.2)],
            "D": [Candidate("D", 0.8), Candidate("X", 0.2)],
        }
        return candidates[source]

    def has_context(self, source: str, previous: str | None) -> bool:
        del source, previous
        return True


def test_zero_payload_walk_is_exactly_the_natural_path() -> None:
    """A null payload must not create a simulator-induced detection signal."""

    frame = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "destination": ["B", "C", "D"],
            "timestamp": [1, 2, 3],
            "sequence_id": ["trip-1", "trip-1", "trip-1"],
        }
    )

    records = make_steganalysis_records(
        _WalkIdentityStub(),
        frame,
        split="validation",
        config=SampleConfig(
            max_bits_per_transition=0,
            seed=11,
            min_entropy_bits=0.0,
            transition_semantics="walk",
        ),
    )
    natural = records.loc[records["label"] == 0].reset_index(drop=True)
    stego = records.loc[records["label"] == 1].reset_index(drop=True)

    assert stego["stego_mode"].tolist() == ["COVER", "COVER", "COVER"]
    assert stego["bits_consumed"].tolist() == [0, 0, 0]
    assert stego["source"].tolist() == natural["source"].tolist()
    assert stego["action"].tolist() == natural["action"].tolist()
    assert stego["previous_action"].tolist() == natural["previous_action"].tolist()
    pd.testing.assert_frame_equal(
        stego[FEATURE_COLUMNS],
        natural[FEATURE_COLUMNS],
        check_dtype=False,
    )
