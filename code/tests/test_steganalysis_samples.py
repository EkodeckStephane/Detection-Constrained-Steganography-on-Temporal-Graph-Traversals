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
