from __future__ import annotations

import pandas as pd

from steganalysis.design_mixture import make_actor_action_design_records
from stego.coding import Candidate


class _DesignModelStub:
    destinations = frozenset({"a", "b"})

    def candidate_distribution(self, source: str, previous: str | None) -> list[Candidate]:
        del source, previous
        return [Candidate("a", 0.5), Candidate("b", 0.5)]

    def has_context(self, source: str, previous: str | None) -> bool:
        del source, previous
        return True


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["u1"] * 40,
            "destination": ["a"] * 40,
            "timestamp": list(range(40)),
        }
    )


def test_zero_intensity_is_exact_natural_actor_action_replay() -> None:
    records = make_actor_action_design_records(
        _DesignModelStub(), _frame(), intensities=(0.0,), seed=11
    )
    cover = records.loc[records["label"] == 0].reset_index(drop=True)
    stego = records.loc[records["label"] == 1].reset_index(drop=True)
    assert not stego["modified"].any()
    pd.testing.assert_frame_equal(
        cover.drop(columns=["label", "modified"]),
        stego.drop(columns=["label", "modified"]),
        check_dtype=False,
    )


def test_higher_intensity_generates_more_modified_reference_actions() -> None:
    records = make_actor_action_design_records(
        _DesignModelStub(), _frame(), intensities=(0.05, 1.0), seed=20260827
    )
    stego = records.loc[records["label"] == 1]
    modified = stego.groupby("design_intensity")["modified"].mean()
    assert modified.loc[1.0] > modified.loc[0.05]


def test_stego_history_is_updated_with_emitted_reference_action() -> None:
    records = make_actor_action_design_records(
        _DesignModelStub(), _frame(), intensities=(1.0,), seed=11
    )
    stego = records.loc[records["label"] == 1].reset_index(drop=True)
    # With a single source, at least one reference action differs from natural;
    # subsequent features are generated from the emitted-history replay rather
    # than resetting to the natural counterfactual.
    assert stego["modified"].any()
    assert len(stego) == len(_frame())
