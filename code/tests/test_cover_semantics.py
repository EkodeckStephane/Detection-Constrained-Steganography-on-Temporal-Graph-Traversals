from __future__ import annotations

import numpy as np
import pytest

from stego.coding import Candidate
from stego.cover_semantics import select_cover_action


def test_actor_action_abstention_preserves_natural_carrier_action() -> None:
    candidates = [Candidate("item-b", 0.9), Candidate("item-c", 0.1)]
    action = select_cover_action(
        transition_semantics="actor_action",
        natural_action="item-c",
        candidates=candidates,
        admissible_actions={"item-b", "item-c"},
        state_diverged=True,
        rng=np.random.default_rng(7),
    )
    assert action == "item-c"


def test_walk_before_divergence_preserves_natural_continuation() -> None:
    action = select_cover_action(
        transition_semantics="walk",
        natural_action="cell-b",
        candidates=[Candidate("cell-b", 0.4), Candidate("cell-c", 0.6)],
        admissible_actions={"cell-b", "cell-c"},
        state_diverged=False,
        rng=np.random.default_rng(7),
    )
    assert action == "cell-b"


def test_walk_after_divergence_samples_only_current_admissible_q() -> None:
    rng = np.random.default_rng(20260828)
    candidates = [Candidate("cell-x", 0.75), Candidate("cell-y", 0.25)]
    draws = [
        select_cover_action(
            transition_semantics="walk",
            natural_action="counterfactual-cell",
            candidates=candidates,
            admissible_actions={"cell-x", "cell-y"},
            state_diverged=True,
            rng=rng,
        )
        for _ in range(200)
    ]
    assert set(draws) <= {"cell-x", "cell-y"}
    assert "counterfactual-cell" not in draws
    assert draws.count("cell-x") > draws.count("cell-y")


def test_walk_after_divergence_requires_payload_independent_cover_rng() -> None:
    with pytest.raises(ValueError, match="cover RNG"):
        select_cover_action(
            transition_semantics="walk",
            natural_action="counterfactual-cell",
            candidates=[Candidate("cell-x", 1.0)],
            admissible_actions={"cell-x"},
            state_diverged=True,
            rng=None,
        )
