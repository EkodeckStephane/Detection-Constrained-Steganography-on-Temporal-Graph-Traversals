from __future__ import annotations

from evaluation.walk_mixture import deterministic_walk_intensity


def test_walk_intensity_assignment_is_order_independent_and_declared() -> None:
    sequence_ids = ["traj-3", "traj-1", "traj-2", "traj-8", "traj-5"]
    forward = {
        sequence_id: deterministic_walk_intensity(sequence_id, seed=20260827)
        for sequence_id in sequence_ids
    }
    reverse = {
        sequence_id: deterministic_walk_intensity(sequence_id, seed=20260827)
        for sequence_id in reversed(sequence_ids)
    }
    assert forward == reverse
    assert set(forward.values()).issubset({0.05, 0.20, 0.50, 1.00})


def test_walk_intensity_assignment_is_reproducible() -> None:
    first = deterministic_walk_intensity("taxi:17:session:4", seed=20260827)
    second = deterministic_walk_intensity("taxi:17:session:4", seed=20260827)
    assert first == second
