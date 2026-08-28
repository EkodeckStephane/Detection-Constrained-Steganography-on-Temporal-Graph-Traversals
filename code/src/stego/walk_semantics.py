from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from stego.coding import Candidate


class WalkDeadEndError(RuntimeError):
    """Raised when a diverged emitted walk has no learned valid continuation."""


def cover_walk_action(
    *,
    observed_source: Hashable,
    natural_action: Hashable,
    emitted_source: Hashable,
    admissible_candidates: Sequence[Candidate],
    cover_rng: np.random.Generator,
) -> Hashable:
    """Return a secret-independent COVER/PAUSE action for walk semantics.

    When the emitted walk is aligned with the observed natural source, the
    observed natural action is passed through exactly.  This includes unseen
    cover-model contexts and is what makes a zero-payload run identical to the
    natural trajectory.

    After a steganographic divergence, the counterfactual natural action is no
    longer assumed reachable.  COVER/PAUSE therefore samples only from the
    current public admissible distribution.  The caller must provide a RNG that
    is independent of the payload-bit RNG.
    """

    if emitted_source == observed_source:
        return natural_action
    if not admissible_candidates:
        raise WalkDeadEndError(
            "diverged walk has no admissible continuation from emitted source"
        )
    probabilities = np.asarray(
        [max(0.0, float(item.probability)) for item in admissible_candidates],
        dtype=float,
    )
    mass = float(probabilities.sum())
    if mass <= 0:
        raise ValueError("admissible candidate probabilities must contain positive mass")
    probabilities /= mass
    index = int(cover_rng.choice(len(admissible_candidates), p=probabilities))
    return admissible_candidates[index].action
