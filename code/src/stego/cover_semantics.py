from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np

from stego.coding import Candidate

TransitionSemantics = Literal["actor_action", "walk"]


def select_cover_action(
    *,
    transition_semantics: TransitionSemantics,
    natural_action: Hashable,
    candidates: Sequence[Candidate],
    admissible_actions: frozenset[Hashable] | set[Hashable],
    state_diverged: bool,
    rng: np.random.Generator | None = None,
) -> Hashable:
    """Select the secret-independent action used by COVER/PAUSE/STOP.

    ``actor_action`` domains have an exogenous actor/source process.  The
    observed natural action is therefore the carrier action that would occur
    without embedding, so abstention leaves it unchanged.  This prevents cover
    model mismatch P-vs-Q from being counted as steganographic modification.

    ``walk`` domains are endogenous: an emitted destination becomes the next
    source.  Before divergence the observed natural continuation is retained.
    After divergence that counterfactual action may be invalid from the stego
    state, so the continuation is sampled from the current admissible causal
    cover distribution Q using a public, payload-independent RNG stream.
    """

    if transition_semantics not in {"actor_action", "walk"}:
        raise ValueError("unknown transition semantics")
    if natural_action not in admissible_actions and (
        transition_semantics == "actor_action" or not state_diverged
    ):
        raise ValueError("natural cover action is not admissible in the current state")

    if transition_semantics == "actor_action" or not state_diverged:
        return natural_action

    if not candidates:
        raise ValueError("diverged walk state has no probabilistic continuation")
    if rng is None:
        raise ValueError("payload-independent cover RNG is required after walk divergence")

    filtered = [item for item in candidates if item.action in admissible_actions]
    if not filtered:
        raise ValueError("diverged walk state has no admissible cover candidate")
    probabilities = np.asarray([float(item.probability) for item in filtered], dtype=float)
    if np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
        raise ValueError("cover candidate probabilities must contain positive mass")
    probabilities /= probabilities.sum()
    index = int(rng.choice(len(filtered), p=probabilities))
    return filtered[index].action
