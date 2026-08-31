from __future__ import annotations

import pandas as pd
import pytest

from steganalysis.neural_eve import (
    OOV_ID,
    _examples,
    _vocabulary,
    fit_and_score_neural_eve,
)


def _frame(split: str, *, offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (split, 0, 0, "u1", "a", 1 + offset),
            (split, 0, 1, "u1", "b", 1 + offset),
            (split, 1, 0, "u1", "a", 2 + offset),
            (split, 1, 1, "u1", "b", 2 + offset),
            (split, 2, 0, "u2", "a", 3 + offset),
            (split, 2, 1, "u2", "b", 3 + offset),
        ],
        columns=["split", "pair_id", "label", "source", "action", "timestamp"],
    )


@pytest.mark.parametrize(
    "kind",
    ["mlp_public_features", "gru_sequence", "sequence_transformer_eve"],
)
def test_neural_eve_uses_public_sequence_fields(kind: str) -> None:
    validation = _frame("validation")
    test = _frame("test", offset=4)

    metrics = fit_and_score_neural_eve(
        validation,
        test,
        kind=kind,
        context_length=3,
        embedding_dim=8,
        hidden_dim=12,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=3,
    )

    assert 0.0 <= metrics.auc <= 1.0
    assert 0.5 <= metrics.adversarial_auc <= 1.0


def test_test_only_identifiers_are_oov_and_do_not_expand_vocabulary() -> None:
    validation = _frame("validation")
    test = _frame("test", offset=4).copy()
    test.loc[test.index[0], "source"] = "never-seen-source"
    test.loc[test.index[0], "action"] = "never-seen-action"

    source_vocab = _vocabulary(validation["source"])
    action_vocab = _vocabulary(validation["action"])
    assert "never-seen-source" not in source_vocab
    assert "never-seen-action" not in action_vocab

    sources, actions, _, _, _ = _examples(test, source_vocab, action_vocab, context_length=3)
    assert int(sources[0]) == OOV_ID
    assert int(actions[0]) == OOV_ID


def test_historical_temporal_graph_name_is_only_backward_compatible_alias() -> None:
    metrics = fit_and_score_neural_eve(
        _frame("validation"),
        _frame("test", offset=4),
        kind="temporal_graph_eve",
        context_length=3,
        embedding_dim=8,
        hidden_dim=12,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=7,
    )
    assert 0.5 <= metrics.adversarial_auc <= 1.0
