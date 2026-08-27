from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DetectorMetrics:
    # Raw orientation is retained for diagnostics and legacy comparability.
    auc: float
    # Primary ASOC V2 detectability metric. A detector/adversary can invert a
    # systematically reversed score, so AUC < 0.5 is not evidence of security.
    adversarial_auc: float
    balanced_accuracy: float
    eer: float


def make_detector(name: str, *, seed: int) -> object:
    if name == "linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        alpha=1e-3,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown detector: {name}")


def fit_detector(name: str, x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> object:
    detector = make_detector(name, seed=seed)
    detector.fit(x_train, y_train)
    return detector


def adversarial_auc(raw_auc: float) -> float:
    """Return orientation-invariant detectability AUC.

    If a fixed score ranks stego below cover (raw AUC < 0.5), an adversary can
    reverse the score.  Policy selection must therefore use
    ``max(AUC, 1-AUC)`` rather than rewarding reversed detectability.
    """

    value = float(raw_auc)
    if not 0.0 <= value <= 1.0:
        raise ValueError("AUC must lie in [0, 1]")
    return max(value, 1.0 - value)


def score_detector(detector: object, x_test: np.ndarray, y_test: np.ndarray) -> DetectorMetrics:
    scores = detector.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    raw_auc = float(roc_auc_score(y_test, scores))
    return DetectorMetrics(
        auc=raw_auc,
        adversarial_auc=adversarial_auc(raw_auc),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        eer=_eer(y_test, scores),
    )


def metrics_to_dict(metrics: DetectorMetrics) -> dict[str, float]:
    return {
        "auc": metrics.auc,
        "adversarial_auc": metrics.adversarial_auc,
        "balanced_accuracy": metrics.balanced_accuracy,
        "eer": metrics.eer,
    }


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    return float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)
