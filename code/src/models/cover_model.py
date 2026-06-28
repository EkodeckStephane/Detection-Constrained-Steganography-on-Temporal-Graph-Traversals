from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.neural_sequence import NeuralSequenceModel
from models.temporal import TemporalBackoffModel
from models.temporal_gnn import TemporalGraphCoverModel
from stego.coding import Candidate

UNKNOWN_DESTINATION = "__unknown_destination__"


@runtime_checkable
class CoverModel(Protocol):
    """Common interface for a next-destination cover model used by the steganographic pipeline."""

    def fit(self, train: pd.DataFrame) -> CoverModel: ...

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]: ...

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool: ...

    @property
    def destinations(self) -> frozenset[Hashable]: ...


class BackoffCoverModel:
    """Thin wrapper around TemporalBackoffModel to satisfy CoverModel."""

    def __init__(self, *, prior_strength: float = 8.0, top_k: int = 32) -> None:
        self._model = TemporalBackoffModel(prior_strength=prior_strength, top_k=top_k)
        self._top_k = top_k

    def fit(self, train: pd.DataFrame) -> BackoffCoverModel:
        self._model.fit(train)
        return self

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        return self._model.candidate_distribution(source, previous_destination)

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        return self._model.has_context(source, previous_destination)

    @property
    def destinations(self) -> frozenset[Hashable]:
        return self._model.destinations


class NeuralSequenceCoverModel:
    """GRU or Transformer sequence model adapted to the CoverModel protocol.

    The underlying model predicts the next destination from a fixed-length
    history of previous destinations for the same source.  During inference we
    maintain that history source-by-source so that the model remains causal.
    """

    def __init__(
        self,
        model: NeuralSequenceModel,
        destination_to_id: dict[str, int],
        id_to_destination: dict[int, str],
        *,
        context_length: int,
        top_k: int = 32,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._destination_to_id = destination_to_id
        self._id_to_destination = id_to_destination
        self._context_length = context_length
        self._top_k = top_k
        self._device = device
        self._history_by_source: dict[Hashable, list[int]] = {}
        self._destinations: set[Hashable] = set(destination_to_id.keys())

    def fit(self, train: pd.DataFrame) -> NeuralSequenceCoverModel:
        # Fitting is performed by the training helper; this method exists to
        # satisfy the CoverModel protocol once the model is already trained.
        return self

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        history = self._history_by_source.setdefault(source, [0] * self._context_length)
        if previous_destination is not None:
            token = self._destination_to_id.get(str(previous_destination), 0)
            history.append(token)
            if len(history) > self._context_length:
                history.pop(0)

        inputs = torch.tensor([history], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self._model(inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Keep top_k most likely known destinations plus an unknown mass.
        k = min(self._top_k, len(probabilities))
        top_indices = np.argpartition(probabilities, -k)[-k:]
        top_indices = top_indices[np.argsort(-probabilities[top_indices])]

        candidates: list[Candidate] = []
        mass = 0.0
        for idx in top_indices:
            idx = int(idx)
            prob = float(probabilities[idx])
            if idx == 0:
                action = UNKNOWN_DESTINATION
            else:
                action = self._id_to_destination.get(idx, UNKNOWN_DESTINATION)
            if prob > 0:
                candidates.append(Candidate(action, prob))
                mass += prob

        if mass <= 0:
            # Fallback to uniform over training destinations.
            uniform = 1.0 / max(1, len(self._destinations))
            candidates = [Candidate(dest, uniform) for dest in list(self._destinations)[:k]]
            mass = sum(c.probability for c in candidates)

        return [Candidate(c.action, c.probability / mass) for c in candidates]

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        # Any source with a non-empty history is considered "seen".
        return source in self._history_by_source or previous_destination in self._destinations

    @property
    def destinations(self) -> frozenset[Hashable]:
        return frozenset(self._destinations)


class TemporalGNNCoverModel:
    """TemporalGraphCoverModel adapted to the CoverModel protocol.

    The underlying model maintains a per-source memory via a GRUCell.  We keep
    that memory, the previous destination and the previous timestamp per source
    so that predictions remain causal.
    """

    def __init__(
        self,
        model: TemporalGraphCoverModel,
        source_to_id: dict[str, int],
        destination_to_id: dict[str, int],
        id_to_destination: dict[int, str],
        *,
        top_k: int = 32,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._source_to_id = source_to_id
        self._destination_to_id = destination_to_id
        self._id_to_destination = id_to_destination
        self._top_k = top_k
        self._device = device
        self._memory_by_source: dict[Hashable, torch.Tensor] = {}
        self._previous_by_source: dict[Hashable, Hashable] = {}
        self._previous_time_by_source: dict[Hashable, float] = {}
        self._destinations: set[Hashable] = set(destination_to_id.keys())
        self._zero_memory: torch.Tensor | None = None

    def fit(self, train: pd.DataFrame) -> TemporalGNNCoverModel:
        return self

    def _zero(self) -> torch.Tensor:
        if self._zero_memory is None:
            hidden_size = self._model.memory_gru.hidden_size
            self._zero_memory = torch.zeros(1, hidden_size, device=self._device)
        return self._zero_memory

    def candidate_distribution(
        self,
        source: Hashable,
        previous_destination: Hashable | None,
    ) -> list[Candidate]:
        source_key = str(source)
        source_id = self._source_to_id.get(source_key, 0)
        previous_id = 0
        if previous_destination is not None:
            previous_id = self._destination_to_id.get(str(previous_destination), 0)
            self._previous_by_source[source] = previous_destination

        previous_time = self._previous_time_by_source.get(source, 0.0)
        current_time = previous_time  # Gap is relative; we use the stored timestamp externally.
        gap = math.log1p(max(0.0, current_time - previous_time))

        source_tensor = torch.tensor([source_id], dtype=torch.long, device=self._device)
        previous_tensor = torch.tensor([previous_id], dtype=torch.long, device=self._device)
        gap_tensor = torch.tensor([gap], dtype=torch.float32, device=self._device)
        memory = self._memory_by_source.get(source, self._zero())

        with torch.no_grad():
            logits, new_memory = self._model(source_tensor, previous_tensor, gap_tensor, memory)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        self._memory_by_source[source] = new_memory

        k = min(self._top_k, len(probabilities))
        top_indices = np.argpartition(probabilities, -k)[-k:]
        top_indices = top_indices[np.argsort(-probabilities[top_indices])]

        candidates: list[Candidate] = []
        mass = 0.0
        for idx in top_indices:
            idx = int(idx)
            prob = float(probabilities[idx])
            if idx == 0:
                action = UNKNOWN_DESTINATION
            else:
                action = self._id_to_destination.get(idx, UNKNOWN_DESTINATION)
            if prob > 0:
                candidates.append(Candidate(action, prob))
                mass += prob

        if mass <= 0:
            uniform = 1.0 / max(1, len(self._destinations))
            candidates = [Candidate(dest, uniform) for dest in list(self._destinations)[:k]]
            mass = sum(c.probability for c in candidates)

        return [Candidate(c.action, c.probability / mass) for c in candidates]

    def update_timestamp(self, source: Hashable, timestamp: float) -> None:
        """Must be called after each event to keep inter-event gaps causal."""
        self._previous_time_by_source[source] = float(timestamp)

    def has_context(self, source: Hashable, previous_destination: Hashable | None) -> bool:
        return source in self._memory_by_source or source in self._source_to_id

    @property
    def destinations(self) -> frozenset[Hashable]:
        return frozenset(self._destinations)


def train_neural_sequence_cover_model(
    frame: pd.DataFrame,
    *,
    kind: str,
    context_length: int,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    top_k: int = 32,
    device: str = "cpu",
) -> NeuralSequenceCoverModel:
    """Train a GRU or Transformer cover model and return it wrapped as CoverModel."""
    from models.neural_sequence import _examples as ns_examples
    from models.neural_sequence import _tensor_dataset as ns_tensor_dataset
    from models.neural_sequence import _vocabulary as ns_vocabulary

    torch.manual_seed(seed)
    np.random.seed(seed)
    train = frame.loc[frame["split"] == "train"]
    destination_to_id = ns_vocabulary(train["destination"])
    model = NeuralSequenceModel(
        vocabulary_size=len(destination_to_id) + 1,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
        kind=kind,  # type: ignore[arg-type]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = ns_tensor_dataset(train, destination_to_id, context_length)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

    model.eval()
    id_to_destination = {index: destination for destination, index in destination_to_id.items()}
    return NeuralSequenceCoverModel(
        model,
        destination_to_id,
        id_to_destination,
        context_length=context_length,
        top_k=top_k,
        device=device,
    )


def train_temporal_gnn_cover_model(
    frame: pd.DataFrame,
    *,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int = 5,
    dropout: float = 0.2,
    top_k: int = 32,
    device: str = "cpu",
) -> TemporalGNNCoverModel:
    """Train the custom temporal-graph cover model and return it wrapped as CoverModel."""
    from models.temporal_gnn import _calibrate_temperature, _examples as gnn_examples
    from models.temporal_gnn import _tensor_dataset as gnn_tensor_dataset
    from models.temporal_gnn import _vocabulary as gnn_vocabulary

    torch.manual_seed(seed)
    np.random.seed(seed)
    train = frame.loc[frame["split"] == "train"]
    source_to_id = gnn_vocabulary(pd.concat([train["source"], train["destination"]], ignore_index=True))
    destination_to_id = gnn_vocabulary(train["destination"])
    model = TemporalGraphCoverModel(
        sources=len(source_to_id) + 1,
        destinations=len(destination_to_id) + 1,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_dataset = gnn_tensor_dataset(train, source_to_id, destination_to_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    best_val_nll = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for sources, previous, gaps, targets in train_loader:
            optimizer.zero_grad()
            logits, _ = model(sources, previous, gaps)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_sources, val_previous, val_gaps, val_targets, _ = gnn_examples(
            frame.loc[frame["split"] == "validation"], source_to_id, destination_to_id
        )
        with torch.no_grad():
            val_logits, _ = model(
                torch.from_numpy(val_sources),
                torch.from_numpy(val_previous),
                torch.from_numpy(val_gaps),
            )
            val_log_probs = torch.log_softmax(val_logits, dim=1)
            val_nll_nats = -val_log_probs[
                torch.arange(len(val_targets)), torch.from_numpy(val_targets)
            ].numpy()
        val_nll = float(np.mean(val_nll_nats))
        scheduler.step(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _calibrate_temperature(
        model,
        frame.loc[frame["split"] == "validation"],
        source_to_id,
        destination_to_id,
    )
    model.eval()

    id_to_destination = {index: destination for destination, index in destination_to_id.items()}
    return TemporalGNNCoverModel(
        model,
        source_to_id,
        destination_to_id,
        id_to_destination,
        top_k=top_k,
        device=device,
    )
