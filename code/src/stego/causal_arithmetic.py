from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

from stego.coding import Candidate


@dataclass(frozen=True)
class ArithmeticState:
    low: int
    high: int
    precision_bits: int

    @property
    def width(self) -> int:
        return self.high - self.low


@dataclass(frozen=True)
class ArithmeticEmission:
    action: Hashable
    committed_prefix_bits: int
    state: ArithmeticState


class CausalArithmeticEncoder:
    """Finite-precision arithmetic sender with no per-step width side channel.

    The payload length is a public session parameter. The encrypted payload bits
    select one point in a shared dyadic interval. At each causal state the
    encoder partitions the *current* interval according to the current candidate
    distribution, emits the action whose subinterval contains that point, and
    updates its arithmetic state. The next candidate distribution may therefore
    depend on the action just emitted.

    Bob needs only the same public payload length, precision, causal cover state,
    and emitted actions; Alice never sends ``bits_consumed`` metadata.
    """

    def __init__(
        self,
        payload_bits: Sequence[int],
        *,
        precision_bits: int = 128,
    ) -> None:
        if not payload_bits:
            raise ValueError("payload_bits must be non-empty")
        if precision_bits < len(payload_bits):
            raise ValueError("precision_bits must cover the public payload length")
        if precision_bits < 2:
            raise ValueError("precision_bits must be at least two")
        self.payload_length = len(payload_bits)
        self.precision_bits = int(precision_bits)
        self._target = _bits_to_int(
            [*payload_bits, *([0] * (precision_bits - len(payload_bits)))]
        )
        self._low = 0
        self._high = 1 << precision_bits

    @property
    def state(self) -> ArithmeticState:
        return ArithmeticState(self._low, self._high, self.precision_bits)

    @property
    def committed_prefix_bits(self) -> int:
        return min(
            self.payload_length,
            _common_prefix_bits(self._low, self._high, self.precision_bits),
        )

    @property
    def complete(self) -> bool:
        return self.committed_prefix_bits >= self.payload_length

    def emit(self, candidates: Iterable[Candidate]) -> ArithmeticEmission:
        if self.complete:
            raise ValueError("payload is already fully committed")
        intervals = _partition_interval(_rank_candidates(candidates), self._low, self._high)
        selected = next(
            (item for item in intervals if item[1] <= self._target < item[2]),
            None,
        )
        if selected is None:
            raise ValueError("finite-precision interval cannot encode the payload point")
        action, self._low, self._high = selected
        return ArithmeticEmission(
            action=action,
            committed_prefix_bits=self.committed_prefix_bits,
            state=self.state,
        )


class CausalArithmeticDecoder:
    """Receiver counterpart for :class:`CausalArithmeticEncoder`.

    The decoder consumes emitted actions under the candidate distribution it
    reconstructs from its own causal history. It never receives a local coding
    width from Alice. A mismatch in candidate sets or ordering is therefore
    exposed immediately as a decoding error rather than hidden by oracle
    metadata.
    """

    def __init__(
        self,
        *,
        payload_length: int,
        precision_bits: int = 128,
    ) -> None:
        if payload_length < 1:
            raise ValueError("payload_length must be positive")
        if precision_bits < payload_length:
            raise ValueError("precision_bits must cover the public payload length")
        if precision_bits < 2:
            raise ValueError("precision_bits must be at least two")
        self.payload_length = int(payload_length)
        self.precision_bits = int(precision_bits)
        self._low = 0
        self._high = 1 << precision_bits

    @property
    def state(self) -> ArithmeticState:
        return ArithmeticState(self._low, self._high, self.precision_bits)

    @property
    def committed_prefix_bits(self) -> int:
        return min(
            self.payload_length,
            _common_prefix_bits(self._low, self._high, self.precision_bits),
        )

    @property
    def complete(self) -> bool:
        return self.committed_prefix_bits >= self.payload_length

    def observe(self, action: Hashable, candidates: Iterable[Candidate]) -> ArithmeticState:
        intervals = _partition_interval(_rank_candidates(candidates), self._low, self._high)
        selected = next((item for item in intervals if item[0] == action), None)
        if selected is None:
            raise ValueError("emitted action is not decodable under Bob's causal candidate set")
        _, self._low, self._high = selected
        return self.state

    def decoded_prefix(self) -> list[int]:
        width = self.committed_prefix_bits
        if width <= 0:
            return []
        prefix = self._low >> (self.precision_bits - width)
        return _int_to_bits(prefix, width)

    def decoded_payload(self) -> list[int]:
        if not self.complete:
            raise ValueError("payload is not yet uniquely determined")
        return self.decoded_prefix()[: self.payload_length]


def _rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    ranked = list(candidates)
    if not ranked:
        raise ValueError("at least one candidate is required")
    if any(candidate.probability < 0 for candidate in ranked):
        raise ValueError("candidate probabilities must be non-negative")
    total = sum(candidate.probability for candidate in ranked)
    if total <= 0:
        raise ValueError("candidate probabilities must have positive mass")
    return sorted(ranked, key=lambda item: (-item.probability, repr(item.action)))


def _partition_interval(
    candidates: Sequence[Candidate],
    low: int,
    high: int,
) -> list[tuple[Hashable, int, int]]:
    width = high - low
    if width <= 0:
        raise ValueError("interval width must be positive")
    total_probability = sum(candidate.probability for candidate in candidates)
    raw_widths = {
        candidate.action: candidate.probability / total_probability * width
        for candidate in candidates
    }
    integer_widths = {
        action: int(value)
        for action, value in raw_widths.items()
    }
    missing = width - sum(integer_widths.values())
    remainders = {
        action: raw_widths[action] - integer_widths[action]
        for action in raw_widths
    }
    sorted_actions = sorted(
        raw_widths,
        key=lambda action: (remainders[action], raw_widths[action], repr(action)),
        reverse=True,
    )
    for action in sorted_actions[:missing]:
        integer_widths[action] += 1

    intervals: list[tuple[Hashable, int, int]] = []
    cursor = low
    for candidate in candidates:
        action_width = integer_widths[candidate.action]
        if action_width > 0:
            intervals.append((candidate.action, cursor, cursor + action_width))
            cursor += action_width
    if cursor != high:
        raise AssertionError("arithmetic partition does not cover the parent interval")
    return intervals


def _common_prefix_bits(low: int, high: int, precision_bits: int) -> int:
    if low >= high:
        raise ValueError("invalid arithmetic interval")
    upper_inclusive = high - 1
    for width in range(precision_bits + 1):
        shift = precision_bits - width
        if low >> shift != upper_inclusive >> shift:
            return width - 1
    return precision_bits


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must be 0 or 1")
        value = (value << 1) | bit
    return value


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
