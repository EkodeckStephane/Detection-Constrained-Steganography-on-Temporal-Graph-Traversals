from __future__ import annotations

import hashlib
import math
import os
import zlib
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class Candidate:
    action: Hashable
    probability: float


@dataclass(frozen=True)
class EncodedAction:
    action: Hashable
    bits_consumed: int
    local_total_variation: float
    local_kl_bits: float


@dataclass(frozen=True)
class ProtectedMessage:
    ciphertext: bytes
    tag: bytes
    nonce: bytes
    length: int
    crc32: int


@dataclass(frozen=True)
class ArithmeticTrace:
    actions: list[Hashable]
    recoverable_bits: int
    interval_low: int
    interval_high: int
    precision_bits: int


class AuthenticationError(ValueError):
    pass


def protect_message(
    message: bytes,
    key: bytes,
    *,
    associated_data: bytes = b"",
    nonce: bytes | None = None,
) -> ProtectedMessage:
    aes_key = _aes_key(key)
    crc = zlib.crc32(message) & 0xFFFFFFFF
    header = len(message).to_bytes(8, "big") + crc.to_bytes(4, "big")
    nonce = nonce or os.urandom(12)
    if len(nonce) != 12:
        raise ValueError("AES-GCM nonce must be 12 bytes")
    protected = AESGCM(aes_key).encrypt(nonce, message, associated_data + header)
    return ProtectedMessage(
        ciphertext=protected[:-16],
        tag=protected[-16:],
        nonce=nonce,
        length=len(message),
        crc32=crc,
    )


def recover_message(
    protected: ProtectedMessage,
    key: bytes,
    *,
    associated_data: bytes = b"",
) -> bytes:
    aes_key = _aes_key(key)
    header = protected.length.to_bytes(8, "big") + protected.crc32.to_bytes(4, "big")
    try:
        message = AESGCM(aes_key).decrypt(
            protected.nonce,
            protected.ciphertext + protected.tag,
            associated_data + header,
        )
    except InvalidTag as exc:
        raise AuthenticationError("message authentication failed") from exc
    if len(message) != protected.length:
        raise AuthenticationError("message length mismatch")
    if (zlib.crc32(message) & 0xFFFFFFFF) != protected.crc32:
        raise AuthenticationError("message CRC mismatch")
    return message


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")
    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            if bit not in (0, 1):
                raise ValueError("bits must be 0 or 1")
            value = (value << 1) | bit
        output.append(value)
    return bytes(output)


def repetition3_encode(bits: Sequence[int]) -> list[int]:
    return [bit for source in bits for bit in (source, source, source)]


def repetition3_decode(bits: Sequence[int]) -> list[int]:
    if len(bits) % 3 != 0:
        raise ValueError("repetition-3 bitstream length must be a multiple of 3")
    decoded = []
    for offset in range(0, len(bits), 3):
        block = bits[offset : offset + 3]
        decoded.append(1 if sum(block) >= 2 else 0)
    return decoded


def reed_solomon_encode(bits: Sequence[int], *, nsym: int) -> list[int]:
    """Encode a bitstream with a Reed--Solomon block code.

    ``nsym`` is the number of error-correction symbols per block. The codec
    uses the ``reedsolo`` library with 8-bit symbols. The input bit length
    must be a multiple of 8 and the block payload length (in bytes) must be
    at most ``255 - nsym``.
    """
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise ImportError("reedsolo is required for Reed-Solomon encoding") from exc
    data = bits_to_bytes(bits)
    rsc = RSCodec(nsym=nsym)
    encoded = rsc.encode(data)
    return bytes_to_bits(encoded)


def reed_solomon_decode(bits: Sequence[int], *, nsym: int) -> list[int]:
    """Decode a Reed--Solomon encoded bitstream.

    Returns the original message bits. Raises ``ValueError`` if the corruption
    exceeds the correction capability of the code.
    """
    try:
        from reedsolo import RSCodec, ReedSolomonError
    except ImportError as exc:
        raise ImportError("reedsolo is required for Reed-Solomon decoding") from exc
    data = bits_to_bytes(bits)
    rsc = RSCodec(nsym=nsym)
    try:
        decoded, _, _ = rsc.decode(data)
    except ReedSolomonError as exc:
        raise ValueError(f"Reed-Solomon decoding failed: {exc}") from exc
    return bytes_to_bits(decoded)


def encode_next_action(
    bits: Sequence[int],
    candidates: Iterable[Candidate],
    *,
    max_bits: int,
) -> EncodedAction:
    ranked = _rank_candidates(candidates)
    capacity = min(max_bits, _usable_width(ranked))
    if capacity <= 0:
        return EncodedAction(
            action=ranked[0].action,
            bits_consumed=0,
            local_total_variation=0.0,
            local_kl_bits=0.0,
        )
    consumed = min(capacity, len(bits))
    value = _bits_to_int(bits[:consumed]) if consumed else 0
    target = _renormalized_probabilities(ranked)
    intervals = _quantized_intervals(ranked, consumed)
    selected = next(
        action
        for action, lower, upper in intervals
        if lower <= value < upper
    )
    induced = _induced_probabilities(intervals, consumed)
    tv = _total_variation(induced, target)
    kl = _kl_bits(induced, target)
    return EncodedAction(
        action=selected,
        bits_consumed=consumed,
        local_total_variation=tv,
        local_kl_bits=kl,
    )


def decode_action_bits(
    action: Hashable,
    candidates: Iterable[Candidate],
    *,
    bits_consumed: int,
) -> list[int]:
    if bits_consumed == 0:
        return []
    ranked = _rank_candidates(candidates)
    intervals = _quantized_intervals(ranked, bits_consumed)
    match = next((item for item in intervals if item[0] == action), None)
    if match is None:
        raise ValueError("action is not decodable under the shared candidate set")
    _, lower, _ = match
    return _int_to_bits(lower, bits_consumed)


def encode_next_action_range(
    bits: Sequence[int],
    candidates: Iterable[Candidate],
    *,
    max_bits: int,
) -> EncodedAction:
    """Cumulative probability encoder with exact local decodability.

    Partitions the dyadic space into intervals whose widths are proportional
    to the cover-model probabilities. Because this local codec decodes bits
    from the emitted action alone, it only claims a consumed prefix when the
    selected action has a singleton integer interval. Wider intervals are
    distributionally useful for sampling, but ambiguous for this per-step
    decoder; those cases fall back to a shorter prefix or to a cover action.
    """
    ranked = _rank_candidates(candidates)
    capacity = min(max_bits, _usable_width(ranked))
    target = _renormalized_probabilities(ranked)
    for consumed in range(min(capacity, len(bits)), 0, -1):
        intervals = _cumulative_intervals(ranked, consumed)
        value = _bits_to_int(bits[:consumed])
        selected = next(
            (item for item in intervals if item[1] <= value < item[2]),
            None,
        )
        if selected is None:
            continue
        action, lower, upper = selected
        if upper - lower != 1:
            continue
        induced = _induced_probabilities(intervals, consumed)
        tv = _total_variation(induced, target)
        kl = _kl_bits(induced, target)
        return EncodedAction(
            action=action,
            bits_consumed=consumed,
            local_total_variation=tv,
            local_kl_bits=kl,
        )
    return EncodedAction(
        action=ranked[0].action,
        bits_consumed=0,
        local_total_variation=0.0,
        local_kl_bits=0.0,
    )


def decode_action_bits_range(
    action: Hashable,
    candidates: Iterable[Candidate],
    *,
    bits_consumed: int,
) -> list[int]:
    if bits_consumed == 0:
        return []
    ranked = _rank_candidates(candidates)
    intervals = _cumulative_intervals(ranked, bits_consumed)
    match = next((item for item in intervals if item[0] == action), None)
    if match is None:
        raise ValueError("action is not decodable under the shared candidate set")
    _, lower, _ = match
    return _int_to_bits(lower, bits_consumed)


def encode_trace(
    bits: Sequence[int],
    candidate_steps: Sequence[Sequence[Candidate]],
    *,
    max_bits_per_transition: int,
) -> tuple[list[Hashable], list[int], float, float]:
    actions = []
    widths = []
    offset = 0
    tv_distortion = 0.0
    kl_distortion = 0.0
    for candidates in candidate_steps:
        encoded = encode_next_action(
            bits[offset:],
            candidates,
            max_bits=max_bits_per_transition,
        )
        actions.append(encoded.action)
        widths.append(encoded.bits_consumed)
        offset += encoded.bits_consumed
        tv_distortion += encoded.local_total_variation
        kl_distortion += encoded.local_kl_bits
        if offset >= len(bits):
            break
    if offset < len(bits):
        raise ValueError("candidate sequence exhausted before all bits were encoded")
    return actions, widths, tv_distortion, kl_distortion


def decode_trace(
    actions: Sequence[Hashable],
    widths: Sequence[int],
    candidate_steps: Sequence[Sequence[Candidate]],
) -> list[int]:
    if len(actions) != len(widths):
        raise ValueError("actions and widths must have the same length")
    bits: list[int] = []
    for action, bits_consumed, candidates in zip(actions, widths, candidate_steps):
        bits.extend(decode_action_bits(action, candidates, bits_consumed=bits_consumed))
    return bits


def encode_trace_arithmetic(
    bits: Sequence[int],
    candidate_steps: Sequence[Sequence[Candidate]],
    *,
    precision_bits: int = 64,
) -> ArithmeticTrace:
    if precision_bits < 2:
        raise ValueError("precision_bits must be at least two")
    if len(bits) > precision_bits:
        raise ValueError("bitstream exceeds the configured arithmetic precision")
    total = 1 << precision_bits
    message_value = _bits_to_int([*bits, *([0] * (precision_bits - len(bits)))])
    low = 0
    high = total
    actions: list[Hashable] = []
    for candidates in candidate_steps:
        if high - low <= 1:
            break
        intervals = _partition_interval(_rank_candidates(candidates), low, high)
        selected = next(
            (item for item in intervals if item[1] <= message_value < item[2]),
            None,
        )
        if selected is None:
            break
        action, low, high = selected
        actions.append(action)
    recoverable = _common_prefix_bits(low, high, precision_bits)
    return ArithmeticTrace(
        actions=actions,
        recoverable_bits=min(recoverable, len(bits)),
        interval_low=low,
        interval_high=high,
        precision_bits=precision_bits,
    )


def decode_trace_arithmetic_prefix(
    actions: Sequence[Hashable],
    candidate_steps: Sequence[Sequence[Candidate]],
    *,
    precision_bits: int = 64,
) -> list[int]:
    if precision_bits < 2:
        raise ValueError("precision_bits must be at least two")
    total = 1 << precision_bits
    low = 0
    high = total
    for action, candidates in zip(actions, candidate_steps):
        intervals = _partition_interval(_rank_candidates(candidates), low, high)
        selected = next((item for item in intervals if item[0] == action), None)
        if selected is None:
            raise ValueError("action is not decodable under the shared candidate set")
        _, low, high = selected
    width = _common_prefix_bits(low, high, precision_bits)
    if width == 0:
        return []
    prefix = low >> (precision_bits - width)
    return _int_to_bits(prefix, width)


def _aes_key(key: bytes) -> bytes:
    if not key:
        raise ValueError("key must not be empty")
    if len(key) in (16, 24, 32):
        return key
    return hashlib.sha256(key).digest()


def _rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    ranked = list(candidates)
    if not ranked:
        raise ValueError("at least one candidate is required")
    if any(candidate.probability < 0 for candidate in ranked):
        raise ValueError("candidate probabilities must be non-negative")
    if sum(candidate.probability for candidate in ranked) <= 0:
        raise ValueError("candidate probabilities must have positive mass")
    return sorted(ranked, key=lambda item: (-item.probability, repr(item.action)))


def _renormalized_probabilities(candidates: Sequence[Candidate]) -> dict[Hashable, float]:
    total = sum(candidate.probability for candidate in candidates)
    return {candidate.action: candidate.probability / total for candidate in candidates}


def _usable_width(candidates: Sequence[Candidate]) -> int:
    return max(0, int(math.log2(len(candidates))))


def _quantized_intervals(
    candidates: Sequence[Candidate],
    width: int,
) -> list[tuple[Hashable, int, int]]:
    if width < 1:
        return [(candidates[0].action, 0, 1)]
    total_mass = 2**width
    intervals = []
    for index, candidate in enumerate(candidates[:total_mass]):
        intervals.append((candidate.action, index, index + 1))
    return intervals


def _cumulative_intervals(
    candidates: Sequence[Candidate],
    width: int,
) -> list[tuple[Hashable, int, int]]:
    """Map candidates to integer intervals proportional to their probabilities.

    The interval width for each candidate is ``round(p * 2**width)``, with
    leftover mass assigned to the most probable candidate to ensure the total
    is exactly ``2**width``. This preserves the cover distribution at the
    resolution of ``width`` bits.
    """
    if width < 1:
        return [(candidates[0].action, 0, 1)]
    total_mass = 2**width
    probabilities = _renormalized_probabilities(candidates)
    widths = {
        candidate.action: probabilities[candidate.action] * total_mass
        for candidate in candidates
    }
    # Assign integer widths using largest-remainder rounding.
    integer_widths = {action: int(math.floor(w)) for action, w in widths.items()}
    remainder = {
        action: widths[action] - integer_widths[action]
        for action in widths
    }
    missing = total_mass - sum(integer_widths.values())
    sorted_actions = sorted(
        widths,
        key=lambda action: (remainder[action], widths[action]),
        reverse=True,
    )
    for action in sorted_actions[:missing]:
        integer_widths[action] += 1

    intervals: list[tuple[Hashable, int, int]] = []
    cursor = 0
    for candidate in candidates:
        action = candidate.action
        width_action = integer_widths[action]
        if width_action > 0:
            intervals.append((action, cursor, cursor + width_action))
            cursor += width_action
    return intervals


def _induced_probabilities(
    intervals: Sequence[tuple[Hashable, int, int]],
    width: int,
) -> dict[Hashable, float]:
    total = 2**width
    return {action: (upper - lower) / total for action, lower, upper in intervals}


def _partition_interval(
    candidates: Sequence[Candidate],
    low: int,
    high: int,
) -> list[tuple[Hashable, int, int]]:
    width = high - low
    if width <= 0:
        raise ValueError("interval width must be positive")
    probabilities = _renormalized_probabilities(candidates)
    raw_widths = {
        candidate.action: probabilities[candidate.action] * width
        for candidate in candidates
    }
    integer_widths = {
        action: int(math.floor(value))
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
        action = candidate.action
        action_width = integer_widths[action]
        if action_width > 0:
            intervals.append((action, cursor, cursor + action_width))
            cursor += action_width
    return intervals


def _common_prefix_bits(low: int, high: int, precision_bits: int) -> int:
    if low >= high:
        raise ValueError("invalid arithmetic interval")
    upper_inclusive = high - 1
    for width in range(precision_bits + 1):
        if low >> (precision_bits - width) != upper_inclusive >> (precision_bits - width):
            return width - 1
    return precision_bits


def _total_variation(
    induced: dict[Hashable, float],
    target: dict[Hashable, float],
) -> float:
    actions = set(induced) | set(target)
    return 0.5 * sum(abs(induced.get(action, 0.0) - target.get(action, 0.0)) for action in actions)


def _kl_bits(
    induced: dict[Hashable, float],
    target: dict[Hashable, float],
) -> float:
    value = 0.0
    for action, probability in induced.items():
        if probability > 0:
            value += probability * math.log2(probability / target[action])
    return value


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must be 0 or 1")
        value = (value << 1) | bit
    return value


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
