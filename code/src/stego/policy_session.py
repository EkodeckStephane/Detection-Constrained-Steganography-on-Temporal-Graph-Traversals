from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from controllers.fuzzy import ControlDecision, ControllerInputs, FuzzyRateController
from stego.causal_arithmetic import CausalArithmeticDecoder, CausalArithmeticEncoder
from stego.coding import Candidate


@dataclass(frozen=True)
class SessionEmission:
    action: Hashable
    decision: ControlDecision
    coder_advanced: bool
    committed_payload_bits: int
    sender_decoder_state_match: bool
    session_terminated: bool


def rate_limited_candidates(
    candidates: Sequence[Candidate],
    *,
    nominal_bits: int,
) -> list[Candidate]:
    """Deterministically limit arithmetic support from a public rate level.

    ``nominal_bits`` is a public controller rate level, not a claim that exactly
    that many bits are committed on the transition. The reported communication
    rate is always measured from the arithmetic coder's committed-prefix gain.
    """

    if nominal_bits < 1:
        raise ValueError("nominal_bits must be positive")
    if not candidates:
        raise ValueError("at least one candidate is required")
    limit = min(len(candidates), 1 << nominal_bits)
    selected = list(candidates[:limit])
    mass = sum(float(item.probability) for item in selected)
    if mass <= 0:
        raise ValueError("selected candidates must have positive probability mass")
    return [Candidate(item.action, float(item.probability) / mass) for item in selected]


class SynchronizedPolicySession:
    """Alice/Bob covert session with deterministic public control decisions.

    Both endpoints receive the same public controller inputs before the emitted
    action is chosen. Thus EMBED/COVER/PAUSE/STOP cannot depend on secret bits.
    In EMBED mode the public nominal rate deterministically defines the same
    arithmetic candidate support at both endpoints. Alice then selects the
    secret-dependent action; Bob observes that action and updates the identical
    arithmetic interval. COVER/PAUSE/STOP never advance the arithmetic coder.
    """

    def __init__(
        self,
        payload_bits: Sequence[int],
        *,
        sender_controller: FuzzyRateController,
        receiver_controller: FuzzyRateController,
        precision_bits: int = 128,
    ) -> None:
        if not payload_bits:
            raise ValueError("payload_bits must be non-empty")
        self._sender_controller = sender_controller
        self._receiver_controller = receiver_controller
        self._encoder = CausalArithmeticEncoder(payload_bits, precision_bits=precision_bits)
        self._decoder = CausalArithmeticDecoder(
            payload_length=len(payload_bits),
            precision_bits=precision_bits,
        )
        self._terminated = False

    @property
    def committed_payload_bits(self) -> int:
        return self._encoder.committed_prefix_bits

    @property
    def complete(self) -> bool:
        return self._encoder.complete

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def state_match(self) -> bool:
        return self._encoder.state == self._decoder.state

    @property
    def decoded_payload(self) -> list[int]:
        return self._decoder.decoded_payload()

    def emit(
        self,
        *,
        inputs: ControllerInputs,
        candidates: Sequence[Candidate],
        cover_action: Hashable | None = None,
        admissible_actions: frozenset[Hashable] | set[Hashable] | None = None,
    ) -> SessionEmission:
        if not candidates:
            raise ValueError("current public state has no candidate continuation")

        if self._terminated or self.complete:
            decision = ControlDecision("COVER", 0, 0.0, 1.0)
            return self._cover_emission(
                decision=decision,
                cover_action=cover_action,
                admissible_actions=admissible_actions,
            )

        alice = self._sender_controller.decide(inputs)
        bob = self._receiver_controller.decide(inputs)
        if alice != bob:
            raise AssertionError("Alice and Bob derived different public control decisions")
        decision = self._effective_decision(alice, candidates)

        if decision.mode == "EMBED":
            public_support = rate_limited_candidates(
                candidates,
                nominal_bits=decision.local_payload_bits,
            )
            emission = self._encoder.emit(public_support)
            self._decoder.observe(emission.action, public_support)
            if not self.state_match:
                raise AssertionError("arithmetic state diverged after emitted action")
            return SessionEmission(
                action=emission.action,
                decision=decision,
                coder_advanced=True,
                committed_payload_bits=self.committed_payload_bits,
                sender_decoder_state_match=True,
                session_terminated=False,
            )

        if decision.mode == "STOP":
            self._terminated = True
        return self._cover_emission(
            decision=decision,
            cover_action=cover_action,
            admissible_actions=admissible_actions,
        )

    def _effective_decision(
        self,
        decision: ControlDecision,
        candidates: Sequence[Candidate],
    ) -> ControlDecision:
        if decision.mode == "EMBED" and len(candidates) < 2:
            return ControlDecision(
                "COVER",
                0,
                decision.rate_score,
                max(decision.abstention_score, 1.0),
            )
        return decision

    def _cover_emission(
        self,
        *,
        decision: ControlDecision,
        cover_action: Hashable | None,
        admissible_actions: frozenset[Hashable] | set[Hashable] | None,
    ) -> SessionEmission:
        if cover_action is None:
            raise ValueError(f"cover_action is required for {decision.mode} mode")
        if admissible_actions is not None and cover_action not in admissible_actions:
            raise ValueError("cover action is not admissible in the current public state")
        return SessionEmission(
            action=cover_action,
            decision=decision,
            coder_advanced=False,
            committed_payload_bits=self.committed_payload_bits,
            sender_decoder_state_match=self.state_match,
            session_terminated=self._terminated,
        )
