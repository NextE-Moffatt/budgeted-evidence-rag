from __future__ import annotations

from dataclasses import dataclass

from .schema import PublicState, probability


@dataclass(frozen=True)
class StopPolicy:
    """Simple research baselines, not published-method reproductions."""

    mode: str
    max_rounds: int
    threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "confidence", "sufficiency", "stability"):
            raise ValueError("unknown stop policy")
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        probability(self.threshold)

    @property
    def name(self) -> str:
        return f"{self.mode}-r{self.max_rounds}-t{self.threshold:g}"

    def stop(self, state: PublicState) -> bool:
        if state.round >= self.max_rounds:
            return True
        proposal = state.proposals[-1]
        if not proposal.answer:
            return False
        if self.mode in ("confidence", "sufficiency"):
            return getattr(proposal, self.mode) >= self.threshold
        if self.mode == "stability" and len(state.proposals) >= 2:
            previous = state.proposals[-2].answer
            return (
                previous is not None
                and previous.casefold().strip() == proposal.answer.casefold().strip()
            )
        return False
