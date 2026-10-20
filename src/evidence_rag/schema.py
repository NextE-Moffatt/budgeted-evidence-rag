from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("score must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score must be finite and between zero and one")
    return float(value)


def strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError("expected a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError("duplicate identifiers/values")
    return tuple(value)


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    source_id: str


@dataclass(frozen=True)
class PublicQuestion:
    id: str
    text: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class Example:
    question: PublicQuestion
    answers: tuple[str, ...]
    gold_passage_ids: tuple[str, ...]
    group_id: str
    split: str


@dataclass(frozen=True)
class Dataset:
    passages: dict[str, Passage]
    examples: tuple[Example, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Proposal:
    answer: str | None
    citations: tuple[str, ...]
    confidence: float
    sufficiency: float
    next_query: str

    @classmethod
    def parse(cls, raw: dict[str, Any], evidence_ids: set[str]) -> Proposal:
        answer = raw["answer"]
        if answer is not None and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("answer must be nonempty text or null")
        citations = strings(raw["citations"])
        if not set(citations) <= evidence_ids:
            raise ValueError("citation outside observed evidence")
        if bool(answer) != bool(citations):
            raise ValueError("answers require citations; abstentions cannot cite")
        query = raw["next_query"]
        if not isinstance(query, str):
            raise ValueError("next_query must be text")
        return cls(
            answer.strip() if answer else None,
            citations,
            probability(raw["confidence"]),
            probability(raw["sufficiency"]),
            query.strip(),
        )


ABSTENTION = Proposal(None, (), 0.0, 0.0, "")


@dataclass(frozen=True)
class Cost:
    retrieval_calls: int = 0
    reader_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    retrieval_ms: float = 0.0
    reader_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("retrieval_calls", "reader_calls", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is None and name.endswith("tokens"):
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid cost field: {name}")
        for name in ("retrieval_ms", "reader_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"invalid latency: {name}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid latency: {name}")


@dataclass(frozen=True)
class Step:
    round: int
    retrieved_ids: tuple[str, ...]
    proposal: Proposal
    cost: Cost
    status: str = "ok"


@dataclass(frozen=True)
class Trace:
    question_id: str
    steps: tuple[Step, ...]
    index_ms: float = 0.0


@dataclass(frozen=True)
class PublicState:
    """Only the observed prefix; no labels, partition, or future steps."""

    question: PublicQuestion
    round: int
    evidence_ids: tuple[str, ...]
    proposals: tuple[Proposal, ...]


def sum_cost(costs: list[Cost]) -> dict[str, int | float | None]:
    return {
        "retrieval_calls": sum(c.retrieval_calls for c in costs),
        "reader_calls": sum(c.reader_calls for c in costs),
        "input_tokens": (
            sum(c.input_tokens for c in costs)
            if all(c.input_tokens is not None for c in costs)
            else None
        ),
        "output_tokens": (
            sum(c.output_tokens for c in costs)
            if all(c.output_tokens is not None for c in costs)
            else None
        ),
        "retrieval_ms": sum(c.retrieval_ms for c in costs),
        "reader_ms": sum(c.reader_ms for c in costs),
    }
