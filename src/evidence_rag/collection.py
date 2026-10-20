from __future__ import annotations

from time import perf_counter
from typing import Protocol

from .reader import ReaderError
from .retrieval import BM25Retriever
from .schema import ABSTENTION, Cost, Passage, Proposal, PublicQuestion, Step, Trace


class Reader(Protocol):
    def read(
        self, question: PublicQuestion, evidence: tuple[Passage, ...]
    ) -> tuple[Proposal, int | None, int | None]: ...


def collect(
    question: PublicQuestion,
    passages: dict[str, Passage],
    reader: Reader,
    max_rounds: int,
    top_k: int,
) -> Trace:
    if max_rounds < 1 or top_k < 1:
        raise ValueError("round and retrieval limits must be positive")
    started = perf_counter()
    retriever = BM25Retriever(tuple(passages[pid] for pid in question.candidate_ids))
    index_ms = (perf_counter() - started) * 1000
    observed: dict[str, Passage] = {}
    query = question.text
    steps = []
    for number in range(1, max_rounds + 1):
        started = perf_counter()
        found = retriever.search(query, top_k, set(observed))
        retrieval_ms = (perf_counter() - started) * 1000
        observed.update({p.id: p for p in found})
        started = perf_counter()
        status = "ok"
        input_tokens = output_tokens = None
        try:
            proposal, input_tokens, output_tokens = reader.read(question, tuple(observed.values()))
        except ReaderError as error:
            proposal, status = ABSTENTION, "reader_error"
            input_tokens, output_tokens = error.input_tokens, error.output_tokens
        reader_ms = (perf_counter() - started) * 1000
        steps.append(
            Step(
                number,
                tuple(p.id for p in found),
                proposal,
                Cost(
                    1,
                    1,
                    input_tokens,
                    output_tokens,
                    retrieval_ms,
                    reader_ms,
                ),
                status,
            )
        )
        query = proposal.next_query or question.text
        if status != "ok" or len(observed) == len(question.candidate_ids):
            break
    return Trace(question.id, tuple(steps), index_ms)
