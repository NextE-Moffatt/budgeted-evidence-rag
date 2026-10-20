from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Cost, Dataset, Example, Passage, Proposal, PublicQuestion, Step, Trace, strings

SPLITS = ("train", "calibration", "validation", "test")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError("unsupported dataset schema")
    passages = {}
    for row in raw["passages"]:
        passage = Passage(**row)
        if any(not isinstance(v, str) or not v.strip() for v in asdict(passage).values()):
            raise ValueError("invalid passage")
        if passage.id in passages:
            raise ValueError("duplicate passage id")
        passages[passage.id] = passage
    examples = []
    ids = set()
    for row in raw["examples"]:
        qid = row["id"]
        if not isinstance(qid, str) or not qid or qid in ids:
            raise ValueError("invalid/duplicate question id")
        ids.add(qid)
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("missing question text")
        candidates = strings(row["candidate_ids"])
        gold = strings(row["gold_passage_ids"])
        answers = strings(row["answers"])
        if not candidates or not set(candidates) <= passages.keys():
            raise ValueError("invalid candidate ids")
        if not set(gold) <= set(candidates) or not answers:
            raise ValueError("invalid evaluation labels")
        if (
            row["split"] not in SPLITS
            or not isinstance(row["group_id"], str)
            or not row["group_id"].strip()
        ):
            raise ValueError("invalid split/group")
        examples.append(
            Example(
                PublicQuestion(qid, row["question"], candidates),
                answers,
                gold,
                row["group_id"],
                row["split"],
            )
        )
    if not examples:
        raise ValueError("empty dataset")
    return Dataset(passages, tuple(examples), raw["provenance"])


def load_traces(path: Path, dataset: Dataset) -> tuple[Trace, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1 or raw["dataset_sha256"] != file_sha(path=None, dataset=dataset):
        raise ValueError("trace schema/dataset fingerprint mismatch")
    examples = {e.question.id: e for e in dataset.examples}
    traces = []
    ids = set()
    for row in raw["traces"]:
        qid = row["question_id"]
        if qid not in examples or qid in ids:
            raise ValueError("unknown/duplicate trace question")
        ids.add(qid)
        seen: set[str] = set()
        steps = []
        for number, item in enumerate(row["steps"], 1):
            retrieved = strings(item["retrieved_ids"])
            if type(item["round"]) is not int or item["round"] != number or set(retrieved) & seen:
                raise ValueError("invalid round or repeated evidence")
            if not set(retrieved) <= set(examples[qid].question.candidate_ids):
                raise ValueError("retrieval outside candidate corpus")
            seen.update(retrieved)
            proposal = Proposal.parse(item["proposal"], seen)
            if item["status"] not in ("ok", "reader_error"):
                raise ValueError("invalid trace status")
            if item["status"] == "reader_error" and proposal.answer is not None:
                raise ValueError("failed state must abstain")
            cost = Cost(**item["cost"])
            if cost.reader_calls != 1 or cost.retrieval_calls != 1:
                raise ValueError("schema v1 requires one retrieval and reader attempt per round")
            steps.append(Step(number, retrieved, proposal, cost, item["status"]))
        index_ms = row.get("index_ms", 0.0)
        Cost(retrieval_ms=index_ms)
        if not steps:
            raise ValueError("empty trace")
        traces.append(Trace(qid, tuple(steps), index_ms))
    if not traces:
        raise ValueError("empty trace set")
    return tuple(traces)


def dataset_dict(dataset: Dataset) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provenance": dataset.provenance,
        "passages": [asdict(p) for p in sorted(dataset.passages.values(), key=lambda p: p.id)],
        "examples": [
            {
                "id": e.question.id,
                "question": e.question.text,
                "candidate_ids": list(e.question.candidate_ids),
                "answers": list(e.answers),
                "gold_passage_ids": list(e.gold_passage_ids),
                "group_id": e.group_id,
                "split": e.split,
            }
            for e in sorted(dataset.examples, key=lambda e: e.question.id)
        ],
    }


def file_sha(path: Path | None = None, *, dataset: Dataset | None = None) -> str:
    if dataset is not None:
        return digest(dataset_dict(dataset))
    if path is None:
        raise ValueError("path or dataset required")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trace_dict(traces: tuple[Trace, ...], dataset: Dataset) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_sha256": file_sha(dataset=dataset),
        "traces": [asdict(t) for t in traces],
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
