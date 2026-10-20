from __future__ import annotations

from dataclasses import replace

from .data import SPLITS, digest
from .schema import Dataset


def grouped_split(dataset: Dataset, seed: str) -> tuple[Dataset, dict]:
    """Connect questions through exact source identities before assigning partitions."""
    parents = {e.question.id: e.question.id for e in dataset.examples}

    def root(qid: str) -> str:
        while parents[qid] != qid:
            parents[qid] = parents[parents[qid]]
            qid = parents[qid]
        return qid

    owners: dict[str, str] = {}
    for example in sorted(dataset.examples, key=lambda e: e.question.id):
        for pid in example.question.candidate_ids:
            source = dataset.passages[pid].source_id
            if source in owners:
                left, right = root(example.question.id), root(owners[source])
                parents[max(left, right)] = min(left, right)
            else:
                owners[source] = example.question.id
    groups: dict[str, list[str]] = {}
    for qid in parents:
        groups.setdefault(root(qid), []).append(qid)
    assignments = {}
    for members in groups.values():
        group_id = digest(sorted(members))
        value = int(digest([seed, group_id])[:16], 16) / 2**64
        split = (
            "train"
            if value < 0.60
            else "calibration"
            if value < 0.75
            else "validation"
            if value < 0.85
            else "test"
        )
        for qid in members:
            assignments[qid] = (group_id, split)
    examples = tuple(
        replace(e, group_id=assignments[e.question.id][0], split=assignments[e.question.id][1])
        for e in dataset.examples
    )
    counts = {s: sum(e.split == s for e in examples) for s in SPLITS}
    audit = {
        "seed": seed,
        "method": "all-candidate-source connected components then hash 60/15/10/15",
        "questions": len(examples),
        "components": len(groups),
        "largest_component": max(map(len, groups.values())),
        "split_counts": counts,
        "empty_splits": [s for s, n in counts.items() if n == 0],
        "entity_disjointness_verified": False,
        "warning": "Do not relax groups silently; empty splits invalidate confirmatory use.",
    }
    return Dataset(dataset.passages, examples, dataset.provenance), audit


def audit_split_leakage(dataset: Dataset) -> None:
    owners: dict[tuple[str, str], str] = {}
    for example in dataset.examples:
        keys = [("group", example.group_id)] + [
            ("source", dataset.passages[pid].source_id) for pid in example.question.candidate_ids
        ]
        for key in keys:
            if key in owners and owners[key] != example.split:
                raise ValueError("group/source leakage across dataset partitions")
            owners[key] = example.split
