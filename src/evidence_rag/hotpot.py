from __future__ import annotations

import json
from pathlib import Path

from .data import digest, file_sha
from .schema import Dataset, Example, Passage, PublicQuestion
from .splits import grouped_split


def convert_hotpot(
    path: Path, limit: int, seed: str, source_split: str = "unspecified"
) -> tuple[Dataset, dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected labelled HotpotQA array")
    qids = [row["_id"] for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate HotpotQA question id")
    selected = sorted(rows, key=lambda r: digest([seed, r["_id"]]))[:limit]
    passages = {}
    examples = []
    for row in selected:
        title_ids = {}
        lengths = {}
        for title, sentences in row["context"]:
            if title in title_ids or not title or not isinstance(sentences, list):
                raise ValueError("invalid/duplicate context title")
            if not sentences or any(not isinstance(s, str) for s in sentences):
                raise ValueError("invalid context sentences")
            text = "".join(sentences).strip()
            if not text:
                raise ValueError("empty context")
            pid = digest([title, text])
            passages[pid] = Passage(pid, title, text, title)
            title_ids[title] = pid
            lengths[title] = len(sentences)
        gold = set()
        for title, sentence_index in row["supporting_facts"]:
            if title not in title_ids or not 0 <= sentence_index < lengths[title]:
                raise ValueError("supporting fact not found in context")
            gold.add(title_ids[title])
        if not gold or not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError("missing gold labels")
        examples.append(
            Example(
                PublicQuestion(row["_id"], row["question"], tuple(title_ids.values())),
                (row["answer"],),
                tuple(sorted(gold)),
                row["_id"],
                "train",
            )
        )
    dataset = Dataset(
        passages,
        tuple(examples),
        {
            "kind": "benchmark",
            "dataset": "HotpotQA",
            "setting": "distractor_candidates",
            "source_url": "https://hotpotqa.github.io/",
            "license": "CC-BY-SA-4.0",
            "input_sha256": file_sha(path),
            "sample_seed": seed,
            "sample_limit": limit,
            "original_split": source_split,
            "support_metric": "passage recall only; sentence support/joint metrics not implemented",
        },
    )
    return grouped_split(dataset, seed)
